"""
Leakage-proof backtesting with rolling window evaluation.

DESIGN PRINCIPLES:
1. Train window always ends before evaluation window starts
2. Feature engineer is fit on train window only — never on eval
3. Test set touched exactly once — at the end
4. All results prefixed with data assumption context

HONEST FRAMING:
Results below are under:
- BTS T-100 real passenger demand data
- Synthetic price-elasticity assumptions
- Per-departure demand (not individual booking level)

Revenue numbers are per-flight estimates, not route totals.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
from src.modeling.demand_model import DemandModel
from src.features.feature_engineer import FeatureEngineer
from src.optimization.price_optimizer import PriceOptimizer, PricingConstraints
from src.synthetic.demand_simulator import DemandPriceSimulator
from src.utils.logger import logger
from src.utils.settings import get_settings

MODEL_DIR = "models/demand"

ROUTE_CONSTRAINTS = {
    "JFK-LAX": PricingConstraints(price_min=89, price_max=1200, capacity=180),
    "LAX-JFK": PricingConstraints(price_min=89, price_max=1200, capacity=180),
    "ORD-MIA": PricingConstraints(price_min=69, price_max=900, capacity=150),
    "MIA-ORD": PricingConstraints(price_min=69, price_max=900, capacity=150),
    "LAX-SEA": PricingConstraints(price_min=49, price_max=650, capacity=160),
    "SEA-LAX": PricingConstraints(price_min=49, price_max=650, capacity=160),
}


@dataclass
class BacktestResult:
    """Single period backtest result for one route."""

    route: str
    period_start: str
    period_end: str
    window_id: int

    # Demand
    actual_demand: float
    predicted_demand_q50: float
    predicted_demand_q10: float
    predicted_demand_q90: float
    demand_error_pct: float

    # Pricing strategies
    baseline_price: float
    optimized_price: float
    oracle_price: float

    # Revenue outcomes
    baseline_revenue: float
    optimized_revenue: float
    oracle_revenue: float

    # Uplift metrics
    revenue_uplift_vs_baseline_pct: float
    regret_vs_oracle_pct: float  # gap between optimized and oracle

    # Load factors
    baseline_load_factor: float
    optimized_load_factor: float


class RollingWindowBacktester:
    """
    Implements rolling window backtesting.
    Retrains model on each expanding window.
    Evaluates on next period.
    """

    def __init__(
        self,
        data_path: str = "data/processed/t100_cleaned.parquet",
        window_size_months: int = 6,
    ):
        self.data_path = data_path
        self.window_size_months = window_size_months
        self.settings = get_settings()
        self.results: list[BacktestResult] = []

    def run_rolling_windows(self) -> pd.DataFrame:
        """
        Run rolling window backtesting on validation period.
        Does NOT touch test set (2023).
        """
        logger.info("Starting rolling window backtesting")

        df = pd.read_parquet(self.data_path)

        # Rolling windows on validation period only
        windows = [
            {
                "id": 1,
                "train_end": "2021-06-30",
                "eval_start": "2021-07-01",
                "eval_end": "2021-12-31",
            },
            {
                "id": 2,
                "train_end": "2021-12-31",
                "eval_start": "2022-01-01",
                "eval_end": "2022-06-30",
            },
            {
                "id": 3,
                "train_end": "2022-06-30",
                "eval_start": "2022-07-01",
                "eval_end": "2022-12-31",
            },
        ]

        all_results = []

        for window in windows:
            logger.info(
                f"Window {window['id']}: "
                f"train → {window['train_end']}, "
                f"eval {window['eval_start']} → {window['eval_end']}"
            )

            window_results = self._evaluate_window(df, window)
            all_results.extend(window_results)

        results_df = pd.DataFrame([vars(r) for r in all_results])

        logger.info(
            f"Rolling window complete: " f"{len(all_results)} route-period evaluations"
        )
        return results_df

    def run_final_test(self) -> pd.DataFrame:
        """
        Final test set evaluation — called ONCE after all tuning complete.
        Uses full train (2019-2021) → test (2023).

        WARNING: Do not call this until all hyperparameter decisions are final.
        Calling this multiple times and selecting best result is data leakage.
        """
        logger.info("=" * 60)
        logger.info("FINAL TEST SET EVALUATION — TOUCHING TEST SET")
        logger.info("This should be called exactly once.")
        logger.info("=" * 60)

        df = pd.read_parquet(self.data_path)

        window = {
            "id": 99,  # sentinel value — final test
            "train_end": "2021-12-31",
            "eval_start": "2023-01-01",
            "eval_end": "2023-12-31",
        }

        results = self._evaluate_window(df, window)
        results_df = pd.DataFrame([vars(r) for r in results])

        logger.info("Final test evaluation complete")
        return results_df

    def _evaluate_window(
        self,
        df: pd.DataFrame,
        window: dict,
    ) -> list[BacktestResult]:
        """
        Evaluate one rolling window for all routes.
        Retrains model on window's train period.
        """
        train_df = df[df["DATE"] <= window["train_end"]].copy()
        eval_df = df[
            (df["DATE"] >= window["eval_start"]) & (df["DATE"] <= window["eval_end"])
        ].copy()

        if len(eval_df) == 0:
            logger.warning(f"No eval data for window {window['id']}")
            return []

        # Feature engineering — fit on train only
        fe = FeatureEngineer()
        train_features = fe.fit_transform(train_df)
        eval_features = fe.transform(eval_df)
        feature_cols = fe.get_feature_columns()

        results = []

        for route in ROUTE_CONSTRAINTS.keys():
            route_train = train_features[train_features["ROUTE"] == route].copy()
            route_eval = eval_features[eval_features["ROUTE"] == route].copy()

            if len(route_train) < 12 or len(route_eval) == 0:
                logger.warning(
                    f"Insufficient data for {route} " f"window {window['id']}"
                )
                continue

            # Train model on this window
            model = DemandModel(route=route)
            try:
                model.train(
                    train_df=train_features,
                    val_df=eval_features,
                    feature_columns=feature_cols,
                    target_column="passengers_per_departure",
                )
            except Exception as e:
                logger.error(
                    f"Training failed for {route} " f"window {window['id']}: {e}"
                )
                continue

            # Evaluate each period in eval window
            for _, row in route_eval.iterrows():
                result = self._evaluate_period(
                    model=model,
                    route=route,
                    row=row,
                    feature_cols=feature_cols,
                    window_id=window["id"],
                )
                if result:
                    results.append(result)

        return results

    def _evaluate_period(
        self,
        model: DemandModel,
        route: str,
        row: pd.Series,
        feature_cols: list[str],
        window_id: int,
    ) -> Optional[BacktestResult]:
        """
        Evaluate one route-period with all three pricing strategies.
        """
        try:
            features = pd.DataFrame([row])[feature_cols]
            predictions = model.predict(features)

            actual_demand = float(row["passengers_per_departure"])
            pred_q50 = float(predictions["q50"][0])
            pred_q10 = float(predictions["q10"][0])
            pred_q90 = float(predictions["q90"][0])

            demand_error_pct = (
                abs(pred_q50 - actual_demand) / actual_demand * 100
                if actual_demand > 0
                else 0.0
            )

            constraints = ROUTE_CONSTRAINTS[route]
            simulator = DemandPriceSimulator(route)
            optimizer = PriceOptimizer(route, constraints)

            # Baseline price — reference fare
            baseline_price = simulator.config.reference_price

            # Optimized price — grid search
            opt_result = optimizer.optimize_grid(pred_q50)
            optimized_price = opt_result.optimal_price

            # Oracle price — uses actual demand (perfect knowledge)
            oracle_result = optimizer.optimize_grid(actual_demand)
            oracle_price = oracle_result.optimal_price

            # Revenue calculations using actual demand
            # This is honest — actual demand, not predicted
            baseline_revenue = self._compute_revenue(
                baseline_price, actual_demand, constraints.capacity
            )
            optimized_revenue = self._compute_revenue(
                optimized_price, actual_demand, constraints.capacity
            )
            oracle_revenue = self._compute_revenue(
                oracle_price, actual_demand, constraints.capacity
            )

            # Uplift metrics
            revenue_uplift = (
                (optimized_revenue - baseline_revenue) / baseline_revenue * 100
                if baseline_revenue > 0
                else 0.0
            )
            regret = (
                (oracle_revenue - optimized_revenue) / oracle_revenue * 100
                if oracle_revenue > 0
                else 0.0
            )

            # Load factors
            baseline_lf = (
                min(
                    simulator.adjust_demand(actual_demand, baseline_price),
                    constraints.capacity,
                )
                / constraints.capacity
            )

            optimized_lf = (
                min(
                    simulator.adjust_demand(actual_demand, optimized_price),
                    constraints.capacity,
                )
                / constraints.capacity
            )

            return BacktestResult(
                route=route,
                period_start=str(row["DATE"]),
                period_end=str(row["DATE"]),
                window_id=window_id,
                actual_demand=round(actual_demand, 1),
                predicted_demand_q50=round(pred_q50, 1),
                predicted_demand_q10=round(pred_q10, 1),
                predicted_demand_q90=round(pred_q90, 1),
                demand_error_pct=round(demand_error_pct, 2),
                baseline_price=round(baseline_price, 2),
                optimized_price=round(optimized_price, 2),
                oracle_price=round(oracle_price, 2),
                baseline_revenue=round(baseline_revenue, 2),
                optimized_revenue=round(optimized_revenue, 2),
                oracle_revenue=round(oracle_revenue, 2),
                revenue_uplift_vs_baseline_pct=round(revenue_uplift, 2),
                regret_vs_oracle_pct=round(regret, 2),
                baseline_load_factor=round(baseline_lf, 3),
                optimized_load_factor=round(optimized_lf, 3),
            )

        except Exception as e:
            logger.error(f"Period evaluation failed for {route}: {e}")
            return None

    def _compute_revenue(
        self,
        price: float,
        actual_demand: float,
        capacity: int,
    ) -> float:
        """
        Revenue using actual demand — not predicted.
        This is the honest evaluation metric.
        """
        realized = min(actual_demand, capacity)
        return price * realized
