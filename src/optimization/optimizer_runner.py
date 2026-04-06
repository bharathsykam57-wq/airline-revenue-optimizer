"""
Runs price optimization for all routes using trained demand models.
Benchmarks grid search vs Bayesian optimization.
"""

import numpy as np
import pandas as pd
from src.modeling.demand_model import DemandModel
from src.features.feature_engineer import FeatureEngineer
from src.optimization.price_optimizer import PriceOptimizer, PricingConstraints
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


class OptimizerRunner:
    """
    Loads trained models and runs price optimization.
    Benchmarks grid search vs Bayesian optimization.
    """

    def __init__(self):
        self.settings = get_settings()

    def run(
        self,
        data_path: str = "data/processed/t100_cleaned.parquet",
        target_period: str = "2023",
    ) -> pd.DataFrame:
        """
        Run optimization for all routes on target period data.
        Returns DataFrame with optimization results per route.
        """
        logger.info("Starting price optimization pipeline")

        df = pd.read_parquet(data_path)

        # Use validation period for optimization demo
        # Test period reserved for final backtesting
        val_df = df[(df["DATE"] >= "2022-01-01") & (df["DATE"] < "2023-01-01")].copy()

        # Feature engineering — fit on train, transform val
        train_df = df[df["DATE"] < "2022-01-01"].copy()
        fe = FeatureEngineer()
        fe.fit(train_df)
        val_features = fe.transform(val_df)

        results = []

        for route, constraints in ROUTE_CONSTRAINTS.items():
            logger.info(f"Optimizing route: {route}")

            try:
                # Load trained model
                model = DemandModel.load(MODEL_DIR, route)

                # Get validation features for this route
                route_features = val_features[val_features["ROUTE"] == route].copy()

                if len(route_features) == 0:
                    logger.warning(f"No validation data for {route}")
                    continue

                # Use median month for demonstration
                # In production: optimize per departure date
                median_features = route_features.iloc[
                    len(route_features) // 2 : len(route_features) // 2 + 1
                ]

                # Get demand predictions
                predictions = model.predict(median_features)
                base_demand_q50 = float(predictions["q50"][0])
                base_demand_q10 = float(predictions["q10"][0])
                base_demand_q90 = float(predictions["q90"][0])

                logger.info(
                    f"{route} — base demand: "
                    f"q10={base_demand_q10:.0f}, "
                    f"q50={base_demand_q50:.0f}, "
                    f"q90={base_demand_q90:.0f}"
                )

                optimizer = PriceOptimizer(route, constraints)

                # Verify revenue curve has interior maximum
                self._check_revenue_curve(optimizer, base_demand_q50, constraints)

                # Grid search
                grid_result = optimizer.optimize_grid(base_demand_q50)
                logger.info(
                    f"{route} grid: price={grid_result.optimal_price:.2f}, "
                    f"revenue={grid_result.expected_revenue:.2f}, "
                    f"uplift={grid_result.revenue_uplift_pct:.1f}%"
                )

                # Bayesian optimization
                bayes_result = optimizer.optimize_bayesian(base_demand_q50, n_trials=50)
                logger.info(
                    f"{route} bayes: price={bayes_result.optimal_price:.2f}, "
                    f"revenue={bayes_result.expected_revenue:.2f}, "
                    f"uplift={bayes_result.revenue_uplift_pct:.1f}%"
                )

                # Compare methods
                price_diff = abs(grid_result.optimal_price - bayes_result.optimal_price)
                revenue_diff_pct = (
                    abs(grid_result.expected_revenue - bayes_result.expected_revenue)
                    / grid_result.expected_revenue
                    * 100
                )

                logger.info(
                    f"{route} — method comparison: "
                    f"price_diff={price_diff:.2f}, "
                    f"revenue_diff={revenue_diff_pct:.2f}%"
                )

                results.append(
                    {
                        "route": route,
                        "base_demand_q10": base_demand_q10,
                        "base_demand_q50": base_demand_q50,
                        "base_demand_q90": base_demand_q90,
                        "grid_optimal_price": grid_result.optimal_price,
                        "grid_revenue": grid_result.expected_revenue,
                        "grid_uplift_pct": grid_result.revenue_uplift_pct,
                        "bayes_optimal_price": bayes_result.optimal_price,
                        "bayes_revenue": bayes_result.expected_revenue,
                        "bayes_uplift_pct": bayes_result.revenue_uplift_pct,
                        "price_diff_grid_vs_bayes": price_diff,
                        "revenue_diff_pct": revenue_diff_pct,
                        "load_factor": grid_result.expected_load_factor,
                        "baseline_price": grid_result.baseline_price,
                        "baseline_revenue": grid_result.baseline_revenue,
                    }
                )

            except Exception as e:
                logger.error(f"Optimization failed for {route}: {e}")
                raise

        results_df = pd.DataFrame(results)
        logger.info("Optimization pipeline complete")
        return results_df

    def _check_revenue_curve(
        self,
        optimizer: PriceOptimizer,
        base_demand: float,
        constraints: PricingConstraints,
    ) -> None:
        """
        Verify revenue curve has an interior maximum.
        If optimizer always returns boundary solution,
        elasticity calibration is wrong.

        This is the check that prevents the boundary solution bug
        documented in the project plan.
        """
        prices, revenues = optimizer.simulator.get_revenue_curve(
            base_demand,
            (constraints.price_min, constraints.price_max),
            constraints.capacity,
            n_points=200,
        )

        max_idx = np.argmax(revenues)
        max_price = prices[max_idx]

        boundary_tolerance = (
            constraints.price_max - constraints.price_min
        ) * 0.05  # 5% tolerance

        if (
            max_price <= constraints.price_min + boundary_tolerance
            or max_price >= constraints.price_max - boundary_tolerance
        ):
            logger.warning(
                f"{optimizer.route} — revenue curve maximum at boundary: "
                f"price={max_price:.2f}. "
                f"Elasticity may need recalibration. "
                f"Interior maximum is at price={max_price:.2f}"
            )
        else:
            logger.info(
                f"{optimizer.route} — revenue curve interior maximum "
                f"confirmed at price={max_price:.2f}"
            )
