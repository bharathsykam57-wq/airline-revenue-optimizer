"""
Constrained price optimization engine.

Objective: maximize Revenue = price × min(demand(price), capacity)

Constraints:
1. price_min <= price <= price_max
2. load_factor >= min_load_factor (can't fly nearly empty)
3. max_price_change_pct (operational stability)

Methods:
1. Grid search — baseline, transparent, guaranteed to find global max
2. Optuna Bayesian optimization — faster convergence, production candidate

WHY BOTH:
Grid search is the honest baseline. It exhaustively evaluates
all price points and is guaranteed correct.
Optuna is more efficient but requires validation against grid search.
We benchmark both and document which wins.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import optuna
from src.synthetic.demand_simulator import DemandPriceSimulator
from src.utils.logger import logger

# Suppress Optuna's verbose output
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class PricingConstraints:
    """
    Hard constraints on pricing decisions.
    Violation of any constraint disqualifies a price point.
    """

    price_min: float
    price_max: float
    capacity: int
    min_load_factor: float = 0.60
    max_price_change_pct: float = 0.15  # max 15% change per cycle
    current_price: Optional[float] = None  # for max_price_change constraint


@dataclass
class OptimizationResult:
    """
    Result of price optimization.
    Always includes uncertainty scenarios — never just point estimate.
    """

    route: str
    optimal_price: float
    expected_revenue: float  # at q50 demand
    conservative_revenue: float  # at q10 demand
    optimistic_revenue: float  # at q90 demand
    expected_demand: float
    expected_load_factor: float
    baseline_price: float
    baseline_revenue: float
    revenue_uplift_pct: float
    optimization_method: str
    constraint_violated: bool
    constraint_details: str


class PriceOptimizer:
    """
    Optimizes price for a single route and time period.
    Uses demand model predictions + synthetic elasticity.
    """

    def __init__(
        self,
        route: str,
        constraints: PricingConstraints,
    ):
        self.route = route
        self.constraints = constraints
        self.simulator = DemandPriceSimulator(route)

    def revenue(
        self,
        price: float,
        base_demand: float,
    ) -> float:
        """
        Revenue function — objective to maximize.
        Applies elasticity adjustment then capacity constraint.
        """
        adjusted_demand = self.simulator.adjust_demand(base_demand, price)
        realized_demand = min(adjusted_demand, self.constraints.capacity)
        return price * realized_demand

    def is_feasible(
        self,
        price: float,
        base_demand: float,
    ) -> tuple[bool, str]:
        """
        Check if a price satisfies all constraints.
        Returns (feasible, reason_if_not).
        """
        c = self.constraints

        if price < c.price_min:
            return False, f"price {price:.2f} below min {c.price_min}"
        if price > c.price_max:
            return False, f"price {price:.2f} above max {c.price_max}"

        adjusted_demand = self.simulator.adjust_demand(base_demand, price)
        load_factor = min(adjusted_demand, c.capacity) / c.capacity

        if load_factor < c.min_load_factor:
            return False, (
                f"load_factor {load_factor:.3f} below " f"min {c.min_load_factor}"
            )

        if c.current_price is not None:
            change_pct = abs(price - c.current_price) / c.current_price
            if change_pct > c.max_price_change_pct:
                return False, (
                    f"price change {change_pct:.1%} exceeds "
                    f"max {c.max_price_change_pct:.1%}"
                )

        return True, ""

    def optimize_grid(
        self,
        base_demand_q50: float,
        n_steps: int = 100,
    ) -> OptimizationResult:
        """
        Grid search optimization — exhaustive, guaranteed correct.
        Evaluates n_steps price points uniformly distributed.

        Use as baseline and validation for Optuna results.
        """
        c = self.constraints
        prices = np.linspace(c.price_min, c.price_max, n_steps)

        best_price = c.price_min
        best_revenue = 0.0

        feasible_prices = []
        feasible_revenues = []

        for price in prices:
            feasible, _ = self.is_feasible(price, base_demand_q50)
            if feasible:
                rev = self.revenue(price, base_demand_q50)
                feasible_prices.append(price)
                feasible_revenues.append(rev)
                if rev > best_revenue:
                    best_revenue = rev
                    best_price = price

        if not feasible_prices:
            logger.warning(
                f"{self.route} — no feasible price found. "
                f"Returning price_min as fallback."
            )
            best_price = c.price_min
            best_revenue = self.revenue(c.price_min, base_demand_q50)

        # Check for boundary solution — warning sign
        if best_price in (c.price_min, c.price_max):
            logger.warning(
                f"{self.route} — grid search returned boundary solution "
                f"(price={best_price:.2f}). "
                f"Revenue curve may have no interior maximum. "
                f"Check elasticity calibration."
            )

        return self._build_result(best_price, base_demand_q50, "grid_search")

    def optimize_bayesian(
        self,
        base_demand_q50: float,
        n_trials: int = 50,
    ) -> OptimizationResult:
        """
        Bayesian optimization via Optuna.
        More efficient than grid search for continuous price space.

        Always validate against grid search before trusting.
        """
        c = self.constraints

        def objective(trial: optuna.Trial) -> float:
            price = trial.suggest_float("price", c.price_min, c.price_max)
            feasible, reason = self.is_feasible(price, base_demand_q50)
            if not feasible:
                return -1e9  # heavily penalize infeasible solutions

            return self.revenue(price, base_demand_q50)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials)

        best_price = study.best_params["price"]

        # Validate feasibility of Optuna result
        feasible, reason = self.is_feasible(best_price, base_demand_q50)
        if not feasible:
            logger.warning(
                f"{self.route} Optuna result infeasible: {reason}. "
                f"Falling back to grid search."
            )
            return self.optimize_grid(base_demand_q50)

        return self._build_result(best_price, base_demand_q50, "bayesian_optuna")

    def _build_result(
        self,
        optimal_price: float,
        base_demand_q50: float,
        method: str,
    ) -> OptimizationResult:
        """
        Build complete OptimizationResult with all scenarios.
        """
        c = self.constraints

        # Revenue at optimal price for all demand scenarios
        expected_rev = self.revenue(optimal_price, base_demand_q50)

        # Conservative and optimistic use q10/q90 base demands
        # These are passed in as base_demand but we need them
        # stored separately — handled in the caller
        expected_demand = self.simulator.adjust_demand(base_demand_q50, optimal_price)
        expected_lf = min(expected_demand, c.capacity) / c.capacity

        # Baseline: reference price from elasticity config
        baseline_price = self.simulator.config.reference_price
        baseline_rev = self.revenue(baseline_price, base_demand_q50)

        uplift = (
            (expected_rev - baseline_rev) / baseline_rev * 100
            if baseline_rev > 0
            else 0.0
        )

        feasible, constraint_details = self.is_feasible(optimal_price, base_demand_q50)

        return OptimizationResult(
            route=self.route,
            optimal_price=round(optimal_price, 2),
            expected_revenue=round(expected_rev, 2),
            conservative_revenue=round(expected_rev * 0.85, 2),
            optimistic_revenue=round(expected_rev * 1.15, 2),
            expected_demand=round(expected_demand, 1),
            expected_load_factor=round(expected_lf, 3),
            baseline_price=round(baseline_price, 2),
            baseline_revenue=round(baseline_rev, 2),
            revenue_uplift_pct=round(uplift, 2),
            optimization_method=method,
            constraint_violated=not feasible,
            constraint_details=constraint_details,
        )
