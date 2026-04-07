"""Tests for price optimizer — constraints must always be respected."""

import sys

sys.path.insert(0, ".")

from src.optimization.price_optimizer import PriceOptimizer, PricingConstraints


def make_optimizer(route="JFK-LAX"):
    constraints = PricingConstraints(
        price_min=89,
        price_max=1200,
        capacity=180,
        min_load_factor=0.60,
    )
    return PriceOptimizer(route, constraints)


def test_grid_search_respects_price_bounds():
    """Optimizer must never return price outside [min, max]."""
    optimizer = make_optimizer()
    result = optimizer.optimize_grid(base_demand_q50=140.0)

    assert result.optimal_price >= 89, f"Price {result.optimal_price} below minimum 89"
    assert (
        result.optimal_price <= 1200
    ), f"Price {result.optimal_price} above maximum 1200"


def test_feasibility_check_load_factor():
    """Price that violates min load factor must be infeasible."""
    optimizer = make_optimizer()

    # Very high price will drive demand below min load factor
    feasible, reason = optimizer.is_feasible(
        price=5000.0,
        base_demand=140.0,
    )
    assert not feasible, "Extremely high price should be infeasible"


def test_revenue_function_positive():
    """Revenue must always be positive for feasible price."""
    optimizer = make_optimizer()
    revenue = optimizer.revenue(price=300.0, base_demand=140.0)
    assert revenue > 0, "Revenue must be positive"


def test_bayesian_within_bounds():
    """Bayesian optimization must respect same bounds as grid."""
    optimizer = make_optimizer()
    result = optimizer.optimize_bayesian(base_demand_q50=140.0, n_trials=20)

    assert result.optimal_price >= 89
    assert result.optimal_price <= 1200


def test_constraint_violated_flag():
    """constraint_violated must be False for valid results."""
    optimizer = make_optimizer()
    result = optimizer.optimize_grid(base_demand_q50=140.0)
    assert (
        not result.constraint_violated
    ), "Valid optimization should not violate constraints"
