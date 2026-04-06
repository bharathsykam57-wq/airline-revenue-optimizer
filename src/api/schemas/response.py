"""
Response schemas — always include uncertainty bounds and data context.
"""

from pydantic import BaseModel


DATA_DISCLAIMER = (
    "Predictions based on BTS T-100 real demand data + "
    "synthetic price elasticity assumptions. "
    "Revenue figures are per-flight estimates."
)


class DemandPredictionResponse(BaseModel):
    route: str
    year: int
    month: int
    predicted_demand_q10: float
    predicted_demand_q50: float
    predicted_demand_q90: float
    prediction_interval_width: float
    model_version: str
    cached: bool
    data_disclaimer: str = DATA_DISCLAIMER


class PriceOptimizationResponse(BaseModel):
    route: str
    year: int
    month: int
    optimal_price: float
    expected_revenue: float
    conservative_revenue: float
    optimistic_revenue: float
    expected_demand: float
    expected_load_factor: float
    baseline_price: float
    baseline_revenue: float
    revenue_uplift_pct: float
    optimization_method: str
    constraint_violated: bool
    model_version: str
    cached: bool
    data_disclaimer: str = DATA_DISCLAIMER


class ScenarioResult(BaseModel):
    price: float
    predicted_demand: float
    expected_revenue: float
    load_factor: float
    feasible: bool


class ScenarioSimulationResponse(BaseModel):
    route: str
    year: int
    month: int
    scenarios: list[ScenarioResult]
    recommended_price: float
    model_version: str
    cached: bool
    data_disclaimer: str = DATA_DISCLAIMER


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: int
    redis_available: bool
    mlflow_uri: str
