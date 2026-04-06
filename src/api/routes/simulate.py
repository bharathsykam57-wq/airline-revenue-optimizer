"""Scenario simulation endpoint."""

import json
from fastapi import APIRouter, HTTPException
from src.api.schemas.request import ScenarioSimulationRequest
from src.api.schemas.response import ScenarioSimulationResponse, ScenarioResult
from src.api.model_registry import get_model, get_model_version
from src.api.feature_builder import build_features_for_request
from src.optimization.price_optimizer import PriceOptimizer, PricingConstraints
from src.synthetic.demand_simulator import DemandPriceSimulator
from src.utils.redis_client import get_redis_client, build_cache_key
from src.utils.logger import logger

router = APIRouter()

ROUTE_CONSTRAINTS = {
    "JFK-LAX": PricingConstraints(price_min=89, price_max=1200, capacity=180),
    "LAX-JFK": PricingConstraints(price_min=89, price_max=1200, capacity=180),
    "ORD-MIA": PricingConstraints(price_min=69, price_max=900, capacity=150),
    "MIA-ORD": PricingConstraints(price_min=69, price_max=900, capacity=150),
    "LAX-SEA": PricingConstraints(price_min=49, price_max=650, capacity=160),
    "SEA-LAX": PricingConstraints(price_min=49, price_max=650, capacity=160),
}


@router.post("/simulate-scenario", response_model=ScenarioSimulationResponse)
async def simulate_scenario(request: ScenarioSimulationRequest):
    """
    Simulate revenue outcomes at multiple price points.
    Used for what-if analysis — not cached as long as predict.
    """
    model = get_model(request.route)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded for route {request.route}",
        )

    model_version = get_model_version()
    cache_key = build_cache_key(
        f"sim_{request.route}",
        f"{request.year}-{request.month:02d}_" f"{hash(tuple(request.price_points))}",
        model_version,
        0,
    )

    # Simulation cache TTL is short — 5 minutes
    redis = get_redis_client()
    if redis:
        try:
            cached = redis.get(cache_key)
            if cached:
                data = json.loads(cached)
                data["cached"] = True
                return ScenarioSimulationResponse(**data)
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")

    features = build_features_for_request(request.route, request.year, request.month)
    predictions = model.predict(features)
    base_demand = float(predictions["q50"][0])

    constraints = ROUTE_CONSTRAINTS[request.route]
    simulator = DemandPriceSimulator(request.route)
    optimizer = PriceOptimizer(request.route, constraints)

    scenarios = []
    best_revenue = 0.0
    best_price = request.price_points[0]

    for price in request.price_points:
        adjusted_demand = simulator.adjust_demand(base_demand, price)
        realized = min(adjusted_demand, constraints.capacity)
        revenue = price * realized
        load_factor = realized / constraints.capacity
        feasible, _ = optimizer.is_feasible(price, base_demand)

        scenarios.append(
            ScenarioResult(
                price=round(price, 2),
                predicted_demand=round(adjusted_demand, 1),
                expected_revenue=round(revenue, 2),
                load_factor=round(load_factor, 3),
                feasible=feasible,
            )
        )

        if feasible and revenue > best_revenue:
            best_revenue = revenue
            best_price = price

    response_data = {
        "route": request.route,
        "year": request.year,
        "month": request.month,
        "scenarios": [s.model_dump() for s in scenarios],
        "recommended_price": best_price,
        "model_version": model_version,
        "cached": False,
    }

    if redis:
        try:
            redis.setex(cache_key, 300, json.dumps(response_data))
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    return ScenarioSimulationResponse(**response_data)
