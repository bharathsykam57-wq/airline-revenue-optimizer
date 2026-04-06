"""Price optimization endpoint."""

import json
from fastapi import APIRouter, HTTPException
from src.api.schemas.request import PriceOptimizationRequest
from src.api.schemas.response import PriceOptimizationResponse
from src.api.model_registry import get_model, get_model_version
from src.api.feature_builder import build_features_for_request
from src.optimization.price_optimizer import PriceOptimizer, PricingConstraints
from src.utils.redis_client import get_redis_client, get_ttl_seconds, build_cache_key
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


@router.post("/optimize-price", response_model=PriceOptimizationResponse)
async def optimize_price(request: PriceOptimizationRequest):
    """
    Optimize price for a route and time period.
    Returns optimal price with revenue scenarios.
    """
    model = get_model(request.route)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded for route {request.route}",
        )

    model_version = get_model_version()
    cache_key = build_cache_key(
        f"opt_{request.route}",
        f"{request.year}-{request.month:02d}",
        model_version,
        request.days_to_departure,
    )

    # Try cache
    redis = get_redis_client()
    if redis:
        try:
            cached = redis.get(cache_key)
            if cached:
                data = json.loads(cached)
                data["cached"] = True
                return PriceOptimizationResponse(**data)
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")

    # Get demand prediction
    features = build_features_for_request(request.route, request.year, request.month)
    predictions = model.predict(features)
    base_demand_q50 = float(predictions["q50"][0])

    # Optimize
    constraints = ROUTE_CONSTRAINTS[request.route]
    if request.current_price:
        constraints.current_price = request.current_price

    optimizer = PriceOptimizer(request.route, constraints)

    if request.method == "bayesian":
        result = optimizer.optimize_bayesian(base_demand_q50)
    else:
        result = optimizer.optimize_grid(base_demand_q50)

    response_data = {
        "route": request.route,
        "year": request.year,
        "month": request.month,
        "optimal_price": result.optimal_price,
        "expected_revenue": result.expected_revenue,
        "conservative_revenue": result.conservative_revenue,
        "optimistic_revenue": result.optimistic_revenue,
        "expected_demand": result.expected_demand,
        "expected_load_factor": result.expected_load_factor,
        "baseline_price": result.baseline_price,
        "baseline_revenue": result.baseline_revenue,
        "revenue_uplift_pct": result.revenue_uplift_pct,
        "optimization_method": result.optimization_method,
        "constraint_violated": result.constraint_violated,
        "model_version": model_version,
        "cached": False,
    }

    if redis:
        try:
            ttl = get_ttl_seconds(request.days_to_departure)
            redis.setex(cache_key, ttl, json.dumps(response_data))
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    return PriceOptimizationResponse(**response_data)
