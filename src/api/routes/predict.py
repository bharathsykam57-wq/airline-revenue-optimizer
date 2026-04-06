"""Demand prediction endpoint."""

import json
from fastapi import APIRouter, HTTPException
from src.api.schemas.request import DemandPredictionRequest
from src.api.schemas.response import DemandPredictionResponse
from src.api.model_registry import get_model, get_model_version
from src.api.feature_builder import build_features_for_request
from src.utils.redis_client import get_redis_client, get_ttl_seconds, build_cache_key
from src.utils.logger import logger

router = APIRouter()


@router.post("/predict-demand", response_model=DemandPredictionResponse)
async def predict_demand(request: DemandPredictionRequest):
    """
    Predict passenger demand for a route and time period.
    Returns q10, q50, q90 quantiles — never just a point estimate.
    """
    model = get_model(request.route)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded for route {request.route}. "
            f"Run training pipeline first.",
        )

    model_version = get_model_version()
    cache_key = build_cache_key(
        request.route,
        f"{request.year}-{request.month:02d}",
        model_version,
        request.days_to_departure,
    )

    # Try cache first
    redis = get_redis_client()
    if redis:
        try:
            cached = redis.get(cache_key)
            if cached:
                data = json.loads(cached)
                data["cached"] = True
                logger.debug(f"Cache hit: {cache_key}")
                return DemandPredictionResponse(**data)
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")

    # Build features and predict
    features = build_features_for_request(request.route, request.year, request.month)
    predictions = model.predict(features)

    q10 = float(predictions["q10"][0])
    q50 = float(predictions["q50"][0])
    q90 = float(predictions["q90"][0])

    response_data = {
        "route": request.route,
        "year": request.year,
        "month": request.month,
        "predicted_demand_q10": round(q10, 1),
        "predicted_demand_q50": round(q50, 1),
        "predicted_demand_q90": round(q90, 1),
        "prediction_interval_width": round(q90 - q10, 1),
        "model_version": model_version,
        "cached": False,
    }

    # Cache the result
    if redis:
        try:
            ttl = get_ttl_seconds(request.days_to_departure)
            redis.setex(cache_key, ttl, json.dumps(response_data))
            logger.debug(f"Cached prediction: {cache_key} TTL={ttl}s")
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    return DemandPredictionResponse(**response_data)
