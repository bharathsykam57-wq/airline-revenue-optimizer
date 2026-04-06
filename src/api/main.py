"""
AROF FastAPI application.
Production-patterned ML serving with Redis caching.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes.predict import router as predict_router
from src.api.routes.optimize import router as optimize_router
from src.api.routes.simulate import router as simulate_router
from src.api.model_registry import load_models, get_loaded_count, get_model_version
from src.utils.redis_client import get_redis_client
from src.utils.settings import get_settings
from src.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup. Clean up at shutdown."""
    logger.info("AROF API starting up")
    load_models()
    logger.info(f"Startup complete — " f"{get_loaded_count()} models loaded")
    yield
    logger.info("AROF API shutting down")


app = FastAPI(
    title="Airline Revenue Optimization Framework",
    description=(
        "Production-patterned ML framework for airline demand "
        "forecasting and price optimization. "
        "Built on BTS public data with synthetic price elasticity. "
        "Limitations documented at /docs."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict_router, tags=["demand"])
app.include_router(optimize_router, tags=["optimization"])
app.include_router(simulate_router, tags=["simulation"])


@app.get("/health")
async def health():
    """Health check — used by Docker and load balancers."""
    settings = get_settings()
    redis = get_redis_client()

    return {
        "status": "healthy",
        "version": "0.1.0",
        "models_loaded": get_loaded_count(),
        "model_version": get_model_version(),
        "redis_available": redis is not None,
        "mlflow_uri": settings.mlflow_tracking_uri,
    }
