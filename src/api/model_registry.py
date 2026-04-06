"""
Loads all trained demand models at API startup.
Single load per process — not per request.

Pattern: module-level singleton loaded once.
FastAPI lifespan events manage the loading lifecycle.
"""

from pathlib import Path
from typing import Optional
from src.modeling.demand_model import DemandModel
from src.utils.logger import logger

MODEL_DIR = "models/demand"
MODEL_VERSION = "1.0.0"

ROUTES = [
    "JFK-LAX",
    "LAX-JFK",
    "ORD-MIA",
    "MIA-ORD",
    "LAX-SEA",
    "SEA-LAX",
]

# Module-level registry
_models: dict[str, DemandModel] = {}
_loaded = False


def load_models() -> None:
    """
    Load all route models into memory.
    Called once at startup via FastAPI lifespan.
    """
    global _models, _loaded

    logger.info("Loading demand models into registry")
    loaded_count = 0

    for route in ROUTES:
        model_path = Path(MODEL_DIR) / route.replace("-", "_")

        if not model_path.exists():
            logger.warning(
                f"Model not found for {route} at {model_path}. "
                f"Train models before starting API."
            )
            continue

        try:
            model = DemandModel.load(MODEL_DIR, route)
            _models[route] = model
            loaded_count += 1
            logger.info(f"Loaded model for {route}")
        except Exception as e:
            logger.error(f"Failed to load model for {route}: {e}")

    _loaded = True
    logger.info(f"Model registry ready: {loaded_count}/{len(ROUTES)} routes loaded")


def get_model(route: str) -> Optional[DemandModel]:
    """Get loaded model for a route. Returns None if not loaded."""
    return _models.get(route)


def get_loaded_count() -> int:
    """Number of successfully loaded models."""
    return len(_models)


def get_model_version() -> str:
    """Current model version string."""
    return MODEL_VERSION
