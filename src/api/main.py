"""
AROF FastAPI application.
Skeleton for container validation — full implementation in Week 6.
"""

from fastapi import FastAPI
from src.utils.logger import logger

app = FastAPI(
    title="Airline Revenue Optimization Framework",
    description=(
        "Production-patterned ML framework for airline demand "
        "forecasting and price optimization. "
        "Built on BTS public data with synthetic booking window modeling. "
        "Limitations documented explicitly in README."
    ),
    version="0.1.0",
)


@app.get("/health")
async def health():
    """
    Health check endpoint.
    Used by Docker healthcheck and load balancers.
    Returns service status and version.
    """
    logger.info("Health check called")
    return {
        "status": "healthy",
        "version": "0.1.0",
        "service": "arof-api",
    }
