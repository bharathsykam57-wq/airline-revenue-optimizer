"""
Request schemas for all API endpoints.
Pydantic validates all inputs — no raw dicts passed to models.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


VALID_ROUTES = {
    "JFK-LAX",
    "LAX-JFK",
    "ORD-MIA",
    "MIA-ORD",
    "LAX-SEA",
    "SEA-LAX",
}


class DemandPredictionRequest(BaseModel):
    route: str = Field(..., description="Route ID e.g. JFK-LAX")
    year: int = Field(..., ge=2019, le=2030)
    month: int = Field(..., ge=1, le=12)
    days_to_departure: int = Field(
        default=30,
        ge=0,
        le=365,
        description="Days until departure — affects cache TTL",
    )

    @field_validator("route")
    @classmethod
    def validate_route(cls, v: str) -> str:
        if v not in VALID_ROUTES:
            raise ValueError(f"Invalid route: {v}. Valid routes: {VALID_ROUTES}")
        return v


class PriceOptimizationRequest(BaseModel):
    route: str = Field(..., description="Route ID e.g. JFK-LAX")
    year: int = Field(..., ge=2019, le=2030)
    month: int = Field(..., ge=1, le=12)
    days_to_departure: int = Field(default=30, ge=0, le=365)
    current_price: Optional[float] = Field(
        default=None,
        description="Current price — enables max_price_change constraint",
    )
    method: str = Field(
        default="grid",
        description="Optimization method: grid or bayesian",
    )

    @field_validator("route")
    @classmethod
    def validate_route(cls, v: str) -> str:
        if v not in VALID_ROUTES:
            raise ValueError(f"Invalid route: {v}")
        return v

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        if v not in ("grid", "bayesian"):
            raise ValueError("method must be 'grid' or 'bayesian'")
        return v


class ScenarioSimulationRequest(BaseModel):
    route: str = Field(..., description="Route ID")
    year: int = Field(..., ge=2019, le=2030)
    month: int = Field(..., ge=1, le=12)
    price_points: list[float] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of prices to simulate",
    )

    @field_validator("route")
    @classmethod
    def validate_route(cls, v: str) -> str:
        if v not in VALID_ROUTES:
            raise ValueError(f"Invalid route: {v}")
        return v

    @field_validator("price_points")
    @classmethod
    def validate_prices(cls, v: list[float]) -> list[float]:
        if any(p <= 0 for p in v):
            raise ValueError("All price points must be positive")
        return v
