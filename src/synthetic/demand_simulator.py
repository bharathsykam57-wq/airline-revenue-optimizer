"""
Synthetic price-demand relationship layer.

CRITICAL DOCUMENTATION:
BTS T-100 data does not include ticket prices.
We cannot learn price elasticity from this data directly.

This module implements a synthetic price-demand adjustment:
    adjusted_demand = base_demand × price_adjustment_factor

Where price_adjustment_factor is derived from:
    price_elasticity = (% change in demand) / (% change in price)

Elasticity values are configured per route in configs/config.yaml.
They are calibrated assumptions based on:
- Academic literature on airline price elasticity
- Route characteristics (business vs leisure share)
- Distance (long-haul vs short-haul sensitivity)

They are NOT learned from data. This is explicitly a limitation.

In production: replace this module with a model trained on
actual booking data with fare information (GDS data).
The interface is designed to make this replacement seamless.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class RouteElasticityConfig:
    """
    Price elasticity configuration for a route.
    Separate elasticity for business and leisure segments.
    """

    route: str
    elasticity_leisure: float  # typically -1.5 to -2.5
    elasticity_business: float  # typically -0.5 to -1.0
    business_share: float  # fraction of passengers that are business
    reference_price: float  # price at which base demand is calibrated


class DemandPriceSimulator:
    """
    Applies synthetic price elasticity to model demand predictions.

    Formula:
        price_ratio = price / reference_price
        leisure_adjustment = price_ratio ^ elasticity_leisure
        business_adjustment = price_ratio ^ elasticity_business
        blended_adjustment = (
            business_share × business_adjustment +
            (1 - business_share) × leisure_adjustment
        )
        adjusted_demand = base_demand × blended_adjustment

    This produces a downward-sloping demand curve:
    - Higher price → lower demand (elasticity < 0)
    - Business segment less sensitive than leisure
    - Blended by segment mix per route
    """

    # Route elasticity configs — calibrated assumptions
    # Source: airline economics literature + route characteristics
    ROUTE_CONFIGS = {
        "JFK-LAX": RouteElasticityConfig(
            route="JFK-LAX",
            elasticity_leisure=-1.8,
            elasticity_business=-0.7,
            business_share=0.35,
            reference_price=312.0,  # median fare from research
        ),
        "LAX-JFK": RouteElasticityConfig(
            route="LAX-JFK",
            elasticity_leisure=-1.8,
            elasticity_business=-0.7,
            business_share=0.35,
            reference_price=298.0,
        ),
        "ORD-MIA": RouteElasticityConfig(
            route="ORD-MIA",
            elasticity_leisure=-2.1,
            elasticity_business=-0.9,
            business_share=0.25,
            reference_price=218.0,
        ),
        "MIA-ORD": RouteElasticityConfig(
            route="MIA-ORD",
            elasticity_leisure=-2.1,
            elasticity_business=-0.9,
            business_share=0.25,
            reference_price=224.0,
        ),
        "LAX-SEA": RouteElasticityConfig(
            route="LAX-SEA",
            elasticity_leisure=-1.6,
            elasticity_business=-0.8,
            business_share=0.30,
            reference_price=167.0,
        ),
        "SEA-LAX": RouteElasticityConfig(
            route="SEA-LAX",
            elasticity_leisure=-1.6,
            elasticity_business=-0.8,
            business_share=0.30,
            reference_price=171.0,
        ),
    }

    def __init__(self, route: str):
        if route not in self.ROUTE_CONFIGS:
            raise ValueError(
                f"No elasticity config for route: {route}. "
                f"Available: {list(self.ROUTE_CONFIGS.keys())}"
            )
        self.config = self.ROUTE_CONFIGS[route]
        self.route = route

    def adjust_demand(
        self,
        base_demand: float,
        price: float,
    ) -> float:
        """
        Apply price elasticity adjustment to base demand prediction.

        base_demand: model prediction at reference price
        price: proposed price to evaluate
        returns: demand estimate at given price

        NOTE: Returns float, not integer.
        Demand is a continuous prediction — floor/ceil at optimization.
        """
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")

        config = self.config
        price_ratio = price / config.reference_price

        # Segment-specific price adjustments
        leisure_adjustment = price_ratio**config.elasticity_leisure
        business_adjustment = price_ratio**config.elasticity_business

        # Blended adjustment weighted by segment mix
        blended_adjustment = (
            config.business_share * business_adjustment
            + (1 - config.business_share) * leisure_adjustment
        )

        adjusted = base_demand * blended_adjustment
        return max(0.0, adjusted)  # demand cannot be negative

    def get_demand_curve(
        self,
        base_demand: float,
        price_range: tuple[float, float],
        n_points: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate full demand curve over price range.
        Used for visualization and optimization.
        Returns (prices, demands) arrays.
        """
        prices = np.linspace(price_range[0], price_range[1], n_points)
        demands = np.array([self.adjust_demand(base_demand, p) for p in prices])
        return prices, demands

    def get_revenue_curve(
        self,
        base_demand: float,
        price_range: tuple[float, float],
        capacity: int,
        n_points: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate revenue curve over price range.
        Revenue = price × min(demand, capacity)
        Used to verify interior maximum exists before optimization.
        """
        prices, demands = self.get_demand_curve(base_demand, price_range, n_points)
        revenues = prices * np.minimum(demands, capacity)
        return prices, revenues
