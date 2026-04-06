"""
Builds feature vectors from API requests.
Reuses FeatureEngineer but applies to single-row inputs.

This is intentionally simple — single row inference.
Production would use a proper feature store with pre-computed features.
"""

import pandas as pd
import numpy as np
from src.features.feature_engineer import PEAK_LEISURE_MONTHS, PEAK_BUSINESS_MONTHS

# Route static features
ROUTE_DISTANCE = {
    "JFK-LAX": 3983,
    "LAX-JFK": 3983,
    "ORD-MIA": 2163,
    "MIA-ORD": 2163,
    "LAX-SEA": 1537,
    "SEA-LAX": 1537,
}
ROUTE_BUSINESS_SHARE = {
    "JFK-LAX": 0.35,
    "LAX-JFK": 0.35,
    "ORD-MIA": 0.25,
    "MIA-ORD": 0.25,
    "LAX-SEA": 0.30,
    "SEA-LAX": 0.30,
}
ROUTE_CODES = {
    "JFK-LAX": 0,
    "LAX-JFK": 1,
    "LAX-SEA": 2,
    "MIA-ORD": 3,
    "ORD-MIA": 4,
    "SEA-LAX": 5,
}
# Route baseline demand (from training data mean)
ROUTE_BASELINE_DEMAND = {
    "JFK-LAX": 127.0,
    "LAX-JFK": 127.8,
    "ORD-MIA": 146.3,
    "MIA-ORD": 146.0,
    "LAX-SEA": 120.9,
    "SEA-LAX": 121.0,
}


def build_features_for_request(
    route: str,
    year: int,
    month: int,
) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame for API inference.

    Uses route baseline demand for lag features since we don't
    have historical context for a real-time request.
    This is documented as a limitation — production would use
    a feature store with pre-computed lag values.
    """
    import pandas as pd

    quarter = (month - 1) // 3 + 1
    baseline_demand = ROUTE_BASELINE_DEMAND.get(route, 130.0)
    distance = ROUTE_DISTANCE.get(route, 2000)
    business_share = ROUTE_BUSINESS_SHARE.get(route, 0.30)
    route_code = ROUTE_CODES.get(route, 0)

    features = {
        # Temporal
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "quarter_sin": np.sin(2 * np.pi * quarter / 4),
        "quarter_cos": np.cos(2 * np.pi * quarter / 4),
        "is_peak_leisure": int(month in PEAK_LEISURE_MONTHS),
        "is_peak_business": int(month in PEAK_BUSINESS_MONTHS),
        "is_summer": int(month in [6, 7, 8]),
        "is_december": int(month == 12),
        "is_q1": int(quarter == 1),
        "years_since_2019": year - 2019,
        # Lag features — use baseline as approximation
        "lag_1m_passengers": baseline_demand,
        "lag_12m_passengers": baseline_demand,
        "rolling_3m_passengers": baseline_demand,
        "yoy_growth": 0.0,
        # Seasonality
        "seasonality_index": 1.0,
        # Capacity
        "load_factor": 0.82,  # historical average
        "lag_1m_load_factor": 0.82,
        "high_load_factor": 0,
        "low_load_factor": 0,
        # Route
        "route_distance_km": distance,
        "route_business_share": business_share,
        "is_long_haul": int(distance > 3000),
        "is_short_haul": int(distance < 2000),
        "route_encoded": route_code,
        # COVID
        "is_covid_period": 0,
        "is_covid_recovery": 0,
        "months_since_covid": max(0, (year - 2020) * 12 + month - 3),
    }

    return pd.DataFrame([features])
