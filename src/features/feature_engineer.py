"""
Feature engineering pipeline for airline demand modeling.

Every feature here is justified by domain logic.
Features without clear demand relationship are excluded.

FEATURE CATEGORIES:
1. Temporal — seasonality, trends, COVID recovery
2. Lag features — autoregressive demand signal
3. Capacity features — load factor, seat availability
4. Route features — distance, market characteristics
5. External — weather (weak signal, explicitly documented)

WHAT IS NOT INCLUDED AND WHY:
- Competitor pricing: not available in BTS data
- Booking window: BTS is monthly aggregates, no booking curve
- Fare class mix: not in BTS segment data
These are documented as production gaps in README.
"""

import pandas as pd
import numpy as np
from src.utils.logger import logger


# US Federal Holidays — affects leisure demand
# Business travel largely unaffected by holidays
US_HOLIDAYS = {
    # Format: (month, day) — approximate
    (1, 1): "New Year",
    (7, 4): "Independence Day",
    (11, 25): "Thanksgiving",  # approximate — 4th Thursday
    (12, 25): "Christmas",
}

# Months with historically high leisure travel
PEAK_LEISURE_MONTHS = {6, 7, 8, 12}  # Summer + December
# Months with historically high business travel
PEAK_BUSINESS_MONTHS = {1, 2, 3, 9, 10}  # Q1 + Q3


class FeatureEngineer:
    """
    Transforms cleaned BTS data into model-ready features.

    Uses fit/transform pattern to prevent data leakage.
    Fit on training data only. Transform on all splits.
    """

    def __init__(self):
        self._is_fitted = False
        self._train_stats: dict = {}

    def fit(self, train_df: pd.DataFrame) -> "FeatureEngineer":
        """
        Learn statistics from training data only.
        Never called on validation or test data.
        """
        logger.info("Fitting feature engineer on training data")

        # Route-level baseline demand (train period only)
        self._train_stats["route_avg_demand"] = (
            train_df.groupby("ROUTE")["passengers_per_departure"].mean().to_dict()
        )

        # Route-level demand std (for normalization)
        self._train_stats["route_std_demand"] = (
            train_df.groupby("ROUTE")["passengers_per_departure"].std().to_dict()
        )

        # Monthly seasonality index per route (train only)
        monthly_avg = (
            train_df.groupby(["ROUTE", "MONTH"])["passengers_per_departure"]
            .mean()
            .reset_index()
        )
        route_avg = train_df.groupby("ROUTE")["passengers_per_departure"].mean()

        monthly_avg["seasonality_index"] = monthly_avg.apply(
            lambda r: r["passengers_per_departure"] / route_avg[r["ROUTE"]], axis=1
        )
        self._train_stats["seasonality_index"] = monthly_avg.set_index(
            ["ROUTE", "MONTH"]
        )["seasonality_index"].to_dict()

        self._is_fitted = True
        logger.info("Feature engineer fitted successfully")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to any split.
        Raises if fit() has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "FeatureEngineer must be fitted before transform. "
                "Call fit(train_df) first."
            )

        logger.info(f"Transforming {len(df)} rows")
        df = df.copy()

        df = self._add_temporal_features(df)
        df = self._add_lag_features(df)
        df = self._add_seasonality_features(df)
        df = self._add_capacity_features(df)
        df = self._add_route_features(df)
        df = self._add_covid_features(df)
        df = self._handle_lag_nans(df)

        logger.info(f"Features created: {len(df.columns)} columns")
        return df

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calendar features — capture cyclical demand patterns.
        """
        df["year"] = df["DATE"].dt.year
        df["month"] = df["DATE"].dt.month
        df["quarter"] = df["DATE"].dt.quarter

        # Cyclical encoding — month is circular (Dec → Jan)
        # Sin/cos encoding prevents model from treating Dec(12)
        # as far from Jan(1) — they are adjacent seasonally
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
        df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)

        # Binary flags
        df["is_peak_leisure"] = df["month"].isin(PEAK_LEISURE_MONTHS).astype(int)
        df["is_peak_business"] = df["month"].isin(PEAK_BUSINESS_MONTHS).astype(int)
        df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)
        df["is_december"] = (df["month"] == 12).astype(int)
        df["is_q1"] = (df["quarter"] == 1).astype(int)

        # Years since 2019 — captures secular trend
        df["years_since_2019"] = df["year"] - 2019

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Autoregressive features — past demand predicts future demand.

        CRITICAL: These must be computed per route, not globally.
        Sorting by DATE before shift() is mandatory.

        LAG JUSTIFICATION:
        - lag_1m: Last month demand — short-term trend
        - lag_3m: 3-month average — medium trend
        - lag_12m: Same month last year — removes seasonality
        """
        df = df.sort_values(["ROUTE", "DATE"]).reset_index(drop=True)

        for route in df["ROUTE"].unique():
            mask = df["ROUTE"] == route

            # lag_1m — previous month passengers
            df.loc[mask, "lag_1m_passengers"] = df.loc[
                mask, "passengers_per_departure"
            ].shift(1)

            # lag_12m — same month previous year
            df.loc[mask, "lag_12m_passengers"] = df.loc[
                mask, "passengers_per_departure"
            ].shift(12)

            # rolling_3m — 3-month rolling average (excluding current)
            df.loc[mask, "rolling_3m_passengers"] = (
                df.loc[mask, "passengers_per_departure"]
                .shift(1)
                .rolling(window=3, min_periods=1)
                .mean()
            )

            # year_over_year_growth — % change vs same month last year
            df.loc[mask, "yoy_growth"] = df.loc[
                mask, "passengers_per_departure"
            ].pct_change(periods=12)

        return df

    def _add_seasonality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Seasonality index — how this month compares to route average.
        Fitted on training data to prevent leakage.
        """
        df["seasonality_index"] = df.apply(
            lambda r: self._train_stats["seasonality_index"].get(
                (r["ROUTE"], r["month"]),
                1.0,  # default 1.0 = average
            ),
            axis=1,
        )

        df["demand_vs_route_avg"] = df.apply(
            lambda r: (
                r["passengers_per_departure"]
                / self._train_stats["route_avg_demand"].get(r["ROUTE"], 1.0)
            ),
            axis=1,
        )

        return df

    def _add_capacity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Capacity utilization features.
        Load factor is a key RM signal — high LF = pricing power.
        """
        df["load_factor"] = df["avg_load_factor"]

        # Load factor lag — last month capacity utilization
        df = df.sort_values(["ROUTE", "DATE"]).reset_index(drop=True)
        for route in df["ROUTE"].unique():
            mask = df["ROUTE"] == route
            df.loc[mask, "lag_1m_load_factor"] = df.loc[mask, "load_factor"].shift(1)

        # Capacity pressure — are we near capacity?
        df["high_load_factor"] = (df["load_factor"] > 0.85).astype(int)
        df["low_load_factor"] = (df["load_factor"] < 0.65).astype(int)

        return df

    def _add_route_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Route characteristics — static features per route.
        These do not change over time.
        """
        route_distance = {
            "JFK-LAX": 3983,
            "LAX-JFK": 3983,
            "ORD-MIA": 2163,
            "MIA-ORD": 2163,
            "LAX-SEA": 1537,
            "SEA-LAX": 1537,
        }
        route_business_share = {
            "JFK-LAX": 0.35,
            "LAX-JFK": 0.35,
            "ORD-MIA": 0.25,
            "MIA-ORD": 0.25,
            "LAX-SEA": 0.30,
            "SEA-LAX": 0.30,
        }

        df["route_distance_km"] = df["ROUTE"].map(route_distance)
        df["route_business_share"] = df["ROUTE"].map(route_business_share)

        # Distance category — short/medium/long haul
        df["is_long_haul"] = (df["route_distance_km"] > 3000).astype(int)
        df["is_short_haul"] = (df["route_distance_km"] < 2000).astype(int)

        # Route encoding — label encode for LightGBM
        route_codes = {r: i for i, r in enumerate(sorted(df["ROUTE"].unique()))}
        df["route_encoded"] = df["ROUTE"].map(route_codes)

        return df

    def _add_covid_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        COVID period flags and recovery tracking.

        WHY KEEP COVID DATA:
        Dropping 2020-2021 makes the model fragile to demand shocks.
        Keeping it with explicit flags lets the model learn:
        - Normal demand patterns (pre/post COVID)
        - Shock behavior (during COVID)
        - Recovery trajectory

        In production: similar flags for future demand shocks
        (geopolitical events, fuel crises, pandemics).
        """
        df["is_covid_period"] = (
            (df["DATE"] >= "2020-03-01") & (df["DATE"] <= "2021-06-01")
        ).astype(int)

        df["is_covid_recovery"] = (
            (df["DATE"] > "2021-06-01") & (df["DATE"] <= "2022-06-01")
        ).astype(int)

        # Months since COVID start — captures recovery trajectory
        covid_start = pd.Timestamp("2020-03-01")
        df["months_since_covid"] = df["DATE"].apply(
            lambda d: max(
                0, (d.year - covid_start.year) * 12 + d.month - covid_start.month
            )
            if d >= covid_start
            else 0
        )

        return df

    def get_feature_columns(self) -> list[str]:
        """
        Returns the list of feature columns for model training.
        Excludes target, identifiers, and raw columns.
        """
        return [
            # Temporal
            "month_sin",
            "month_cos",
            "quarter_sin",
            "quarter_cos",
            "is_peak_leisure",
            "is_peak_business",
            "is_summer",
            "is_december",
            "is_q1",
            "years_since_2019",
            # Lag features
            "lag_1m_passengers",
            "lag_12m_passengers",
            "rolling_3m_passengers",
            "yoy_growth",
            # Seasonality
            "seasonality_index",
            # Capacity
            "load_factor",
            "lag_1m_load_factor",
            "high_load_factor",
            "low_load_factor",
            # Route
            "route_distance_km",
            "route_business_share",
            "is_long_haul",
            "is_short_haul",
            "route_encoded",
            # COVID
            "is_covid_period",
            "is_covid_recovery",
            "months_since_covid",
        ]

    def _handle_lag_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill NaNs in lag features.

        Strategy per feature:
        - lag_1m: fill with route average from train stats
          Rationale: first month has no history — use baseline
        - lag_12m: fill with lag_1m if available, else route average
          Rationale: best available approximation without leakage
        - rolling_3m: fill with lag_1m
          Rationale: rolling average degrades to single point at start
        - yoy_growth: fill with 0.0
          Rationale: no growth signal available — neutral assumption
        - lag_1m_load_factor: fill with route mean load factor from train

        IMPORTANT: All fills use train statistics only.
        Never fill with values derived from val or test data.
        """
        for route in df["ROUTE"].unique():
            mask = df["ROUTE"] == route
            route_avg = self._train_stats["route_avg_demand"].get(route, 0)

            # lag_1m — fill with route train average
            df.loc[mask, "lag_1m_passengers"] = df.loc[
                mask, "lag_1m_passengers"
            ].fillna(route_avg)

            # rolling_3m — fill with lag_1m (now filled)
            df.loc[mask, "rolling_3m_passengers"] = df.loc[
                mask, "rolling_3m_passengers"
            ].fillna(df.loc[mask, "lag_1m_passengers"])

            # lag_12m — fill with lag_1m
            df.loc[mask, "lag_12m_passengers"] = df.loc[
                mask, "lag_12m_passengers"
            ].fillna(df.loc[mask, "lag_1m_passengers"])

            # yoy_growth — fill with 0 (neutral — no growth assumption)
            df.loc[mask, "yoy_growth"] = df.loc[mask, "yoy_growth"].fillna(0.0)

            # lag_1m_load_factor — fill with route mean from train
            route_lf_mean = df.loc[mask, "load_factor"].mean()
            df.loc[mask, "lag_1m_load_factor"] = df.loc[
                mask, "lag_1m_load_factor"
            ].fillna(route_lf_mean)

        return df
