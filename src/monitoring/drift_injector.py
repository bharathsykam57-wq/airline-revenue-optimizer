"""
Deliberate drift injection for monitoring system verification.

WHY THIS EXISTS:
Synthetic data does not drift naturally.
A monitoring system that never triggers has never been tested.
This module injects known drift patterns so we can verify
the detection system catches them correctly.

This is NOT used in production data paths.
It is ONLY used for testing and demonstration.
All injected data is clearly labeled.

In production: drift occurs naturally from market changes,
seasonal shifts, competitor actions, and demand shocks.
"""

import numpy as np
import pandas as pd
from src.utils.logger import logger


class DriftInjector:
    """
    Injects synthetic drift into feature or demand data.
    Used exclusively for monitoring system verification.
    """

    def inject_feature_drift(
        self,
        df: pd.DataFrame,
        feature: str,
        shift_magnitude: float,
        noise_scale: float = 0.1,
    ) -> pd.DataFrame:
        """
        Shift a feature's distribution by shift_magnitude.
        Simulates gradual feature drift.

        shift_magnitude=0.5 means shift by 0.5 standard deviations
        """
        df = df.copy()
        feature_std = df[feature].std()
        shift = shift_magnitude * feature_std

        df[feature] = (
            df[feature]
            + shift
            + np.random.normal(0, noise_scale * feature_std, len(df))
        )
        df["_drift_injected"] = True
        df["_drift_feature"] = feature
        df["_drift_magnitude"] = shift_magnitude

        logger.info(
            f"Injected feature drift: {feature}, "
            f"shift={shift:.4f} ({shift_magnitude} std)"
        )
        return df

    def inject_concept_drift(
        self,
        df: pd.DataFrame,
        demand_column: str,
        scale_factor: float,
    ) -> pd.DataFrame:
        """
        Scale demand values to simulate concept drift.
        scale_factor=0.7 means demand drops to 70% of original.
        Simulates demand shock or model degradation.
        """
        df = df.copy()
        original_mean = df[demand_column].mean()
        df[demand_column] = df[demand_column] * scale_factor
        new_mean = df[demand_column].mean()

        df["_drift_injected"] = True
        df["_drift_type"] = "concept"
        df["_scale_factor"] = scale_factor

        logger.info(
            f"Injected concept drift: {demand_column}, "
            f"demand {original_mean:.1f} → {new_mean:.1f} "
            f"(scale={scale_factor})"
        )
        return df

    def inject_revenue_drop(
        self,
        revenues: list[float],
        drop_pct: float,
        start_idx: int,
    ) -> list[float]:
        """
        Inject gradual revenue decline starting at start_idx.
        Used to test CUSUM detector.

        drop_pct=0.15 means 15% revenue drop applied gradually.
        """
        revenues = revenues.copy()
        drop_per_period = drop_pct / (len(revenues) - start_idx)

        for i in range(start_idx, len(revenues)):
            cumulative_drop = drop_per_period * (i - start_idx + 1)
            revenues[i] = revenues[i] * (1 - cumulative_drop)

        logger.info(
            f"Injected revenue drop: {drop_pct*100:.0f}% over "
            f"{len(revenues) - start_idx} periods starting at idx {start_idx}"
        )
        return revenues
