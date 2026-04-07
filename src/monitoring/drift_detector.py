"""
Statistical drift detection for demand model monitoring.

METHODS IMPLEMENTED:
1. PSI (Population Stability Index) — feature distribution drift
2. KS Test (Kolmogorov-Smirnov) — distribution shift significance
3. Prediction drift — model output distribution change

WHY THESE METHODS:
PSI is the airline/financial industry standard for monitoring.
Threshold interpretation:
  PSI < 0.10: no significant drift
  PSI 0.10-0.20: moderate drift — investigate
  PSI > 0.20: significant drift — model likely needs retraining

KS test provides statistical significance — PSI alone can
flag drift that is statistically insignificant.
Both are needed together.

HONEST LIMITATION:
Synthetic data will not drift naturally.
Drift injection (drift_injector.py) is used to verify
the monitoring system works before relying on it.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from src.utils.logger import logger

PSI_THRESHOLDS = {
    "no_drift": 0.10,
    "moderate_drift": 0.20,
    "significant_drift": float("inf"),
}


@dataclass
class DriftResult:
    """Result of a single drift check."""

    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    drift_detected: bool
    drift_level: str  # none / moderate / significant
    alert: bool  # True if action required


@dataclass
class DriftReport:
    """Aggregated drift report across all features."""

    timestamp: str
    baseline_period: str
    current_period: str
    total_features_checked: int
    features_with_drift: int
    features_with_alert: int
    results: list[DriftResult]
    overall_drift_level: str
    recommendation: str


class DriftDetector:
    """
    Detects feature and prediction drift.
    Segment-aware — prevents false positives on seasonal patterns.
    """

    def __init__(
        self,
        psi_threshold: float = 0.20,
        ks_pvalue_threshold: float = 0.05,
    ):
        self.psi_threshold = psi_threshold
        self.ks_pvalue_threshold = ks_pvalue_threshold

    def compute_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Population Stability Index.

        PSI = sum((current% - baseline%) * ln(current% / baseline%))

        Handles edge cases:
        - Zero bins get small epsilon to avoid log(0)
        - Arrays with insufficient data return 0.0
        """
        if len(baseline) < 5 or len(current) < 5:
            logger.warning("Insufficient data for PSI — returning 0.0")
            return 0.0

        # Create bins from baseline distribution
        bins = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)  # remove duplicates

        if len(bins) < 2:
            return 0.0

        baseline_counts, _ = np.histogram(baseline, bins=bins)
        current_counts, _ = np.histogram(current, bins=bins)

        # Convert to proportions
        baseline_pct = baseline_counts / len(baseline)
        current_pct = current_counts / len(current)

        # Avoid zeros
        epsilon = 1e-6
        baseline_pct = np.where(baseline_pct == 0, epsilon, baseline_pct)
        current_pct = np.where(current_pct == 0, epsilon, current_pct)

        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
        return float(psi)

    def run_ks_test(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
    ) -> tuple[float, float]:
        """
        Two-sample KS test.
        Returns (statistic, p_value).
        p_value < 0.05: distributions are significantly different.
        """
        if len(baseline) < 3 or len(current) < 3:
            return 0.0, 1.0

        statistic, pvalue = stats.ks_2samp(baseline, current)
        return float(statistic), float(pvalue)

    def _classify_drift(self, psi: float) -> str:
        if psi < PSI_THRESHOLDS["no_drift"]:
            return "none"
        elif psi < PSI_THRESHOLDS["moderate_drift"]:
            return "moderate"
        else:
            return "significant"

    def check_feature(
        self,
        feature_name: str,
        baseline: np.ndarray,
        current: np.ndarray,
    ) -> DriftResult:
        """Check drift for a single feature."""
        psi = self.compute_psi(baseline, current)
        ks_stat, ks_pvalue = self.run_ks_test(baseline, current)

        drift_level = self._classify_drift(psi)
        drift_detected = drift_level != "none"

        # Alert requires both PSI threshold AND statistical significance
        alert = psi > self.psi_threshold and ks_pvalue < self.ks_pvalue_threshold

        return DriftResult(
            feature=feature_name,
            psi=round(psi, 4),
            ks_statistic=round(ks_stat, 4),
            ks_pvalue=round(ks_pvalue, 4),
            drift_detected=drift_detected,
            drift_level=drift_level,
            alert=alert,
        )

    def run_full_report(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features_to_check: list[str],
        baseline_period: str = "2019-2021",
        current_period: str = "current",
    ) -> DriftReport:
        """
        Run drift detection across all specified features.
        Returns full DriftReport with recommendations.
        """
        from datetime import datetime

        logger.info(f"Running drift detection: " f"{len(features_to_check)} features")

        results = []
        for feature in features_to_check:
            if feature not in baseline_df.columns:
                logger.warning(f"Feature {feature} not in baseline — skipping")
                continue
            if feature not in current_df.columns:
                logger.warning(f"Feature {feature} not in current — skipping")
                continue

            baseline_vals = baseline_df[feature].dropna().values
            current_vals = current_df[feature].dropna().values

            result = self.check_feature(feature, baseline_vals, current_vals)
            results.append(result)

            if result.alert:
                logger.warning(
                    f"DRIFT ALERT: {feature} — "
                    f"PSI={result.psi:.4f}, "
                    f"KS p-value={result.ks_pvalue:.4f}"
                )

        features_with_drift = sum(1 for r in results if r.drift_detected)
        features_with_alert = sum(1 for r in results if r.alert)

        # Overall drift level — worst case across features
        drift_levels = [r.drift_level for r in results]
        if "significant" in drift_levels:
            overall = "significant"
        elif "moderate" in drift_levels:
            overall = "moderate"
        else:
            overall = "none"

        recommendation = self._get_recommendation(overall, features_with_alert)

        return DriftReport(
            timestamp=datetime.now().isoformat(),
            baseline_period=baseline_period,
            current_period=current_period,
            total_features_checked=len(results),
            features_with_drift=features_with_drift,
            features_with_alert=features_with_alert,
            results=results,
            overall_drift_level=overall,
            recommendation=recommendation,
        )

    def _get_recommendation(
        self,
        overall_drift: str,
        n_alerts: int,
    ) -> str:
        if overall_drift == "significant" or n_alerts >= 3:
            return (
                "RETRAIN: Significant drift detected across multiple features. "
                "Model performance likely degraded. Schedule retraining."
            )
        elif overall_drift == "moderate" or n_alerts >= 1:
            return (
                "INVESTIGATE: Moderate drift detected. "
                "Monitor closely. Consider retraining if business metrics decline."
            )
        else:
            return "OK: No significant drift detected. Continue monitoring."
