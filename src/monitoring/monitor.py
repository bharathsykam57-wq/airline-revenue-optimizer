"""
Orchestrates full monitoring pipeline.
Runs on schedule via Airflow DAG (implemented in Day 10).
Can also be triggered manually or via API endpoint.
"""

import json
from pathlib import Path
from dataclasses import asdict
import pandas as pd
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.cusum_detector import CUSUMDetector
from src.utils.logger import logger

MONITORING_OUTPUT_DIR = "data/monitoring"

# Features most important to monitor
# Chosen because they are high-impact on predictions
FEATURES_TO_MONITOR = [
    "lag_1m_passengers",
    "lag_12m_passengers",
    "rolling_3m_passengers",
    "load_factor",
    "seasonality_index",
    "yoy_growth",
]


class MonitoringPipeline:
    """
    Runs drift detection and revenue monitoring.
    Saves reports to data/monitoring/.
    """

    def __init__(self):
        self.drift_detector = DriftDetector(
            psi_threshold=0.20,
            ks_pvalue_threshold=0.05,
        )
        self.cusum_detector = CUSUMDetector(k=0.5, h=5.0)
        Path(MONITORING_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    def run_drift_check(
        self,
        baseline_features_path: str = "data/features/train_features.parquet",
        current_features_path: str = "data/features/val_features.parquet",
    ) -> dict:
        """
        Compare current feature distributions against baseline.
        Uses pre-COVID baseline only (2019 data) if available.
        COVID period (2020-03 to 2021-06) has anomalous distributions and is excluded from baseline to avoid false positive alerts. See GitHub Issue #XX.
        """
        logger.info("Running feature drift check")

        baseline_df = pd.read_parquet(baseline_features_path)
        current_df = pd.read_parquet(current_features_path)

        # Use pre-COVID baseline only — 2019 data
        # COVID period (2020-03 to 2021-06) has anomalous distributions
        # Including it in baseline causes false positive alerts on all features
        # This is documented in GitHub Issue #XX
        if "year" in baseline_df.columns:
            pre_covid_baseline = baseline_df[baseline_df["year"] == 2019].copy()
            if len(pre_covid_baseline) > 10:
                baseline_df = pre_covid_baseline
                logger.info(
                    f"Using pre-COVID baseline: {len(baseline_df)} rows (2019 only)"
                )
            else:
                logger.warning("Insufficient pre-COVID data — using full baseline")

        report = self.drift_detector.run_full_report(
            baseline_df=baseline_df,
            current_df=current_df,
            features_to_check=FEATURES_TO_MONITOR,
            baseline_period="2019 pre-COVID (stable baseline)",
            current_period="2022 (validation)",
        )

        # Save report
        report_dict = asdict(report)
        output_path = Path(MONITORING_OUTPUT_DIR) / "drift_report.json"
        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        logger.info(
            f"Drift report saved: {report.features_with_alert} alerts, overall={report.overall_drift_level}"
        )
        return report_dict

    def run_cusum_check(
        self,
        backtesting_results_path: str = "data/backtesting/rolling_window_results.parquet",
    ) -> dict:
        """
        Run CUSUM on backtesting revenue results.
        """
        logger.info("Running CUSUM revenue monitoring")

        results_df = pd.read_parquet(backtesting_results_path)

        # Use optimized revenue per route
        cusum_results = {}

        for route in results_df["route"].unique():
            route_df = results_df[results_df["route"] == route].sort_values(
                "period_start"
            )

            revenues = route_df["optimized_revenue"].tolist()

            if len(revenues) < 4:
                continue

            # Set target from first half
            mid = len(revenues) // 2
            self.cusum_detector = CUSUMDetector(k=0.5, h=5.0)
            self.cusum_detector.set_target(revenues[:mid])

            alerts = []
            for i, (rev, period) in enumerate(
                zip(revenues[mid:], route_df["period_start"].tolist()[mid:])
            ):
                alert_fired = self.cusum_detector.update(rev, str(period))
                if alert_fired:
                    alerts.append(str(period))

            cusum_results[route] = {
                "n_periods": len(revenues),
                "mean_revenue": round(sum(revenues) / len(revenues), 2),
                "alerts_fired": len(alerts),
                "alert_periods": alerts,
                "status": self.cusum_detector.get_status(),
            }

        output_path = Path(MONITORING_OUTPUT_DIR) / "cusum_report.json"
        with open(output_path, "w") as f:
            json.dump(cusum_results, f, indent=2)

        logger.info("CUSUM report saved")
        return cusum_results

    def run_drift_injection_test(self) -> dict:
        """
        Verify monitoring system works by injecting known drift.
        Should ALWAYS detect the injected drift.
        If it doesn't — monitoring is broken.
        """
        from src.monitoring.drift_injector import DriftInjector

        logger.info("Running drift injection test")

        baseline_df = pd.read_parquet("data/features/train_features.parquet")

        injector = DriftInjector()

        # Inject strong drift on lag_1m_passengers
        drifted_df = injector.inject_feature_drift(
            baseline_df.copy(),
            feature="lag_1m_passengers",
            shift_magnitude=2.0,  # 2 standard deviations — strong drift
        )

        report = self.drift_detector.run_full_report(
            baseline_df=baseline_df,
            current_df=drifted_df,
            features_to_check=["lag_1m_passengers"],
            baseline_period="original",
            current_period="injected_drift_2std",
        )

        detected = report.features_with_alert > 0

        result = {
            "test": "drift_injection_2std",
            "drift_detected": detected,
            "psi": report.results[0].psi if report.results else 0,
            "alert": report.results[0].alert if report.results else False,
            "verdict": "PASS" if detected else "FAIL — monitoring broken",
        }

        logger.info(
            f"Drift injection test: {result['verdict']} " f"(PSI={result['psi']:.4f})"
        )
        return result
