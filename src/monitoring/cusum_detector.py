"""
CUSUM (Cumulative Sum) control chart for revenue monitoring.

WHY CUSUM OVER SIMPLE THRESHOLD:
Simple threshold: alert if revenue < X
Problem: misses gradual decline that never crosses the threshold

CUSUM: accumulates small deviations over time
Problem: a 5% revenue drop every month for 3 months
         = 15% cumulative decline
         Simple threshold misses each individual month
         CUSUM detects the cumulative trend

This is the correct approach for revenue monitoring in
production ML systems.
"""

from dataclasses import dataclass, field
import numpy as np
from src.utils.logger import logger


@dataclass
class CUSUMState:
    """Current state of the CUSUM detector."""

    cusum_positive: float = 0.0  # upper CUSUM — detects increase
    cusum_negative: float = 0.0  # lower CUSUM — detects decrease
    n_observations: int = 0
    alerts: list[dict] = field(default_factory=list)
    target_revenue: float = 0.0


class CUSUMDetector:
    """
    Two-sided CUSUM detector for revenue monitoring.

    Parameters:
    - k: allowance parameter — half the minimum detectable shift
    - h: decision threshold — how large CUSUM must be before alerting
    """

    def __init__(
        self,
        k: float = 0.5,  # allowance — half of minimum detectable shift
        h: float = 5.0,  # threshold — triggers alert
    ):
        self.k = k
        self.h = h
        self.state = CUSUMState()

    def set_target(self, historical_revenues: list[float]) -> None:
        """
        Set target revenue from historical data.
        Call this once with baseline period revenues.
        """
        self.state.target_revenue = float(np.mean(historical_revenues))
        logger.info(
            f"CUSUM target set: {self.state.target_revenue:.2f} "
            f"(mean of {len(historical_revenues)} observations)"
        )

    def update(
        self,
        observed_revenue: float,
        period: str = "",
    ) -> bool:
        """
        Update CUSUM with new observation.
        Returns True if drift detected.

        Standardized deviation:
        z = (observed - target) / target  (relative change)
        """
        if self.state.target_revenue == 0:
            logger.warning("CUSUM target not set — call set_target() first")
            return False

        # Standardized deviation
        z = (observed_revenue - self.state.target_revenue) / self.state.target_revenue

        # Update both-sided CUSUM
        self.state.cusum_positive = max(0, self.state.cusum_positive + z - self.k)
        self.state.cusum_negative = max(0, self.state.cusum_negative - z - self.k)
        self.state.n_observations += 1

        # Check for alert
        alert_positive = self.state.cusum_positive > self.h
        alert_negative = self.state.cusum_negative > self.h

        if alert_positive or alert_negative:
            direction = "increase" if alert_positive else "decrease"
            alert = {
                "period": period,
                "observed_revenue": round(observed_revenue, 2),
                "target_revenue": round(self.state.target_revenue, 2),
                "deviation_pct": round(z * 100, 2),
                "cusum_positive": round(self.state.cusum_positive, 4),
                "cusum_negative": round(self.state.cusum_negative, 4),
                "direction": direction,
            }
            self.state.alerts.append(alert)

            logger.warning(
                f"CUSUM ALERT: Revenue {direction} detected. "
                f"Period={period}, "
                f"Observed={observed_revenue:.2f}, "
                f"Target={self.state.target_revenue:.2f}, "
                f"Deviation={z*100:.1f}%"
            )

            # Reset after alert
            self.state.cusum_positive = 0.0
            self.state.cusum_negative = 0.0

            return True

        return False

    def get_status(self) -> dict:
        """Current CUSUM status."""
        return {
            "target_revenue": round(self.state.target_revenue, 2),
            "cusum_positive": round(self.state.cusum_positive, 4),
            "cusum_negative": round(self.state.cusum_negative, 4),
            "n_observations": self.state.n_observations,
            "total_alerts": len(self.state.alerts),
            "threshold": self.h,
        }
