"""
Aggregates backtesting results into summary metrics.
Every metric prefixed with honest data framing.
"""

import pandas as pd
from src.utils.logger import logger


class BacktestMetrics:
    """
    Computes and formats backtesting metrics.
    """

    def summarize(self, results_df: pd.DataFrame) -> dict:
        """
        Compute summary metrics across all routes and windows.
        """
        if len(results_df) == 0:
            logger.warning("No results to summarize")
            return {}

        summary = {
            "data_disclaimer": (
                "All results under BTS T-100 real demand data + "
                "synthetic price elasticity assumptions. "
                "Revenue figures are per-flight estimates."
            ),
            "total_evaluations": len(results_df),
            "routes_evaluated": results_df["route"].nunique(),
            "windows_evaluated": results_df["window_id"].nunique(),
        }

        # Demand accuracy
        summary["mean_demand_error_pct"] = round(
            results_df["demand_error_pct"].mean(), 2
        )
        summary["median_demand_error_pct"] = round(
            results_df["demand_error_pct"].median(), 2
        )

        # Prediction interval coverage
        coverage = (
            (results_df["actual_demand"] >= results_df["predicted_demand_q10"])
            & (results_df["actual_demand"] <= results_df["predicted_demand_q90"])
        ).mean()
        summary["prediction_interval_coverage"] = round(float(coverage), 3)

        # Revenue uplift
        summary["mean_revenue_uplift_pct"] = round(
            results_df["revenue_uplift_vs_baseline_pct"].mean(), 2
        )
        summary["median_revenue_uplift_pct"] = round(
            results_df["revenue_uplift_vs_baseline_pct"].median(), 2
        )
        summary["revenue_uplift_std"] = round(
            results_df["revenue_uplift_vs_baseline_pct"].std(), 2
        )

        # Regret vs oracle
        summary["mean_regret_pct"] = round(results_df["regret_vs_oracle_pct"].mean(), 2)

        # Per-route breakdown
        per_route = (
            results_df.groupby("route")
            .agg(
                mean_uplift=("revenue_uplift_vs_baseline_pct", "mean"),
                mean_regret=("regret_vs_oracle_pct", "mean"),
                mean_demand_error=("demand_error_pct", "mean"),
                n_periods=("route", "count"),
            )
            .round(2)
            .reset_index()
        )
        summary["per_route"] = per_route.to_dict("records")

        return summary

    def print_report(self, summary: dict) -> None:
        """Print formatted backtesting report."""
        print()
        print("=" * 65)
        print("BACKTESTING REPORT")
        print(f"⚠️  {summary.get('data_disclaimer', '')}")
        print("=" * 65)
        print(f"Evaluations:  {summary['total_evaluations']}")
        print(f"Routes:       {summary['routes_evaluated']}")
        print(f"Windows:      {summary['windows_evaluated']}")
        print()
        print("DEMAND ACCURACY")
        print(f"  Mean error:   {summary['mean_demand_error_pct']:.1f}%")
        print(f"  Median error: {summary['median_demand_error_pct']:.1f}%")
        print(
            f"  PI coverage:  "
            f"{summary['prediction_interval_coverage']:.3f} "
            f"(target: 0.80+)"
        )
        print()
        print("REVENUE PERFORMANCE")
        print(
            f"  Mean uplift vs baseline:  " f"{summary['mean_revenue_uplift_pct']:.2f}%"
        )
        print(
            f"  Median uplift vs baseline: "
            f"{summary['median_revenue_uplift_pct']:.2f}%"
        )
        print(f"  Uplift std (risk):         " f"{summary['revenue_uplift_std']:.2f}%")
        print(f"  Mean regret vs oracle:     " f"{summary['mean_regret_pct']:.2f}%")
        print()
        print("PER-ROUTE BREAKDOWN")
        for r in summary["per_route"]:
            print(
                f"  {r['route']:8} | "
                f"uplift={r['mean_uplift']:+.1f}% | "
                f"regret={r['mean_regret']:.1f}% | "
                f"demand_err={r['mean_demand_error']:.1f}%"
            )
        print("=" * 65)
