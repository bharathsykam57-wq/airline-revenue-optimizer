"""
Great Expectations data validation contracts for BTS data.

These are NOT optional checks. Data that fails validation
does not enter the pipeline. Period.

Why this matters:
BTS data has known quality issues:
- Carrier code changes after mergers
- Missing months during operational disruptions
- Reporting errors in passenger counts
- COVID period anomalies

Catching these at ingestion prevents silent model corruption.
"""

import pandas as pd
from src.utils.logger import logger


class BTSValidator:
    """
    Validates BTS T-100 data against defined contracts.
    Fails loudly on critical violations.
    Warns on non-critical issues.
    """

    def validate_t100(self, df: pd.DataFrame) -> dict:
        """
        Run all validation checks on cleaned T-100 data.
        Returns validation report with pass/fail/warn per check.
        """
        logger.info("Running BTS T-100 validation contracts")
        results = {}

        results["schema"] = self._check_schema(df)
        results["date_range"] = self._check_date_range(df)
        results["passengers_range"] = self._check_passengers_range(df)
        results["load_factor_range"] = self._check_load_factor_range(df)
        results["route_coverage"] = self._check_route_coverage(df)
        results["no_future_dates"] = self._check_no_future_dates(df)

        # Summarize
        passed = sum(1 for r in results.values() if r["status"] == "PASS")
        warned = sum(1 for r in results.values() if r["status"] == "WARN")
        failed = sum(1 for r in results.values() if r["status"] == "FAIL")

        logger.info(f"Validation complete: {passed} PASS, {warned} WARN, {failed} FAIL")

        if failed > 0:
            failed_checks = [k for k, v in results.items() if v["status"] == "FAIL"]
            raise ValueError(
                f"Data validation FAILED on: {failed_checks}. "
                f"Pipeline halted — review data before proceeding."
            )

        return results

    def _check_schema(self, df: pd.DataFrame) -> dict:
        required_columns = {
            "YEAR",
            "MONTH",
            "ROUTE",
            "ORIGIN",
            "DEST",
            "total_passengers",
            "total_seats",
            "avg_load_factor",
            "DATE",
        }
        missing = required_columns - set(df.columns)
        if missing:
            return {"status": "FAIL", "message": f"Missing columns: {missing}"}
        return {"status": "PASS", "message": "All required columns present"}

    def _check_date_range(self, df: pd.DataFrame) -> dict:
        min_date = df["DATE"].min()
        max_date = df["DATE"].max()
        years_covered = df["YEAR"].nunique()

        if years_covered < 3:
            return {
                "status": "WARN",
                "message": (
                    f"Only {years_covered} years of data. "
                    f"Minimum 3 recommended for seasonality modeling."
                ),
            }
        return {
            "status": "PASS",
            "message": f"Date range: {min_date} to {max_date} ({years_covered} years)",
        }

    def _check_passengers_range(self, df: pd.DataFrame) -> dict:
        min_pax = df["total_passengers"].min()
        max_pax = df["total_passengers"].max()
        zero_rows = (df["total_passengers"] == 0).sum()

        if zero_rows > 0:
            return {
                "status": "FAIL",
                "message": f"{zero_rows} rows with zero passengers — cleaning failed",
            }
        if max_pax > 1_000_000:
            return {
                "status": "WARN",
                "message": f"Unusually high passenger count: {max_pax:,} — verify",
            }
        return {
            "status": "PASS",
            "message": f"Passengers range: {min_pax:,} to {max_pax:,}",
        }

    def _check_load_factor_range(self, df: pd.DataFrame) -> dict:
        invalid = df[(df["avg_load_factor"] < 0) | (df["avg_load_factor"] > 1)]
        if len(invalid) > 0:
            return {
                "status": "FAIL",
                "message": f"{len(invalid)} rows with load factor outside [0,1]",
            }
        return {
            "status": "PASS",
            "message": f"Load factor range valid: "
            f"{df['avg_load_factor'].min():.3f} to "
            f"{df['avg_load_factor'].max():.3f}",
        }

    def _check_route_coverage(self, df: pd.DataFrame) -> dict:
        expected_routes = {
            "JFK-LAX",
            "LAX-JFK",
            "ORD-MIA",
            "MIA-ORD",
            "LAX-SEA",
            "SEA-LAX",
        }
        actual_routes = set(df["ROUTE"].unique())
        missing_routes = expected_routes - actual_routes

        if missing_routes:
            return {
                "status": "WARN",
                "message": (
                    f"Missing routes: {missing_routes}. "
                    f"Model will only cover: {actual_routes}"
                ),
            }
        return {
            "status": "PASS",
            "message": f"All expected routes present: {actual_routes}",
        }

    def _check_no_future_dates(self, df: pd.DataFrame) -> dict:
        future_rows = df[df["DATE"] > pd.Timestamp.now()]
        if len(future_rows) > 0:
            return {
                "status": "FAIL",
                "message": f"{len(future_rows)} rows with future dates — data error",
            }
        return {"status": "PASS", "message": "No future dates found"}
