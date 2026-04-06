"""
BTS (Bureau of Transportation Statistics) data loader.

Data sources:
- T-100 Segment Data: monthly passenger counts by route/carrier
- DB1B Market Data: quarterly OD fares and passenger volumes

HONEST SCOPE:
BTS T-100 provides real passenger counts — this is genuine
demand signal, not synthetic. It does NOT provide:
- Individual booking transactions
- Booking window data (how far in advance tickets were bought)
- Fare class breakdowns
- Load factor by fare class

These limitations are documented here and in README.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
from src.utils.logger import logger

# Routes we are modeling — must match configs/config.yaml
TARGET_ROUTES = {
    ("JFK", "LAX"),
    ("LAX", "JFK"),
    ("ORD", "MIA"),
    ("MIA", "ORD"),
    ("LAX", "SEA"),
    ("SEA", "LAX"),
}

# Carriers with consistent coverage 2019-2023
# Excludes bankrupt carriers and regional feeders
MAJOR_CARRIERS = {
    "AA",  # American Airlines
    "DL",  # Delta Air Lines
    "UA",  # United Airlines
    "WN",  # Southwest Airlines
    "B6",  # JetBlue Airways
    "AS",  # Alaska Airlines
    "NK",  # Spirit Airlines
    "F9",  # Frontier Airlines
}


class BTSLoader:
    """
    Loads and cleans BTS T-100 and DB1B data.
    Filters to target routes and validates data quality.
    """

    def __init__(self, data_dir: str = "data/external/bts_t100"):
        self.data_dir = Path(data_dir)
        self.db1b_dir = Path("data/external/bts_db1b")

    def load_t100(
        self,
        years: list[int],
        save_processed: bool = True,
    ) -> pd.DataFrame:
        """
        Load and clean BTS T-100 segment data.

        Steps:
        1. Load raw CSV files for each year
        2. Filter to target routes and major carriers
        3. Validate data quality
        4. Engineer basic time features
        5. Save processed parquet

        Returns cleaned DataFrame with real passenger demand.
        """
        logger.info(f"Loading BTS T-100 data for years: {years}")

        dfs = []
        for year in years:
            # BTS files may be named differently — handle variations
            possible_names = [
                f"T_T100_SEGMENT_US_CARRIER_ONLY_{year}.csv",
                f"T100_{year}.csv",
                f"{year}.csv",
            ]

            file_path = None
            for name in possible_names:
                candidate = self.data_dir / name
                if candidate.exists():
                    file_path = candidate
                    break

            if file_path is None:
                logger.warning(
                    f"BTS T-100 file not found for {year} in {self.data_dir}. "
                    f"Skipping. Expected one of: {possible_names}"
                )
                continue

            logger.info(f"Loading {file_path}")
            df = pd.read_csv(file_path, low_memory=False)
            df["YEAR"] = year
            dfs.append(df)
            logger.info(f"Loaded {len(df):,} rows for {year}")

        if not dfs:
            raise FileNotFoundError(
                f"No BTS T-100 files found in {self.data_dir}. "
                f"Download from: https://www.transtats.bts.gov"
            )

        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total raw rows: {len(df):,}")

        df = self._clean_t100(df)
        logger.info(f"Rows after cleaning: {len(df):,}")

        if save_processed:
            output_path = Path("data/processed/t100_cleaned.parquet")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved processed T-100 to {output_path}")

        return df

    def _clean_t100(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and filter T-100 data.
        Documents every filter decision with rationale.
        """
        original_len = len(df)

        # Standardize column names — BTS uses all-caps
        df.columns = df.columns.str.upper().str.strip()

        # Verify required columns exist
        required_cols = {
            "YEAR",
            "MONTH",
            "UNIQUE_CARRIER",
            "ORIGIN",
            "DEST",
            "PASSENGERS",
            "SEATS",
        }
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"BTS T-100 missing required columns: {missing_cols}. "
                f"Available: {list(df.columns)}"
            )

        # Filter to target routes
        # Rationale: we model 3 specific routes — all others are noise
        route_mask = df.apply(
            lambda r: (r["ORIGIN"], r["DEST"]) in TARGET_ROUTES, axis=1
        )
        df = df[route_mask].copy()
        logger.info(
            f"After route filter: {len(df):,} rows "
            f"({original_len - len(df):,} removed)"
        )

        # Filter to major carriers
        # Rationale: bankrupt/regional carriers have gaps that corrupt
        # time series — model stability requires consistent coverage
        df = df[df["UNIQUE_CARRIER"].isin(MAJOR_CARRIERS)].copy()
        logger.info(f"After carrier filter: {len(df):,} rows")

        # Remove rows with zero or negative passengers
        # Rationale: operational zeros (cancelled flights) not demand zeros
        df = df[df["PASSENGERS"] > 0].copy()

        # Remove rows with zero seats — data quality issue
        df = df[df["SEATS"] > 0].copy()

        # Create route identifier
        df["ROUTE"] = df["ORIGIN"] + "-" + df["DEST"]

        # Create load factor
        # Note: this is carrier-reported, not booking-curve load factor
        df["LOAD_FACTOR"] = (df["PASSENGERS"] / df["SEATS"]).clip(0, 1)

        # Aggregate to route-month level
        # Rationale: we model route-level demand, not carrier-level
        df = (
            df.groupby(["YEAR", "MONTH", "ROUTE", "ORIGIN", "DEST"])
            .agg(
                total_passengers=("PASSENGERS", "sum"),
                total_seats=("SEATS", "sum"),
                total_departures=("DEPARTURES_PERFORMED", "sum"),
                num_carriers=("UNIQUE_CARRIER", "nunique"),
                avg_load_factor=("LOAD_FACTOR", "mean"),
            )
            .reset_index()
        )

        # Per-departure demand — correct scale for single-flight optimization
        # Route-level passengers / departures = avg passengers per flight
        # This is what one flight actually sees — optimization scale
        df["passengers_per_departure"] = (
            df["total_passengers"] / df["total_departures"]
        ).round(1)

        df["seats_per_departure"] = (df["total_seats"] / df["total_departures"]).round(
            1
        )

        # Create date column
        df["DATE"] = pd.to_datetime(
            df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str).str.zfill(2) + "-01"
        )

        # Sort by route and date
        df = df.sort_values(["ROUTE", "DATE"]).reset_index(drop=True)

        logger.info(f"Final cleaned shape: {df.shape}")
        logger.info(f"Routes present: {df['ROUTE'].unique().tolist()}")
        logger.info(f"Date range: {df['DATE'].min()} to {df['DATE'].max()}")

        return df

    def load_db1b(
        self,
        years: list[int],
        save_processed: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Load BTS DB1B market fare data.
        Used for price range calibration only.

        Returns None if files not found — DB1B is optional.
        Pipeline continues without it using config price bounds.
        """
        logger.info(f"Loading BTS DB1B data for years: {years}")

        dfs = []
        for year in years:
            possible_names = [
                f"Origin_and_Destination_Survey_DB1BMarket_{year}_1.csv",
                f"DB1B_{year}.csv",
                f"{year}_db1b.csv",
            ]

            file_path = None
            for name in possible_names:
                candidate = self.db1b_dir / name
                if candidate.exists():
                    file_path = candidate
                    break

            if file_path is None:
                logger.warning(f"DB1B file not found for {year} — skipping")
                continue

            df = pd.read_csv(file_path, low_memory=False)
            dfs.append(df)

        if not dfs:
            logger.warning(
                "No DB1B files found. Price bounds will use config defaults. "
                "Download from: https://www.transtats.bts.gov for better calibration."
            )
            return None

        df = pd.concat(dfs, ignore_index=True)
        return self._clean_db1b(df)

    def _clean_db1b(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DB1B fare data for price range calibration."""
        df.columns = df.columns.str.upper().str.strip()

        required = {"YEAR", "ORIGIN", "DEST", "MARKET_FARE", "PASSENGERS"}
        missing = required - set(df.columns)
        if missing:
            logger.warning(f"DB1B missing columns: {missing} — skipping")
            return df

        # Filter to target routes
        route_mask = df.apply(
            lambda r: (r["ORIGIN"], r["DEST"]) in TARGET_ROUTES, axis=1
        )
        df = df[route_mask].copy()

        # Remove fare outliers — below $30 or above $5000 are data errors
        df = df[(df["MARKET_FARE"] >= 30) & (df["MARKET_FARE"] <= 5000)].copy()

        # Aggregate fare statistics by route
        fare_stats = (
            df.groupby(["ORIGIN", "DEST"])
            .agg(
                fare_p10=("MARKET_FARE", lambda x: x.quantile(0.10)),
                fare_p50=("MARKET_FARE", lambda x: x.quantile(0.50)),
                fare_p90=("MARKET_FARE", lambda x: x.quantile(0.90)),
                fare_mean=("MARKET_FARE", "mean"),
                total_passengers=("PASSENGERS", "sum"),
            )
            .reset_index()
        )

        fare_stats["ROUTE"] = fare_stats["ORIGIN"] + "-" + fare_stats["DEST"]
        logger.info(
            f"DB1B fare stats computed for routes: {fare_stats['ROUTE'].tolist()}"
        )
        return fare_stats

    def get_data_quality_report(self, df: pd.DataFrame) -> dict:
        """
        Generates a data quality summary.
        Run this after loading — review before proceeding to features.
        """
        report = {
            "total_rows": len(df),
            "date_range": {
                "start": str(df["DATE"].min()),
                "end": str(df["DATE"].max()),
            },
            "routes": df["ROUTE"].unique().tolist(),
            "missing_months": {},
            "covid_period_rows": 0,
        }

        # Check for missing months per route
        for route in df["ROUTE"].unique():
            route_df = df[df["ROUTE"] == route]
            expected_months = pd.date_range(
                df["DATE"].min(), df["DATE"].max(), freq="MS"
            )
            actual_months = set(route_df["DATE"])
            missing = [str(m) for m in expected_months if m not in actual_months]
            if missing:
                report["missing_months"][route] = missing

        # Flag COVID period — 2020-03 to 2021-06
        covid_mask = (df["DATE"] >= "2020-03-01") & (df["DATE"] <= "2021-06-01")
        report["covid_period_rows"] = int(covid_mask.sum())
        report["covid_pct"] = round(report["covid_period_rows"] / len(df) * 100, 1)

        return report
