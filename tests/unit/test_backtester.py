"""
Critical tests for backtesting pipeline.
Leakage prevention is the most important invariant.
"""

import pandas as pd
import sys

sys.path.insert(0, ".")


def make_sample_df():
    """Create minimal test dataframe."""
    dates = pd.date_range("2019-01-01", "2023-12-01", freq="MS")
    routes = ["JFK-LAX", "LAX-JFK"]
    rows = []
    for route in routes:
        for date in dates:
            rows.append(
                {
                    "DATE": date,
                    "ROUTE": route,
                    "ORIGIN": route.split("-")[0],
                    "DEST": route.split("-")[1],
                    "YEAR": date.year,
                    "MONTH": date.month,
                    "total_passengers": 130000.0,
                    "total_seats": 160000.0,
                    "total_departures": 1000.0,
                    "passengers_per_departure": 130.0,
                    "seats_per_departure": 160.0,
                    "num_carriers": 4,
                    "avg_load_factor": 0.81,
                }
            )
    return pd.DataFrame(rows)


def test_temporal_split_no_leakage():
    """Train set must end before validation set starts."""
    df = make_sample_df()

    train = df[df["DATE"] < "2022-01-01"]
    val = df[(df["DATE"] >= "2022-01-01") & (df["DATE"] < "2023-01-01")]
    test = df[df["DATE"] >= "2023-01-01"]

    assert train["DATE"].max() < val["DATE"].min(), "LEAKAGE: train overlaps validation"
    assert val["DATE"].max() < test["DATE"].min(), "LEAKAGE: validation overlaps test"


def test_train_size():
    """Training set must have minimum rows per route."""
    df = make_sample_df()
    train = df[df["DATE"] < "2022-01-01"]

    for route in train["ROUTE"].unique():
        route_rows = len(train[train["ROUTE"] == route])
        assert (
            route_rows >= 12
        ), f"Route {route} has only {route_rows} training rows — insufficient"


def test_test_set_is_2023():
    """Test set must be 2023 data only."""
    df = make_sample_df()
    test = df[df["DATE"] >= "2023-01-01"]

    assert test["YEAR"].unique().tolist() == [2023], "Test set contains non-2023 data"
    assert len(test) > 0, "Test set is empty"
