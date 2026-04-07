"""Tests for feature engineering — leakage and NaN are critical."""

import pandas as pd
import numpy as np
import pytest
import sys

sys.path.insert(0, ".")

from src.features.feature_engineer import FeatureEngineer


def make_sample_df(n_months=36, route="JFK-LAX"):
    dates = pd.date_range("2019-01-01", periods=n_months, freq="MS")
    return pd.DataFrame(
        {
            "DATE": dates,
            "ROUTE": route,
            "ORIGIN": "JFK",
            "DEST": "LAX",
            "YEAR": dates.year,
            "MONTH": dates.month,
            "total_passengers": np.random.uniform(100000, 160000, n_months),
            "total_seats": np.random.uniform(150000, 180000, n_months),
            "total_departures": np.random.uniform(900, 1100, n_months),
            "passengers_per_departure": np.random.uniform(100, 155, n_months),
            "seats_per_departure": np.random.uniform(140, 165, n_months),
            "num_carriers": 4,
            "avg_load_factor": np.random.uniform(0.70, 0.95, n_months),
        }
    )


def test_fit_transform_no_nans():
    """Feature pipeline must produce zero NaNs after fitting."""
    df = make_sample_df(36)
    fe = FeatureEngineer()
    features = fe.fit_transform(df)

    feature_cols = fe.get_feature_columns()
    nan_counts = features[feature_cols].isna().sum()
    assert (
        nan_counts.sum() == 0
    ), f"NaNs found in features: {nan_counts[nan_counts > 0]}"


def test_transform_without_fit_raises():
    """Transform before fit must raise RuntimeError."""
    df = make_sample_df(24)
    fe = FeatureEngineer()

    with pytest.raises(RuntimeError, match="must be fitted"):
        fe.transform(df)


def test_feature_columns_count():
    """Must produce exactly 27 feature columns."""
    df = make_sample_df(36)
    fe = FeatureEngineer()
    fe.fit(df)

    assert (
        len(fe.get_feature_columns()) == 27
    ), f"Expected 27 features, got {len(fe.get_feature_columns())}"


def test_cyclical_encoding_range():
    """Sin/cos features must be in [-1, 1]."""
    df = make_sample_df(36)
    fe = FeatureEngineer()
    features = fe.fit_transform(df)

    for col in ["month_sin", "month_cos", "quarter_sin", "quarter_cos"]:
        assert features[col].between(-1, 1).all(), f"{col} has values outside [-1, 1]"
