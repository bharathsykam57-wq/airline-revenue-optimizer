"""
Orchestrates training of all route-specific demand models.
Handles MLflow experiment setup, model persistence,
and cross-route metric aggregation.
"""

from pathlib import Path
import pandas as pd
import mlflow
from src.modeling.demand_model import DemandModel
from src.features.feature_engineer import FeatureEngineer
from src.utils.logger import logger
from src.utils.settings import get_settings

MODEL_DIR = "models/demand"
ROUTES = ["JFK-LAX", "LAX-JFK", "ORD-MIA", "MIA-ORD", "LAX-SEA", "SEA-LAX"]


class DemandModelTrainer:
    """
    Trains and evaluates demand models for all routes.
    """

    def __init__(self):
        self.settings = get_settings()
        self.models: dict[str, DemandModel] = {}

    def run(
        self,
        data_path: str = "data/processed/t100_cleaned.parquet",
    ) -> dict:
        """
        Full training pipeline:
        1. Load and split data
        2. Engineer features
        3. Train per-route models
        4. Evaluate and log to MLflow
        5. Save models
        """
        logger.info("Starting demand model training pipeline")

        # Setup MLflow
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        mlflow.set_experiment(self.settings.mlflow_experiment_name)

        # Load data
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(df)} rows from {data_path}")

        # Time-based split — NO leakage
        train_df = df[df["DATE"] < "2022-01-01"].copy()
        val_df = df[(df["DATE"] >= "2022-01-01") & (df["DATE"] < "2023-01-01")].copy()
        test_df = df[df["DATE"] >= "2023-01-01"].copy()

        logger.info(
            f"Split — train: {len(train_df)}, "
            f"val: {len(val_df)}, "
            f"test: {len(test_df)}"
        )

        # Feature engineering — fit on train only
        fe = FeatureEngineer()
        train_features = fe.fit_transform(train_df)
        val_features = fe.transform(val_df)
        # Test split is intentionally not transformed here.
        # Test evaluation happens in src/evaluation/backtesting.py
        # to enforce strict separation between training and evaluation.

        feature_cols = fe.get_feature_columns()
        logger.info(f"Features: {len(feature_cols)} columns")

        # Train all routes
        all_metrics = {}

        with mlflow.start_run(run_name="demand_model_all_routes"):
            mlflow.log_param("routes", ROUTES)
            mlflow.log_param("train_period", "2019-01 to 2021-12")
            mlflow.log_param("val_period", "2022-01 to 2022-12")
            mlflow.log_param("test_period", "2023-01 to 2023-12")
            mlflow.log_param("feature_count", len(feature_cols))
            mlflow.log_param(
                "model_note",
                "Trained on BTS T-100 real demand data. "
                "Booking window is synthetic — see README limitations.",
            )

            for route in ROUTES:
                logger.info(f"Training route: {route}")
                model = DemandModel(route=route)

                try:
                    metrics = model.train(
                        train_df=train_features,
                        val_df=val_features,
                        feature_columns=feature_cols,
                    )
                    self.models[route] = model
                    all_metrics[route] = metrics

                    # Save model
                    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
                    model.save(MODEL_DIR)

                except Exception as e:
                    logger.error(f"Training failed for {route}: {e}")
                    raise

            # Aggregate metrics across routes
            self._log_aggregate_metrics(all_metrics)

        logger.info("Training pipeline complete")
        return all_metrics

    def _log_aggregate_metrics(self, all_metrics: dict) -> None:
        """Log mean metrics across all routes."""
        q50_rmses = [m["q50"]["rmse"] for m in all_metrics.values() if "q50" in m]
        coverages = [
            m["coverage_80pct"] for m in all_metrics.values() if "coverage_80pct" in m
        ]

        if q50_rmses:
            mean_rmse = sum(q50_rmses) / len(q50_rmses)
            mlflow.log_metric("mean_q50_rmse_all_routes", mean_rmse)
            logger.info(f"Mean q50 RMSE across routes: {mean_rmse:.2f}")

        if coverages:
            mean_coverage = sum(coverages) / len(coverages)
            mlflow.log_metric("mean_coverage_all_routes", mean_coverage)
            logger.info(f"Mean prediction interval coverage: {mean_coverage:.3f}")

            if mean_coverage < 0.70:
                logger.warning(
                    f"Coverage {mean_coverage:.3f} below 0.70 threshold. "
                    f"Model is overconfident — consider wider quantiles."
                )
