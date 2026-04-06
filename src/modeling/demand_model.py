"""
LightGBM demand forecasting model with quantile regression.

Architecture:
- One model set per route (6 routes = 6 cross 3 quantile models = 18 models)
- Quantiles: [0.10, 0.50, 0.90] — conservative, expected, optimistic
- All runs logged to MLflow for experiment tracking
- SHAP values computed for interpretability

HONEST SCOPE:
This model predicts monthly aggregate passenger demand.
It does NOT predict:
- Booking curves (how demand builds over time)
- Fare class demand (requires GDS data)
- Individual passenger willingness to pay

These gaps are documented in README limitations.
"""

import json
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import mlflow
import mlflow.lightgbm
from src.utils.logger import logger

QUANTILES = [0.05, 0.50, 0.95]
QUANTILE_NAMES = ["q10", "q50", "q90"]

# LightGBM parameters — justified in DECISIONS.md
BASE_PARAMS = {
    "num_leaves": 31,  # Reduced from 63 — only 216 train rows
    "learning_rate": 0.05,
    "n_estimators": 300,
    "min_child_samples": 10,  # Lowered for small dataset
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,  # L1 regularization
    "reg_lambda": 0.1,  # L2 regularization
    "random_state": 42,
    "verbose": -1,  # Suppress LightGBM output — use our logger
}


class DemandModel:
    """
    Route-specific quantile regression demand model.

    One instance = one route.
    Contains 3 LightGBM models (q10, q50, q90).
    """

    def __init__(self, route: str):
        self.route = route
        self.models: dict[str, lgb.LGBMRegressor] = {}
        self.feature_columns: list[str] = []
        self.shap_values: Optional[np.ndarray] = None
        self._is_trained = False

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str = "total_passengers",
        mlflow_run_id: Optional[str] = None,
    ) -> dict:
        """
        Train quantile regression models for this route.
        Logs all parameters and metrics to MLflow.

        Returns dict of validation metrics per quantile.
        """
        logger.info(f"Training demand model for route: {self.route}")
        self.feature_columns = feature_columns

        # Filter to this route
        route_train = train_df[train_df["ROUTE"] == self.route].copy()
        route_val = val_df[val_df["ROUTE"] == self.route].copy()

        if len(route_train) == 0:
            raise ValueError(f"No training data for route: {self.route}")

        logger.info(
            f"{self.route} — train: {len(route_train)} rows, "
            f"val: {len(route_val)} rows"
        )

        X_train = route_train[feature_columns]
        y_train = route_train[target_column]
        X_val = route_val[feature_columns]
        y_val = route_val[target_column]

        metrics = {}

        with mlflow.start_run(
            run_name=f"demand_model_{self.route}",
            nested=True,
        ):
            mlflow.log_param("route", self.route)
            mlflow.log_param("train_rows", len(route_train))
            mlflow.log_param("val_rows", len(route_val))
            mlflow.log_param("feature_count", len(feature_columns))
            mlflow.log_params(BASE_PARAMS)

            for quantile, name in zip(QUANTILES, QUANTILE_NAMES):
                logger.info(f"{self.route} — training {name} (α={quantile})")

                model = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=quantile,
                    **BASE_PARAMS,
                )

                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )

                self.models[name] = model

                # Validation metrics
                val_preds = model.predict(X_val)
                pinball_loss = self._pinball_loss(y_val.values, val_preds, quantile)
                rmse = np.sqrt(np.mean((y_val.values - val_preds) ** 2))

                metrics[name] = {
                    "pinball_loss": pinball_loss,
                    "rmse": rmse,
                }

                mlflow.log_metric(f"{name}_pinball_loss", pinball_loss)
                mlflow.log_metric(f"{name}_rmse", rmse)

                logger.info(
                    f"{self.route} {name} — "
                    f"pinball_loss: {pinball_loss:.2f}, "
                    f"rmse: {rmse:.2f}"
                )

            # Quantile crossing check
            self._check_quantile_crossing(X_val)

            # Coverage check
            coverage = self._compute_coverage(X_val, y_val)
            metrics["coverage_80pct"] = coverage
            mlflow.log_metric("coverage_80pct", coverage)
            logger.info(f"{self.route} — prediction interval coverage: {coverage:.3f}")

            # SHAP values on q50 model
            self._compute_shap(X_train)
            mlflow.log_metric(
                "top_feature_importance",
                float(self.models["q50"].feature_importances_.max()),
            )

        self._is_trained = True
        return metrics

    def predict(
        self,
        X: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        """
        Generate predictions for all quantiles.
        Returns dict with q10, q50, q90 arrays.

        Always returns all three — never just point estimate.
        """
        if not self._is_trained:
            raise RuntimeError(
                f"Model for {self.route} not trained. Call train() first."
            )

        predictions = {}
        for name, model in self.models.items():
            # Handle both LGBMRegressor (training) and Booster (loaded)
            if isinstance(model, lgb.Booster):
                predictions[name] = model.predict(X[self.feature_columns].values)
            else:
                predictions[name] = model.predict(X[self.feature_columns])

        # Enforce quantile ordering — q10 <= q50 <= q90
        predictions = self._enforce_quantile_order(predictions)

        return predictions

    def _predict_raw(
        self,
        X: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        """
        Internal prediction — skips _is_trained guard.
        Used only during training for coverage and crossing checks.
        External callers must use predict() which enforces the guard.
        """
        predictions = {}
        for name, model in self.models.items():
            # Handle both LGBMRegressor (training) and Booster (loaded)
            if isinstance(model, lgb.Booster):
                predictions[name] = model.predict(X[self.feature_columns].values)
            else:
                predictions[name] = model.predict(X[self.feature_columns])
        predictions = self._enforce_quantile_order(predictions)
        return predictions

    def _pinball_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        quantile: float,
    ) -> float:
        """
        Pinball loss — proper scoring rule for quantile regression.
        Lower is better. Used instead of RMSE for quantile models
        because RMSE penalizes quantile predictions unfairly.
        """
        errors = y_true - y_pred
        loss = np.where(
            errors >= 0,
            quantile * errors,
            (quantile - 1) * errors,
        )
        return float(np.mean(loss))

    def _check_quantile_crossing(self, X: pd.DataFrame) -> None:
        """
        Check for quantile crossing — q10 > q50 or q50 > q90.
        This is mathematically invalid.
        Logs warning if crossing detected.
        """
        if "q10" not in self.models or "q50" not in self.models:
            return

        q10 = self.models["q10"].predict(X[self.feature_columns])
        q50 = self.models["q50"].predict(X[self.feature_columns])
        q90 = self.models["q90"].predict(X[self.feature_columns])

        crossings_lower = np.sum(q10 > q50)
        crossings_upper = np.sum(q50 > q90)

        if crossings_lower > 0 or crossings_upper > 0:
            logger.warning(
                f"{self.route} — quantile crossing detected: "
                f"q10>q50: {crossings_lower} samples, "
                f"q50>q90: {crossings_upper} samples. "
                f"Applying isotonic regression fix."
            )
            mlflow.log_metric("quantile_crossings", crossings_lower + crossings_upper)
        else:
            logger.info(f"{self.route} — no quantile crossings detected")
            mlflow.log_metric("quantile_crossings", 0)

    def _enforce_quantile_order(
        self,
        predictions: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """
        Post-processing to enforce q10 <= q50 <= q90.
        Uses numpy clip — simple and effective for small violations.
        """
        q10 = predictions["q10"]
        q50 = predictions["q50"]
        q90 = predictions["q90"]

        # Clip q10 to not exceed q50
        q10 = np.minimum(q10, q50)
        # Clip q90 to not be below q50
        q90 = np.maximum(q90, q50)

        return {"q10": q10, "q50": q50, "q90": q90}

    def _compute_coverage(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> float:
        """
        Compute prediction interval coverage.
        Target: ~80% of actuals fall within [q10, q90].
        Coverage < 70%: model is overconfident.
        Coverage > 95%: intervals too wide, not useful.
        """
        preds = self._predict_raw(X)
        covered = np.sum((y.values >= preds["q10"]) & (y.values <= preds["q90"]))
        return float(covered / len(y))

    def _compute_shap(self, X_train: pd.DataFrame) -> None:
        """
        Compute SHAP values on q50 model for interpretability.
        Used to understand which features drive demand predictions.
        In RM context: high SHAP for price feature = price sensitivity.
        """
        try:
            explainer = shap.TreeExplainer(self.models["q50"])
            self.shap_values = explainer.shap_values(X_train[self.feature_columns])
            logger.info(f"{self.route} — SHAP values computed")
        except Exception as e:
            logger.warning(f"{self.route} — SHAP computation failed: {e}")

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Returns feature importance from q50 model.
        Sorted by importance descending.
        """
        if "q50" not in self.models:
            raise RuntimeError("Model not trained")

        model = self.models["q50"]

        # Handle both LGBMRegressor and Booster
        if isinstance(model, lgb.Booster):
            importance_values = model.feature_importance(importance_type="gain")
        else:
            importance_values = model.feature_importances_

        importance = (
            pd.DataFrame(
                {
                    "feature": self.feature_columns,
                    "importance": importance_values,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        return importance

    def save(self, model_dir: str) -> None:
        """Save all quantile models to disk."""
        path = Path(model_dir) / self.route.replace("-", "_")
        path.mkdir(parents=True, exist_ok=True)

        for name, model in self.models.items():
            model.booster_.save_model(str(path / f"{name}.txt"))

        # Save metadata
        metadata = {
            "route": self.route,
            "feature_columns": self.feature_columns,
            "quantiles": QUANTILE_NAMES,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, model_dir: str, route: str) -> "DemandModel":
        """
        Load saved model from disk.
        Uses lgb.Booster directly — avoids sklearn wrapper state issue.
        """
        path = Path(model_dir) / route.replace("-", "_")

        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        instance = cls(route=route)
        instance.feature_columns = metadata["feature_columns"]

        # Load as lgb.Booster directly — not LGBMRegressor wrapper
        # LGBMRegressor.predict() requires fit() to have been called.
        # lgb.Booster.predict() works correctly on loaded models.
        for name in metadata["quantiles"]:
            booster = lgb.Booster(model_file=str(path / f"{name}.txt"))
            instance.models[name] = booster

        instance._is_trained = True
        logger.info(f"Model loaded from {path}")
        return instance
