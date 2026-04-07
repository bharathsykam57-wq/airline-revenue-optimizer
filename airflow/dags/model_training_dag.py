"""
Weekly model retraining DAG.
Retrains all route models and logs to MLflow.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    "owner": "arof",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}


def run_feature_pipeline(**context):
    """Build features from latest processed data."""
    import sys

    sys.path.insert(0, "/opt/airflow")
    import pandas as pd
    from src.features.feature_engineer import FeatureEngineer

    df = pd.read_parquet("/opt/airflow/data/processed/t100_cleaned.parquet")
    train_df = df[df["DATE"] < "2022-01-01"].copy()

    fe = FeatureEngineer()
    train_features = fe.fit_transform(train_df)
    train_features.to_parquet(
        "/opt/airflow/data/features/train_features.parquet", index=False
    )
    print(f"Features built: {train_features.shape}")


def run_model_training(**context):
    """Retrain all route demand models."""
    import sys

    sys.path.insert(0, "/opt/airflow")
    from src.modeling.trainer import DemandModelTrainer

    trainer = DemandModelTrainer()
    metrics = trainer.run()
    routes_trained = len(metrics)
    print(f"Models trained: {routes_trained} routes")
    return routes_trained


def run_monitoring_check(**context):
    """Run post-training drift check."""
    import sys

    sys.path.insert(0, "/opt/airflow")
    from src.monitoring.monitor import MonitoringPipeline

    pipeline = MonitoringPipeline()
    report = pipeline.run_drift_check()
    print(f"Drift check: {report['overall_drift_level']}")
    if report["features_with_alert"] > 3:
        raise ValueError(
            f"Post-training drift check failed: "
            f"{report['features_with_alert']} features drifted"
        )


with DAG(
    dag_id="model_training",
    default_args=default_args,
    description="Weekly model retraining pipeline",
    schedule_interval="0 2 * * 0",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["training", "weekly"],
) as dag:
    start = EmptyOperator(task_id="start")

    build_features = PythonOperator(
        task_id="build_features",
        python_callable=run_feature_pipeline,
    )

    train_models = PythonOperator(
        task_id="train_models",
        python_callable=run_model_training,
    )

    check_drift = PythonOperator(
        task_id="post_training_drift_check",
        python_callable=run_monitoring_check,
    )

    end = EmptyOperator(task_id="end")

    start >> build_features >> train_models >> check_drift >> end
