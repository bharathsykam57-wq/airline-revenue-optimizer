"""Daily monitoring DAG."""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    "owner": "arof",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def run_drift_detection(**context):
    import sys

    sys.path.insert(0, "/opt/airflow")
    from src.monitoring.monitor import MonitoringPipeline

    pipeline = MonitoringPipeline()
    report = pipeline.run_drift_check()
    print(
        f"Drift: {report['overall_drift_level']}, "
        f"alerts: {report['features_with_alert']}"
    )


def run_cusum_check(**context):
    import sys

    sys.path.insert(0, "/opt/airflow")
    from src.monitoring.monitor import MonitoringPipeline

    pipeline = MonitoringPipeline()
    results = pipeline.run_cusum_check()
    total_alerts = sum(r["alerts_fired"] for r in results.values())
    print(f"CUSUM: {total_alerts} revenue alerts fired")


with DAG(
    dag_id="monitoring",
    default_args=default_args,
    description="Daily monitoring — drift and revenue checks",
    schedule_interval="0 8 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["monitoring", "daily"],
) as dag:
    start = EmptyOperator(task_id="start")

    drift_check = PythonOperator(
        task_id="drift_detection",
        python_callable=run_drift_detection,
    )

    cusum_check = PythonOperator(
        task_id="cusum_revenue_check",
        python_callable=run_cusum_check,
    )

    end = EmptyOperator(task_id="end")

    start >> [drift_check, cusum_check] >> end
