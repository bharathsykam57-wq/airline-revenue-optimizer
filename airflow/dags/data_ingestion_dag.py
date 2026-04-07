"""
Data ingestion DAG.

HONEST SCOPE NOTE:
Airflow is included to demonstrate orchestration pattern knowledge.
At this data scale (360 rows, 3 routes), cron jobs would suffice.
Airflow becomes necessary when pipelines have:
- Complex task dependencies
- SLA requirements
- Team-level visibility needs
- Multiple data sources with different failure modes

This DAG demonstrates the pattern — not operational necessity.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    "owner": "arof",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}


def fetch_weather_data(**context):
    """Fetch weather for all route airports."""
    import sys

    sys.path.insert(0, "/opt/airflow")
    from src.ingestion.openweather_client import OpenWeatherClient

    client = OpenWeatherClient()
    routes = ["JFK-LAX", "LAX-JFK", "ORD-MIA", "MIA-ORD", "LAX-SEA", "SEA-LAX"]
    weather = client.get_weather_for_routes(routes)
    print(f"Weather fetched for {len(weather)} airports")
    return len(weather)


def fetch_aviation_data(**context):
    """Fetch route metadata from Aviationstack."""
    import sys

    sys.path.insert(0, "/opt/airflow")
    from src.ingestion.aviationstack_client import AviationstackClient

    client = AviationstackClient()
    routes = [("JFK", "LAX"), ("ORD", "MIA"), ("LAX", "SEA")]
    results = []
    for origin, dest in routes:
        data = client.get_routes(origin, dest)
        if data:
            results.append(f"{origin}-{dest}")
    print(f"Aviation data fetched for routes: {results}")
    return len(results)


def validate_data(**context):
    """Run Great Expectations validation on latest data."""
    import sys

    sys.path.insert(0, "/opt/airflow")
    import pandas as pd
    from src.validation.bts_validator import BTSValidator

    df = pd.read_parquet("/opt/airflow/data/processed/t100_cleaned.parquet")
    validator = BTSValidator()
    results = validator.validate_t100(df)
    failed = [k for k, v in results.items() if v["status"] == "FAIL"]
    if failed:
        raise ValueError(f"Data validation failed: {failed}")
    print(f"Validation passed: {len(results)} checks")


with DAG(
    dag_id="data_ingestion",
    default_args=default_args,
    description="Daily data ingestion — weather and aviation metadata",
    schedule_interval="0 6 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ingestion", "daily"],
) as dag:
    start = EmptyOperator(task_id="start")

    fetch_weather = PythonOperator(
        task_id="fetch_weather",
        python_callable=fetch_weather_data,
    )

    fetch_aviation = PythonOperator(
        task_id="fetch_aviation",
        python_callable=fetch_aviation_data,
    )

    validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    end = EmptyOperator(task_id="end")

    start >> [fetch_weather, fetch_aviation] >> validate >> end
