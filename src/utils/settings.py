"""
Central settings management.
All secrets flow through here via pydantic-settings.
No os.getenv() calls anywhere else in the codebase.
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    env: str = Field(default="development", alias="ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # External APIs
    aviationstack_api_key: str = Field(..., alias="AVIATIONSTACK_API_KEY")
    openweather_api_key: str = Field(..., alias="OPENWEATHER_API_KEY")

    # Redis
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    redis_password: str = Field(default="", alias="REDIS_PASSWORD")

    # MLflow
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000", alias="MLFLOW_TRACKING_URI"
    )
    mlflow_experiment_name: str = Field(
        default="arof-demand-model", alias="MLFLOW_EXPERIMENT_NAME"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        populate_by_name = True


@lru_cache()
def get_settings() -> Settings:
    """
    Cached singleton — .env parsed once at startup.
    lru_cache prevents re-reading on every call.
    """
    return Settings()
