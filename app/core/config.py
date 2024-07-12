from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Configuration for server, ML models, databases, and cloud deployment.
    """

    # Application
    APP_NAME: str = "LogSense AI"
    VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Server
    PORT: int = 8080
    HOST: str = "0.0.0.0"
    WORKERS: int = 1

    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "https://*.run.app"  # Cloud Run URLs
    ]

    # Machine Learning
    ML_MODEL_PATH: str = "/tmp/models"
    ML_CONFIDENCE_THRESHOLD: float = 0.85
    ML_ANOMALY_RATIO: float = 0.1
    ML_MIN_SAMPLES: int = 50

    # Redis (optional for caching)
    REDIS_URL: str = ""
    CACHE_TTL: int = 3600  # 1 hour

    # MongoDB (optional for storage)
    MONGODB_URL: str = ""
    DATABASE_NAME: str = "logsense"

    # Log Processing
    MAX_LOG_SIZE_MB: int = 100
    BATCH_SIZE: int = 1000
    SUPPORTED_FORMATS: List[str] = [
        "json",
        "apache_combined",
        "apache_common",
        "nginx",
        "syslog",
        "csv",
        "custom"
    ]

    # Security
    API_KEY: str = ""
    ENABLE_AUTH: bool = False

    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PATH: str = "/metrics"

    # Cloud Run specific
    CLOUD_RUN: bool = os.getenv("K_SERVICE", None) is not None
    GCP_PROJECT_ID: str = ""
    GCP_REGION: str = "us-central1"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()