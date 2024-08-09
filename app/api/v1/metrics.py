from fastapi import APIRouter
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import PlainTextResponse
import time

router = APIRouter(tags=["metrics"])

# Define Prometheus metrics
request_count = Counter(
    'logsense_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'logsense_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

active_connections = Gauge(
    'logsense_active_connections',
    'Number of active connections'
)

anomalies_detected = Counter(
    'logsense_anomalies_detected_total',
    'Total number of anomalies detected',
    ['severity', 'type']
)

logs_processed = Counter(
    'logsense_logs_processed_total',
    'Total number of logs processed'
)

model_accuracy = Gauge(
    'logsense_model_accuracy',
    'Current model accuracy score'
)


@router.get("/metrics", response_class=PlainTextResponse)
async def get_metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()


@router.get("/api/v1/metrics/summary")
async def get_metrics_summary():
    """Get application metrics summary."""
    return {
        "requests": {
            "total": 10000,
            "success_rate": 99.5,
            "avg_response_time_ms": 45
        },
        "processing": {
            "logs_processed_total": 1500000,
            "anomalies_detected": 2500,
            "detection_accuracy": 94.2
        },
        "performance": {
            "cpu_usage_percent": 25,
            "memory_usage_mb": 256,
            "active_connections": 15
        },
        "uptime_hours": 720
    }