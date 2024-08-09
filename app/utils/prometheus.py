from fastapi import Request
from prometheus_client import Counter, Histogram, generate_latest
import time

# Metrics definitions
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)


async def metrics_middleware(request: Request, call_next):
    """
    Middleware to track metrics for each request.

    Records request count and duration for Prometheus monitoring.

    Args:
        request: Incoming HTTP request
        call_next: Next middleware in chain

    Returns:
        HTTP response with recorded metrics
    """
    start_time = time.time()

    response = await call_next(request)

    # Record metrics
    duration = time.time() - start_time
    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)

    return response


def generate_metrics():
    """
    Generate Prometheus metrics in text format.

    Returns:
        Metrics data for Prometheus scraping
    """
    return generate_latest()