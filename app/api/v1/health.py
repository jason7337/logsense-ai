from fastapi import APIRouter
from datetime import datetime
import psutil
import platform

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint.

    Returns basic service status and version information.

    Returns:
        Health status with timestamp and version
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "LogSense AI",
        "version": "1.0.0",
        "environment": {
            "python_version": platform.python_version(),
            "system": platform.system(),
            "architecture": platform.machine()
        }
    }


@router.get("/health/detailed")
async def detailed_health():
    """
    Detailed health check with system metrics.

    Includes CPU, memory, and disk usage statistics.

    Returns:
        Detailed system health metrics
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "system_metrics": {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "disk_usage_percent": disk.percent,
                "disk_free_gb": disk.free / (1024 * 1024 * 1024)
            },
            "components": {
                "ml_engine": "operational",
                "log_parser": "operational",
                "api": "operational"
            }
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }


@router.get("/ready")
async def readiness_check():
    """
    Kubernetes readiness probe endpoint.

    Indicates if the service is ready to accept traffic.

    Returns:
        Readiness status
    """
    return {"ready": True}