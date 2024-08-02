from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
import uvicorn
from loguru import logger
import os
from datetime import datetime
from pathlib import Path

from app.core.config import settings
from app.api.v1 import logs, anomalies, health, metrics
from app.core.ml_engine import MLEngine
from app.core.log_parser import LogParser
from app.utils.prometheus import metrics_middleware, generate_metrics

# Initialize components
ml_engine = MLEngine()
log_parser = LogParser()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting LogSense AI...")
    await ml_engine.initialize()
    yield
    # Shutdown
    logger.info("Shutting down LogSense AI...")
    await ml_engine.cleanup()

# Create FastAPI app
app = FastAPI(
    title="LogSense AI",
    description="Intelligent Log Anomaly Detection System",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,  # Disable default docs
    redoc_url=None  # Disable default redoc
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus metrics middleware
app.middleware("http")(metrics_middleware)

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Custom favicon endpoint
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve custom favicon for all pages."""
    favicon_path = Path(__file__).parent / "static" / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(favicon_path, media_type="image/x-icon")
    raise HTTPException(status_code=404, detail="Favicon not found")

# Custom Swagger UI with custom favicon
@app.get("/api/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI with TJson favicon."""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - API Documentation",
        swagger_favicon_url="/static/favicon.ico"
    )

# Custom ReDoc with custom favicon
@app.get("/api/redoc", include_in_schema=False)
async def custom_redoc_html():
    """Custom ReDoc with TJson favicon."""
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - API Documentation",
        redoc_favicon_url="/static/favicon.ico"
    )

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    template_path = Path(__file__).parent / "templates" / "index.html"
    if template_path.exists():
        return FileResponse(template_path, media_type="text/html")
    else:
        # Fallback to simple HTML if template not found
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LogSense AI</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="icon" type="image/x-icon" href="/static/favicon.ico">
        </head>
        <body style="font-family: system-ui; padding: 2rem; text-align: center;">
            <h1>LogSense AI</h1>
            <p>Intelligent Log Anomaly Detection System</p>
            <p>
                <a href="/api/docs">API Documentation</a> |
                <a href="/health">Health</a> |
                <a href="/metrics">Metrics</a>
            </p>
        </body>
        </html>
        """)

# Health check endpoint
@app.get("/health")
async def health_check(format: Optional[str] = None):
    """
    Health check endpoint with optional HTML format.

    Args:
        format: Optional format parameter (html for HTML response)

    Returns:
        Health status as JSON or HTML
    """
    health_data = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "ml_engine": ml_engine.is_ready()
    }

    if format == "html":
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Health Check - LogSense AI</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="icon" type="image/x-icon" href="/static/favicon.ico">
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    max-width: 800px;
                    margin: 2rem auto;
                    padding: 2rem;
                    background: #f8fafc;
                }}
                .header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 2rem;
                }}
                .back-button {{
                    background: #0284c7;
                    color: white;
                    padding: 0.5rem 1rem;
                    border-radius: 0.5rem;
                    text-decoration: none;
                    font-weight: 500;
                }}
                .back-button:hover {{
                    background: #0369a1;
                }}
                .card {{
                    background: white;
                    border-radius: 1rem;
                    padding: 2rem;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .status {{
                    display: inline-block;
                    padding: 0.5rem 1rem;
                    background: #10b981;
                    color: white;
                    border-radius: 0.5rem;
                    font-weight: bold;
                }}
                table {{
                    width: 100%;
                    margin-top: 1rem;
                    border-collapse: collapse;
                }}
                td {{
                    padding: 0.75rem;
                    border-bottom: 1px solid #e5e7eb;
                }}
                td:first-child {{
                    font-weight: 600;
                    color: #374151;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>System Health</h1>
                <a href="/" class="back-button">Back to Home</a>
            </div>
            <div class="card">
                <span class="status">{health_data['status'].upper()}</span>
                <table>
                    <tr>
                        <td>Service</td>
                        <td>LogSense AI</td>
                    </tr>
                    <tr>
                        <td>Version</td>
                        <td>{health_data['version']}</td>
                    </tr>
                    <tr>
                        <td>ML Engine</td>
                        <td>{'Ready' if health_data['ml_engine'] else 'Not Ready'}</td>
                    </tr>
                    <tr>
                        <td>Timestamp</td>
                        <td>{health_data['timestamp']}</td>
                    </tr>
                </table>
            </div>
        </body>
        </html>
        """)

    return health_data

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    return generate_metrics()

# Upload and analyze logs
@app.post("/api/v1/analyze")
async def analyze_logs(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Upload a log file and get anomaly analysis results.

    Accepts log files in various formats (JSON, Apache, Nginx, Syslog, CSV).
    Returns detected anomalies, patterns, and actionable suggestions.

    Args:
        file: Uploaded log file
        background_tasks: Background task queue for model updates

    Returns:
        Analysis results with anomaly count, patterns, and suggestions
    """
    try:
        # Read file content
        content = await file.read()
        content_str = content.decode('utf-8')

        # Parse logs
        parsed_logs = log_parser.parse(content_str, file.filename)

        # Detect anomalies
        results = await ml_engine.detect_anomalies(parsed_logs)

        # Schedule background training if needed
        if len(parsed_logs) > 100:
            background_tasks.add_task(ml_engine.update_model, parsed_logs)

        return {
            "status": "success",
            "file": file.filename,
            "total_logs": len(parsed_logs),
            "anomalies_found": results["anomaly_count"],
            "anomaly_percentage": results["anomaly_percentage"],
            "patterns": results["patterns"],
            "suggestions": results["suggestions"]
        }

    except Exception as e:
        logger.error(f"Error analyzing logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Stream logs endpoint
@app.post("/api/v1/stream")
async def stream_logs(logs: List[Dict[str, Any]]):
    """
    Stream logs for real-time anomaly detection.

    Processes logs using rule-based quick checks for immediate results.

    Args:
        logs: List of log entries as JSON objects

    Returns:
        Detected anomalies from the stream
    """
    try:
        results = await ml_engine.process_stream(logs)
        return {
            "status": "success",
            "processed": len(logs),
            "anomalies": results
        }
    except Exception as e:
        logger.error(f"Error in stream processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Get anomaly patterns
@app.get("/api/v1/patterns")
async def get_patterns(limit: int = 10):
    """
    Get the most common anomaly patterns.

    Args:
        limit: Maximum number of patterns to return

    Returns:
        List of recurring anomaly patterns with occurrence counts
    """
    patterns = await ml_engine.get_top_patterns(limit)
    return {
        "patterns": patterns,
        "total": len(patterns)
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )