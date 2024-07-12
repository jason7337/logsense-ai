# LogSense AI

Log analysis system using machine learning for anomaly detection.

## Overview

LogSense AI analyzes application logs using Isolation Forest and DBSCAN clustering to detect anomalies and identify patterns. Supports multiple log formats with automatic detection.

## Features

- Multi-format log parsing (JSON, Apache, Nginx, Syslog, CSV)
- Anomaly detection with machine learning
- Pattern clustering and analysis
- RESTful API with documentation
- Prometheus metrics
- Docker support

## Requirements

- Python 3.12+
- Docker

## Installation

### Docker

```bash
docker build -t logsense-ai .
docker run -p 8080:8080 logsense-ai
```

### Local

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Usage

### Analyze logs

```bash
curl -X POST "http://localhost:8080/api/v1/analyze" \
  -F "file=@logfile.log"
```

### Stream logs

```bash
curl -X POST "http://localhost:8080/api/v1/stream" \
  -H "Content-Type: application/json" \
  -d '[{"timestamp": "2024-07-15T12:00:00", "level": "ERROR", "message": "Connection failed"}]'
```

### Get patterns

```bash
curl "http://localhost:8080/api/v1/patterns?limit=10"
```

## API Documentation

Available at `/api/docs` when running.

## Configuration

Environment variables:

- `PORT`: Server port (default: 8080)
- `ML_CONFIDENCE_THRESHOLD`: Anomaly detection threshold (default: 0.85)
- `ML_ANOMALY_RATIO`: Expected anomaly ratio (default: 0.1)
- `REDIS_URL`: Redis URL for caching (optional)
- `MONGODB_URL`: MongoDB URL for storage (optional)

## Stack

- FastAPI
- scikit-learn
- spaCy
- Prometheus
- Docker

## Structure

```
logsense-ai/
├── app/
│   ├── api/v1/          # API endpoints
│   ├── core/            # Parser and ML engine
│   ├── utils/           # Utilities
│   └── main.py          # Entry point
├── tests/
├── Dockerfile
└── requirements.txt
```

## Deployment

Google Cloud Run:

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/logsense-ai
gcloud run deploy logsense-ai --image gcr.io/PROJECT_ID/logsense-ai --platform managed
```

## License

MIT

## Author

Jasson Gomez - [tjson.net](https://tjson.net)
