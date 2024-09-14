import pytest
from fastapi.testclient import TestClient
from io import BytesIO
from app.main import app

client = TestClient(app)


class TestAPI:
    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "LogSense AI" in response.text

    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_metrics_endpoint(self):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_api_docs(self):
        response = client.get("/api/docs")
        assert response.status_code == 200

    def test_analyze_logs_endpoint(self):
        # Create a sample log file
        log_content = '''{"timestamp": "2024-01-01T12:00:00", "level": "INFO", "message": "User login"}
{"timestamp": "2024-01-01T12:01:00", "level": "ERROR", "message": "Database error"}
{"timestamp": "2024-01-01T12:02:00", "level": "INFO", "message": "User logout"}'''

        files = {"file": ("test.log", BytesIO(log_content.encode()), "text/plain")}
        response = client.post("/api/v1/analyze", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "total_logs" in data
        assert "anomalies_found" in data
        assert "patterns" in data
        assert "suggestions" in data

    def test_stream_logs_endpoint(self):
        stream_data = [
            {
                "timestamp": "2024-01-01T12:00:00",
                "level": "ERROR",
                "message": "Database connection failed",
                "metadata": {"server": "api-01"}
            }
        ]

        response = client.post("/api/v1/stream", json=stream_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "processed" in data
        assert "anomalies" in data

    def test_patterns_endpoint(self):
        response = client.get("/api/v1/patterns?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert "patterns" in data
        assert "total" in data

    def test_log_validation_endpoint(self):
        sample_log = '{"timestamp": "2024-01-01T12:00:00", "level": "ERROR", "message": "Test error"}'
        response = client.post("/api/v1/logs/validate", json=sample_log)
        assert response.status_code == 200
        data = response.json()
        assert "format" in data
        assert "is_valid" in data

    def test_anomaly_summary_endpoint(self):
        response = client.get("/api/v1/anomalies/summary")
        assert response.status_code == 200
        data = response.json()
        assert "total_anomalies_24h" in data
        assert "critical" in data
        assert "high" in data

    def test_anomaly_trends_endpoint(self):
        response = client.get("/api/v1/anomalies/trends?period=24h")
        assert response.status_code == 200
        data = response.json()
        assert "period" in data
        assert "data_points" in data
        assert "trend" in data

    def test_detailed_health_endpoint(self):
        response = client.get("/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert "system_metrics" in data
        assert "components" in data

    def test_readiness_endpoint(self):
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True

    def test_metrics_summary_endpoint(self):
        response = client.get("/api/v1/metrics/summary")
        assert response.status_code == 200
        data = response.json()
        assert "requests" in data
        assert "processing" in data
        assert "performance" in data

    def test_invalid_file_upload(self):
        # Test with empty file
        files = {"file": ("empty.log", BytesIO(b""), "text/plain")}
        response = client.post("/api/v1/analyze", files=files)
        # Should handle gracefully
        assert response.status_code in [200, 400]

    def test_cors_headers(self):
        response = client.options("/api/v1/patterns")
        # CORS should be configured
        assert response.status_code == 200