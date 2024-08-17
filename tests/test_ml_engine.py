import pytest
import asyncio
from datetime import datetime
from app.core.ml_engine import MLEngine


class TestMLEngine:
    def setup_method(self):
        self.ml_engine = MLEngine()

    @pytest.mark.asyncio
    async def test_initialization(self):
        await self.ml_engine.initialize()
        assert self.ml_engine.is_ready() is True

    @pytest.mark.asyncio
    async def test_detect_anomalies_empty_logs(self):
        await self.ml_engine.initialize()
        result = await self.ml_engine.detect_anomalies([])

        assert result['anomaly_count'] == 0
        assert result['anomaly_percentage'] == 0
        assert result['patterns'] == []

    @pytest.mark.asyncio
    async def test_detect_anomalies_normal_logs(self):
        await self.ml_engine.initialize()

        # Create normal logs
        logs = []
        for i in range(20):
            logs.append({
                'timestamp': datetime.now(),
                'level': 'INFO',
                'status': 200,
                'size': 1000 + i * 10,
                'message': f'Normal request {i}'
            })

        result = await self.ml_engine.detect_anomalies(logs)

        assert isinstance(result['anomaly_count'], int)
        assert isinstance(result['anomaly_percentage'], float)
        assert isinstance(result['patterns'], list)

    @pytest.mark.asyncio
    async def test_detect_anomalies_with_errors(self):
        await self.ml_engine.initialize()

        # Create logs with obvious anomalies
        logs = []

        # Normal logs
        for i in range(15):
            logs.append({
                'timestamp': datetime.now(),
                'level': 'INFO',
                'status': 200,
                'size': 1000,
                'message': 'Normal request'
            })

        # Anomalous logs
        for i in range(5):
            logs.append({
                'timestamp': datetime.now(),
                'level': 'ERROR',
                'status': 500,
                'size': 50000,  # Much larger than normal
                'message': 'Database connection failed with timeout error'
            })

        result = await self.ml_engine.detect_anomalies(logs)

        # Should detect some anomalies
        assert result['anomaly_count'] > 0
        assert len(result['suggestions']) > 0

    @pytest.mark.asyncio
    async def test_quick_anomaly_check(self):
        await self.ml_engine.initialize()

        # Test server error detection
        error_log = {
            'status': 500,
            'message': 'Internal server error'
        }
        assert self.ml_engine._quick_anomaly_check(error_log) is True

        # Test normal log
        normal_log = {
            'status': 200,
            'message': 'Request processed successfully'
        }
        assert self.ml_engine._quick_anomaly_check(normal_log) is False

        # Test error message detection
        error_message_log = {
            'status': 200,
            'message': 'Request failed with timeout error'
        }
        assert self.ml_engine._quick_anomaly_check(error_message_log) is True

    def test_calculate_severity(self):
        # Critical severity
        critical_log = {
            'status': 500,
            'level': 'CRITICAL',
            'message': 'Fatal database error'
        }
        severity = self.ml_engine._calculate_severity(critical_log)
        assert severity == 'CRITICAL'

        # High severity
        high_log = {
            'status': 500,
            'level': 'ERROR',
            'message': 'Database error'
        }
        severity = self.ml_engine._calculate_severity(high_log)
        assert severity == 'HIGH'

        # Low severity
        low_log = {
            'status': 200,
            'level': 'INFO',
            'message': 'Request processed'
        }
        severity = self.ml_engine._calculate_severity(low_log)
        assert severity == 'LOW'

    def test_classify_anomaly(self):
        # Server error
        server_error = {'status': 500}
        assert self.ml_engine._classify_anomaly(server_error) == 'server_error'

        # Client error
        client_error = {'status': 404}
        assert self.ml_engine._classify_anomaly(client_error) == 'client_error'

        # Timeout error
        timeout_error = {'message': 'Request timeout occurred'}
        assert self.ml_engine._classify_anomaly(timeout_error) == 'timeout'

        # Memory issue
        memory_error = {'message': 'Out of memory error'}
        assert self.ml_engine._classify_anomaly(memory_error) == 'memory_issue'

        # Large response
        large_response = {'size': 2000000}
        assert self.ml_engine._classify_anomaly(large_response) == 'large_response'

    @pytest.mark.asyncio
    async def test_process_stream(self):
        await self.ml_engine.initialize()

        stream_logs = [
            {
                'timestamp': datetime.now(),
                'status': 500,
                'message': 'Database connection failed'
            },
            {
                'timestamp': datetime.now(),
                'status': 200,
                'message': 'Request processed successfully'
            }
        ]

        anomalies = await self.ml_engine.process_stream(stream_logs)

        # Should detect at least one anomaly
        assert len(anomalies) >= 1
        assert anomalies[0]['severity'] in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']

    @pytest.mark.asyncio
    async def test_feature_extraction(self):
        await self.ml_engine.initialize()

        logs = [
            {
                'timestamp': datetime(2024, 1, 1, 12, 0, 0),
                'status': 200,
                'size': 1000,
                'message': 'Request processed',
                'level': 'INFO'
            }
        ]

        features, feature_names = self.ml_engine._extract_features(logs)

        assert features.shape[0] == 1  # One log
        assert features.shape[1] > 0   # Multiple features
        assert len(feature_names) == features.shape[1]

    @pytest.mark.asyncio
    async def test_cleanup(self):
        await self.ml_engine.initialize()
        await self.ml_engine.cleanup()
        # Should not raise any exceptions