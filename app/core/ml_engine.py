import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter, defaultdict
import pickle
import os
import asyncio
from datetime import datetime, timedelta
from loguru import logger
import hashlib


class MLEngine:
    """
    Machine Learning engine for anomaly detection in logs.

    Uses Isolation Forest for outlier detection and DBSCAN for clustering.
    Extracts features from logs and identifies anomalous patterns.
    """

    def __init__(self):
        self.isolation_forest = None
        self.dbscan = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% variance
        self.is_trained = False
        self.feature_names = []
        self.anomaly_patterns = defaultdict(list)
        self.model_path = "/tmp/models"

    async def initialize(self):
        """
        Initialize ML models and load existing models if available.

        Creates model directory and attempts to load persisted models.
        Falls back to new model initialization if loading fails.
        """
        os.makedirs(self.model_path, exist_ok=True)

        # Try to load existing models
        if await self._load_models():
            logger.info("Loaded existing ML models")
        else:
            # Initialize with default parameters
            self.isolation_forest = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            self.dbscan = DBSCAN(
                eps=0.5,
                min_samples=5,
                metric='euclidean'
            )
            logger.info("Initialized new ML models")

    async def cleanup(self):
        """Cleanup resources."""
        await self._save_models()

    def is_ready(self) -> bool:
        """Check if ML engine is ready."""
        return self.isolation_forest is not None

    async def detect_anomalies(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect anomalies in parsed logs using ML algorithms.

        Args:
            logs: List of parsed log dictionaries

        Returns:
            Dictionary containing anomaly count, patterns, and suggestions
        """
        if not logs:
            return self._empty_result()

        # Extract features
        features, feature_names = self._extract_features(logs)

        if features.shape[0] < 10:
            return {
                "anomaly_count": 0,
                "anomaly_percentage": 0,
                "patterns": [],
                "suggestions": ["Need more data for accurate anomaly detection (minimum 10 logs)"],
                "details": []
            }

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Apply PCA if high dimensional
        if features_scaled.shape[1] > 10:
            features_scaled = self.pca.fit_transform(features_scaled)

        # Detect anomalies using Isolation Forest
        anomaly_scores = self._detect_with_isolation_forest(features_scaled)

        # Cluster analysis for pattern detection
        clusters = self._detect_with_clustering(features_scaled)

        # Combine results
        anomalies = []
        for i, (score, cluster) in enumerate(zip(anomaly_scores, clusters)):
            if score == -1:  # Anomaly detected
                anomaly_info = self._analyze_anomaly(logs[i], features[i], feature_names)
                anomaly_info['cluster'] = int(cluster) if cluster != -1 else 'outlier'
                anomalies.append(anomaly_info)

        # Extract patterns
        patterns = self._extract_patterns(anomalies)

        # Generate suggestions
        suggestions = self._generate_suggestions(anomalies, patterns)

        return {
            "anomaly_count": len(anomalies),
            "anomaly_percentage": round(len(anomalies) / len(logs) * 100, 2),
            "patterns": patterns[:10],  # Top 10 patterns
            "suggestions": suggestions,
            "details": anomalies[:50]  # Return max 50 anomalies
        }

    async def process_stream(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process streaming logs for real-time anomaly detection.

        Args:
            logs: List of log dictionaries from stream

        Returns:
            List of detected anomalies with severity and classification
        """
        anomalies = []

        for log in logs:
            # Quick anomaly check using rules and patterns
            if self._quick_anomaly_check(log):
                anomalies.append({
                    "timestamp": log.get('timestamp', datetime.now()),
                    "severity": self._calculate_severity(log),
                    "type": self._classify_anomaly(log),
                    "log": log
                })

        return anomalies

    async def update_model(self, new_logs: List[Dict[str, Any]]):
        """Update model with new training data."""
        if len(new_logs) < 50:
            return

        try:
            features, _ = self._extract_features(new_logs)
            features_scaled = self.scaler.fit_transform(features)

            # Retrain models
            self.isolation_forest.fit(features_scaled)
            self.is_trained = True

            # Save updated models
            await self._save_models()
            logger.info(f"Model updated with {len(new_logs)} new samples")
        except Exception as e:
            logger.error(f"Failed to update model: {e}")

    async def get_top_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most common anomaly patterns."""
        pattern_counts = Counter()

        for pattern_type, patterns in self.anomaly_patterns.items():
            for pattern in patterns:
                pattern_counts[f"{pattern_type}: {pattern}"] += 1

        top_patterns = []
        for pattern, count in pattern_counts.most_common(limit):
            top_patterns.append({
                "pattern": pattern,
                "occurrences": count,
                "severity": self._pattern_severity(pattern)
            })

        return top_patterns

    def _extract_features(self, logs: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract numerical features from logs for ML processing.

        Features include time-based, status codes, message patterns, and sizes.

        Args:
            logs: List of parsed log dictionaries

        Returns:
            Tuple of (feature array, feature names list)
        """
        features = []
        feature_names = []

        for log in logs:
            log_features = []

            # Time-based features
            if log.get('timestamp'):
                dt = log['timestamp'] if isinstance(log['timestamp'], datetime) else datetime.now()
                log_features.extend([
                    dt.hour,
                    dt.minute,
                    dt.weekday(),
                    1 if dt.weekday() >= 5 else 0  # Weekend flag
                ])
                if not feature_names:
                    feature_names.extend(['hour', 'minute', 'weekday', 'is_weekend'])

            # Status code features
            if log.get('status'):
                log_features.extend([
                    log['status'],
                    1 if log['status'] >= 400 else 0,
                    1 if log['status'] >= 500 else 0
                ])
                if 'status' not in feature_names:
                    feature_names.extend(['status_code', 'is_4xx', 'is_5xx'])

            # Size features
            if log.get('size'):
                log_features.extend([
                    np.log1p(log['size']),  # Log transform for better distribution
                    1 if log['size'] > 10000 else 0
                ])
                if 'log_size' not in feature_names:
                    feature_names.extend(['log_size', 'is_large'])

            # Message features
            if log.get('message'):
                msg = str(log['message']).lower()
                log_features.extend([
                    len(msg),
                    msg.count('error'),
                    msg.count('warning'),
                    msg.count('exception'),
                    1 if 'timeout' in msg else 0,
                    1 if 'failed' in msg else 0
                ])
                if 'msg_length' not in feature_names:
                    feature_names.extend([
                        'msg_length', 'error_count', 'warning_count',
                        'exception_count', 'has_timeout', 'has_failed'
                    ])

            # Level encoding
            level_map = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3, 'CRITICAL': 4}
            level = log.get('level', 'INFO').upper()
            log_features.append(level_map.get(level, 1))
            if 'level_encoded' not in feature_names:
                feature_names.append('level_encoded')

            features.append(log_features)

        # Convert to numpy array, padding with zeros if needed
        max_features = max(len(f) for f in features) if features else 0
        padded_features = []
        for f in features:
            if len(f) < max_features:
                f.extend([0] * (max_features - len(f)))
            padded_features.append(f)

        return np.array(padded_features), feature_names[:max_features]

    def _detect_with_isolation_forest(self, features: np.ndarray) -> np.ndarray:
        """Detect anomalies using Isolation Forest."""
        if not self.is_trained:
            self.isolation_forest.fit(features)
            self.is_trained = True

        return self.isolation_forest.predict(features)

    def _detect_with_clustering(self, features: np.ndarray) -> np.ndarray:
        """Detect anomalies using clustering."""
        try:
            clusters = self.dbscan.fit_predict(features)
            return clusters
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            return np.zeros(features.shape[0])

    def _analyze_anomaly(self, log: Dict[str, Any], features: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Analyze individual anomaly."""
        anomaly_info = {
            "timestamp": log.get('timestamp', datetime.now()),
            "log_line": log.get('line_number', -1),
            "severity": self._calculate_severity(log),
            "type": self._classify_anomaly(log),
            "reason": []
        }

        # Find which features are unusual
        if len(features) > 0:
            mean_features = np.mean(features)
            std_features = np.std(features) + 1e-6  # Avoid division by zero
            z_scores = np.abs((features - mean_features) / std_features)

            # Features with high z-scores are unusual
            unusual_indices = np.where(z_scores > 2)[0]
            for idx in unusual_indices[:5]:  # Top 5 unusual features
                if idx < len(feature_names):
                    anomaly_info["reason"].append({
                        "feature": feature_names[idx],
                        "value": float(features[idx]),
                        "z_score": float(z_scores[idx])
                    })

        # Add contextual information
        if log.get('status') and log['status'] >= 500:
            anomaly_info["reason"].append({"feature": "status", "issue": "Server error"})

        if log.get('message') and 'error' in str(log['message']).lower():
            anomaly_info["reason"].append({"feature": "message", "issue": "Contains error keyword"})

        return anomaly_info

    def _quick_anomaly_check(self, log: Dict[str, Any]) -> bool:
        """Quick rule-based anomaly check for streaming."""
        # Check for error status codes
        if log.get('status') and log['status'] >= 500:
            return True

        # Check for error keywords
        if log.get('message'):
            msg = str(log['message']).lower()
            error_keywords = ['error', 'exception', 'failed', 'timeout', 'critical']
            if any(keyword in msg for keyword in error_keywords):
                return True

        # Check for unusual patterns
        if log.get('size') and log['size'] > 1000000:  # > 1MB response
            return True

        return False

    def _calculate_severity(self, log: Dict[str, Any]) -> str:
        """Calculate anomaly severity."""
        severity_score = 0

        if log.get('status'):
            if log['status'] >= 500:
                severity_score += 3
            elif log['status'] >= 400:
                severity_score += 2

        if log.get('level') in ['ERROR', 'CRITICAL']:
            severity_score += 2

        if log.get('message'):
            msg = str(log['message']).lower()
            if 'critical' in msg or 'fatal' in msg:
                severity_score += 3
            elif 'error' in msg or 'exception' in msg:
                severity_score += 2
            elif 'warning' in msg:
                severity_score += 1

        if severity_score >= 4:
            return "CRITICAL"
        elif severity_score >= 2:
            return "HIGH"
        elif severity_score >= 1:
            return "MEDIUM"
        else:
            return "LOW"

    def _classify_anomaly(self, log: Dict[str, Any]) -> str:
        """Classify the type of anomaly."""
        if log.get('status'):
            if log['status'] >= 500:
                return "server_error"
            elif log['status'] >= 400:
                return "client_error"
            elif log['status'] == 0:
                return "connection_failure"

        if log.get('message'):
            msg = str(log['message']).lower()
            if 'timeout' in msg:
                return "timeout"
            elif 'memory' in msg:
                return "memory_issue"
            elif 'permission' in msg or 'denied' in msg:
                return "permission_error"
            elif 'database' in msg or 'sql' in msg:
                return "database_error"

        if log.get('size') and log['size'] > 1000000:
            return "large_response"

        return "unknown"

    def _extract_patterns(self, anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract common patterns from anomalies."""
        patterns = []

        # Group by anomaly type
        type_groups = defaultdict(list)
        for anomaly in anomalies:
            type_groups[anomaly['type']].append(anomaly)

        for anomaly_type, group in type_groups.items():
            pattern = {
                "type": anomaly_type,
                "count": len(group),
                "severity": max(a['severity'] for a in group),
                "timespan": self._calculate_timespan(group),
                "common_reasons": self._get_common_reasons(group)
            }
            patterns.append(pattern)

        # Sort by count
        patterns.sort(key=lambda x: x['count'], reverse=True)

        return patterns

    def _generate_suggestions(self, anomalies: List[Dict[str, Any]], patterns: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable suggestions based on anomalies."""
        suggestions = []

        if not anomalies:
            return ["No anomalies detected. System appears to be running normally."]

        # Check for high error rate
        error_count = sum(1 for a in anomalies if a['type'] in ['server_error', 'client_error'])
        if error_count > len(anomalies) * 0.5:
            suggestions.append(f"High error rate detected ({error_count} errors). Review server configuration and error logs.")

        # Check for performance issues
        timeout_count = sum(1 for a in anomalies if a['type'] == 'timeout')
        if timeout_count > 3:
            suggestions.append("Multiple timeouts detected. Consider optimizing slow queries or increasing timeout limits.")

        # Check for security issues
        permission_errors = sum(1 for a in anomalies if a['type'] == 'permission_error')
        if permission_errors > 0:
            suggestions.append("Permission errors detected. Review access controls and authentication mechanisms.")

        # Check for resource issues
        memory_issues = sum(1 for a in anomalies if a['type'] == 'memory_issue')
        if memory_issues > 0:
            suggestions.append("Memory-related issues detected. Monitor resource usage and consider scaling.")

        # Pattern-based suggestions
        for pattern in patterns[:3]:  # Top 3 patterns
            if pattern['severity'] == 'CRITICAL':
                suggestions.append(f"Critical pattern detected: {pattern['type']} ({pattern['count']} occurrences). Immediate action required.")

        return suggestions[:5]  # Return top 5 suggestions

    def _calculate_timespan(self, anomalies: List[Dict[str, Any]]) -> str:
        """Calculate timespan of anomalies."""
        if not anomalies:
            return "N/A"

        timestamps = [a['timestamp'] for a in anomalies if a.get('timestamp')]
        if not timestamps:
            return "N/A"

        min_time = min(timestamps)
        max_time = max(timestamps)
        duration = max_time - min_time if isinstance(max_time, datetime) else timedelta(0)

        if duration.total_seconds() < 60:
            return f"{int(duration.total_seconds())} seconds"
        elif duration.total_seconds() < 3600:
            return f"{int(duration.total_seconds() / 60)} minutes"
        else:
            return f"{int(duration.total_seconds() / 3600)} hours"

    def _get_common_reasons(self, anomalies: List[Dict[str, Any]]) -> List[str]:
        """Get common reasons for anomalies."""
        reason_counts = Counter()

        for anomaly in anomalies:
            for reason in anomaly.get('reason', []):
                if isinstance(reason, dict):
                    reason_counts[reason.get('feature', 'unknown')] += 1

        return [reason for reason, _ in reason_counts.most_common(3)]

    def _pattern_severity(self, pattern: str) -> str:
        """Determine pattern severity."""
        critical_keywords = ['critical', 'fatal', 'emergency']
        high_keywords = ['error', 'exception', 'failed']
        medium_keywords = ['warning', 'timeout']

        pattern_lower = pattern.lower()

        if any(keyword in pattern_lower for keyword in critical_keywords):
            return "CRITICAL"
        elif any(keyword in pattern_lower for keyword in high_keywords):
            return "HIGH"
        elif any(keyword in pattern_lower for keyword in medium_keywords):
            return "MEDIUM"
        else:
            return "LOW"

    async def _save_models(self) -> bool:
        """Save trained models to disk."""
        try:
            if self.is_trained:
                model_data = {
                    'isolation_forest': self.isolation_forest,
                    'scaler': self.scaler,
                    'pca': self.pca,
                    'feature_names': self.feature_names,
                    'anomaly_patterns': dict(self.anomaly_patterns)
                }

                model_file = os.path.join(self.model_path, 'model.pkl')
                with open(model_file, 'wb') as f:
                    pickle.dump(model_data, f)

                logger.info(f"Models saved to {model_file}")
                return True
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            return False

    async def _load_models(self) -> bool:
        """Load trained models from disk."""
        try:
            model_file = os.path.join(self.model_path, 'model.pkl')
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)

                self.isolation_forest = model_data['isolation_forest']
                self.scaler = model_data['scaler']
                self.pca = model_data['pca']
                self.feature_names = model_data['feature_names']
                self.anomaly_patterns = defaultdict(list, model_data['anomaly_patterns'])
                self.is_trained = True

                return True
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
            return False

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            "anomaly_count": 0,
            "anomaly_percentage": 0,
            "patterns": [],
            "suggestions": ["No logs to analyze"],
            "details": []
        }