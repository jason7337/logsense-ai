import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum
import parse
from dateutil import parser as date_parser
from loguru import logger


class LogFormat(Enum):
    JSON = "json"
    APACHE_COMBINED = "apache_combined"
    APACHE_COMMON = "apache_common"
    NGINX = "nginx"
    SYSLOG = "syslog"
    CSV = "csv"
    CUSTOM = "custom"


class LogParser:
    """
    Multi-format log parser with intelligent format detection.

    Supports multiple log formats including JSON, Apache, Nginx, Syslog, and CSV.
    Automatically detects the format and extracts relevant fields from log entries.
    """

    def __init__(self):
        self.patterns = {
            LogFormat.APACHE_COMBINED: re.compile(
                r'(?P<ip>\S+) \S+ (?P<user>\S+) \[(?P<time>[^\]]+)\] "(?P<method>\S+) (?P<path>[^"]+) (?P<protocol>[^"]+)" (?P<status>\d+) (?P<size>\S+) "(?P<referrer>[^"]*)" "(?P<agent>[^"]*)"'
            ),
            LogFormat.APACHE_COMMON: re.compile(
                r'(?P<ip>\S+) \S+ (?P<user>\S+) \[(?P<time>[^\]]+)\] "(?P<method>\S+) (?P<path>[^"]+) (?P<protocol>[^"]+)" (?P<status>\d+) (?P<size>\S+)'
            ),
            LogFormat.NGINX: re.compile(
                r'(?P<ip>\S+) - (?P<user>\S+) \[(?P<time>[^\]]+)\] "(?P<method>\S+) (?P<path>[^"]+) (?P<protocol>[^"]+)" (?P<status>\d+) (?P<size>\d+) "(?P<referrer>[^"]*)" "(?P<agent>[^"]*)"'
            ),
            LogFormat.SYSLOG: re.compile(
                r'(?P<timestamp>\w+ \d+ \d+:\d+:\d+) (?P<hostname>\S+) (?P<process>[^\[]+)(?:\[(?P<pid>\d+)\])?: (?P<message>.*)'
            )
        }

    def detect_format(self, log_line: str) -> LogFormat:
        """
        Automatically detect log format from a single log line.

        Args:
            log_line: A single line from a log file

        Returns:
            LogFormat enum indicating the detected format
        """
        # Try JSON first
        if log_line.strip().startswith('{'):
            try:
                json.loads(log_line)
                return LogFormat.JSON
            except:
                pass

        # Try regex patterns
        for format_type, pattern in self.patterns.items():
            if pattern.match(log_line):
                return format_type

        # CSV detection
        if ',' in log_line and len(log_line.split(',')) > 3:
            return LogFormat.CSV

        return LogFormat.CUSTOM

    def parse_line(self, line: str, format_type: Optional[LogFormat] = None) -> Dict[str, Any]:
        """
        Parse a single log line into structured data.

        Args:
            line: Raw log line string
            format_type: Optional format type override, will auto-detect if not provided

        Returns:
            Dictionary containing parsed log fields
        """
        if not line.strip():
            return None

        if format_type is None:
            format_type = self.detect_format(line)

        try:
            if format_type == LogFormat.JSON:
                return self._parse_json(line)
            elif format_type in self.patterns:
                return self._parse_regex(line, format_type)
            elif format_type == LogFormat.CSV:
                return self._parse_csv(line)
            else:
                return self._parse_custom(line)
        except Exception as e:
            logger.warning(f"Failed to parse line: {e}")
            return {
                "raw": line,
                "format": "unknown",
                "parsed": False,
                "error": str(e)
            }

    def parse(self, content: str, filename: str = "") -> List[Dict[str, Any]]:
        """
        Parse multiple log lines from file content.

        Args:
            content: Full log file content as string
            filename: Name of the source file for reference

        Returns:
            List of dictionaries containing parsed log entries
        """
        lines = content.strip().split('\n')
        parsed_logs = []

        # Detect format from first valid line
        format_type = None
        for line in lines[:10]:  # Check first 10 lines
            if line.strip():
                format_type = self.detect_format(line)
                break

        for i, line in enumerate(lines):
            if not line.strip():
                continue

            parsed = self.parse_line(line, format_type)
            if parsed:
                parsed['line_number'] = i + 1
                parsed['source_file'] = filename
                parsed_logs.append(parsed)

        return self._enrich_logs(parsed_logs)

    def _parse_json(self, line: str) -> Dict[str, Any]:
        """Parse JSON formatted log."""
        data = json.loads(line)
        return {
            "format": "json",
            "timestamp": self._extract_timestamp(data),
            "level": data.get("level", data.get("severity", "INFO")),
            "message": data.get("message", data.get("msg", "")),
            "data": data,
            "parsed": True
        }

    def _parse_regex(self, line: str, format_type: LogFormat) -> Dict[str, Any]:
        """Parse log using regex pattern."""
        pattern = self.patterns[format_type]
        match = pattern.match(line)

        if not match:
            return None

        data = match.groupdict()

        # Parse timestamp
        timestamp = None
        if 'time' in data:
            timestamp = self._parse_apache_timestamp(data['time'])
        elif 'timestamp' in data:
            timestamp = self._parse_timestamp(data['timestamp'])

        # Extract level from status code or message
        level = self._infer_level(
            status=data.get('status'),
            message=data.get('message', '')
        )

        return {
            "format": format_type.value,
            "timestamp": timestamp,
            "level": level,
            "ip": data.get('ip'),
            "method": data.get('method'),
            "path": data.get('path'),
            "status": int(data.get('status')) if data.get('status') else None,
            "size": int(data.get('size')) if data.get('size', '').isdigit() else None,
            "user_agent": data.get('agent'),
            "data": data,
            "parsed": True
        }

    def _parse_csv(self, line: str) -> Dict[str, Any]:
        """Parse CSV formatted log."""
        parts = line.split(',')
        return {
            "format": "csv",
            "timestamp": self._parse_timestamp(parts[0]) if parts else None,
            "fields": parts,
            "parsed": True
        }

    def _parse_custom(self, line: str) -> Dict[str, Any]:
        """Parse custom format using heuristics."""
        # Extract timestamp
        timestamp_match = re.search(
            r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}',
            line
        )
        timestamp = None
        if timestamp_match:
            timestamp = self._parse_timestamp(timestamp_match.group())

        # Extract log level
        level_match = re.search(
            r'\b(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\b',
            line,
            re.IGNORECASE
        )
        level = level_match.group().upper() if level_match else "INFO"

        # Extract IP addresses
        ip_match = re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', line)
        ip = ip_match.group() if ip_match else None

        return {
            "format": "custom",
            "timestamp": timestamp,
            "level": level,
            "ip": ip,
            "message": line,
            "parsed": True
        }

    def _extract_timestamp(self, data: Dict) -> Optional[datetime]:
        """Extract timestamp from various field names."""
        timestamp_fields = ['timestamp', 'time', '@timestamp', 'datetime', 'date']
        for field in timestamp_fields:
            if field in data:
                return self._parse_timestamp(data[field])
        return None

    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse timestamp string to datetime."""
        try:
            return date_parser.parse(timestamp_str)
        except:
            return None

    def _parse_apache_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse Apache log timestamp format."""
        try:
            return datetime.strptime(timestamp_str, '%d/%b/%Y:%H:%M:%S %z')
        except:
            try:
                return datetime.strptime(timestamp_str, '%d/%b/%Y:%H:%M:%S')
            except:
                return None

    def _infer_level(self, status: Optional[str] = None, message: str = "") -> str:
        """Infer log level from status code or message."""
        if status:
            status_int = int(status)
            if status_int >= 500:
                return "ERROR"
            elif status_int >= 400:
                return "WARNING"
            elif status_int >= 300:
                return "INFO"
            else:
                return "INFO"

        # Check message for level indicators
        message_lower = message.lower()
        if any(word in message_lower for word in ['error', 'fail', 'exception']):
            return "ERROR"
        elif any(word in message_lower for word in ['warning', 'warn']):
            return "WARNING"
        elif any(word in message_lower for word in ['debug', 'trace']):
            return "DEBUG"
        else:
            return "INFO"

    def _enrich_logs(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add additional metadata and features to parsed logs."""
        for log in logs:
            # Add time-based features
            if log.get('timestamp'):
                dt = log['timestamp']
                log['hour'] = dt.hour
                log['day_of_week'] = dt.weekday()
                log['is_weekend'] = dt.weekday() >= 5

            # Add text features
            if log.get('message'):
                log['message_length'] = len(log['message'])
                log['has_error_keywords'] = any(
                    word in log['message'].lower()
                    for word in ['error', 'exception', 'fail', 'timeout']
                )

            # Add numeric features
            if log.get('status'):
                log['is_error'] = log['status'] >= 400
                log['is_server_error'] = log['status'] >= 500

            if log.get('size'):
                log['size_category'] = self._categorize_size(log['size'])

        return logs

    def _categorize_size(self, size: int) -> str:
        """Categorize response size."""
        if size < 1024:
            return "small"
        elif size < 10240:
            return "medium"
        elif size < 102400:
            return "large"
        else:
            return "very_large"