from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List, Dict, Any
from app.core.log_parser import LogParser

router = APIRouter(prefix="/api/v1/logs", tags=["logs"])
parser = LogParser()


@router.post("/parse")
async def parse_logs(file: UploadFile = File(...)):
    """Parse log file and return structured data."""
    try:
        content = await file.read()
        content_str = content.decode('utf-8')

        parsed_logs = parser.parse(content_str, file.filename)

        return {
            "filename": file.filename,
            "total_lines": len(parsed_logs),
            "formats_detected": list(set(log.get('format', 'unknown') for log in parsed_logs)),
            "logs": parsed_logs[:100]  # Return first 100 logs
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/validate")
async def validate_format(sample: str):
    """Validate log format from sample."""
    try:
        format_type = parser.detect_format(sample)
        parsed = parser.parse_line(sample)

        return {
            "format": format_type.value if hasattr(format_type, 'value') else str(format_type),
            "is_valid": parsed.get('parsed', False),
            "parsed_data": parsed
        }
    except Exception as e:
        return {
            "format": "unknown",
            "is_valid": False,
            "error": str(e)
        }