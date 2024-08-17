import pytest
from app.core.log_parser import LogParser

def test_parser():
    parser = LogParser()
    result = parser.parse("test log")
    assert len(result) > 0
