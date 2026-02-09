"""
Unit tests for monitoring system
"""
import pytest
import json
from pathlib import Path
from monitoring import MonitoringSystem, RequestLog, QueryTracker
"""
Unit tests for monitoring system
"""
import pytest
import json
from pathlib import Path
from monitoring import MonitoringSystem, RequestLog, QueryTracker
import monitoring  # Import module for global reset


@pytest.fixture(autouse=True)
def reset_global_metrics():
    """Reset global metrics before each test"""
    # Reset global metrics dictionary
    monitoring._metrics["total_requests"] = 0
    monitoring._metrics["successful_requests"] = 0
    monitoring._metrics["failed_requests"] = 0
    monitoring._metrics["total_response_time"] = 0.0
    monitoring._metrics["average_response_time"] = 0.0
    monitoring._metrics["requests_by_hour"].clear()
    monitoring._metrics["error_types"].clear()
    yield
    # Cleanup after test (optional)

class TestRequestLog:
    """Test RequestLog dataclass"""
    
    def test_request_log_creation(self):
        """Test creating a request log"""
        log = RequestLog(
            timestamp="2026-02-08T12:00:00",
            query="test query",
            response_time=1.5,
            success=True,
            response_length=100
        )
        
        assert log.timestamp == "2026-02-08T12:00:00"
        assert log.query == "test query"
        assert log.response_time == 1.5
        assert log.success is True
        assert log.response_length == 100
        assert log.error_message is None
    
    def test_request_log_to_dict(self):
        """Test converting request log to dictionary"""
        log = RequestLog(
            timestamp="2026-02-08T12:00:00",
            query="test",
            response_time=1.0,
            success=True
        )
        
        result = log.to_dict()
        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["response_time"] == 1.0


class TestMonitoringSystem:
    """Test MonitoringSystem class"""
    
    @pytest.fixture
    def monitoring_system(self, tmp_path):
        """Create a monitoring system with temporary log directory"""
        return MonitoringSystem(log_dir=str(tmp_path))
    
    def test_monitoring_initialization(self, monitoring_system, tmp_path):
        """Test monitoring system initializes correctly"""
        assert monitoring_system.log_dir == tmp_path
        assert monitoring_system.log_dir.exists()
    
    def test_log_request(self, monitoring_system):
        """Test logging a request"""
        log = RequestLog(
            timestamp="2026-02-08T12:00:00",
            query="test query",
            response_time=1.5,
            success=True,
            response_length=100
        )
        
        monitoring_system.log_request(log)
        
        # Check log file was created
        log_files = list(monitoring_system.log_dir.glob("requests_*.jsonl"))
        assert len(log_files) == 1
        
        # Check log content
        with open(log_files[0], 'r') as f:
            content = f.read().strip()
            data = json.loads(content)
            assert data["query"] == "test query"
            assert data["success"] is True
    
    def test_update_metrics(self, monitoring_system):
        """Test metrics updating"""
        # Update metrics with successful request
        monitoring_system.update_metrics(1.5, True)
        
        metrics = monitoring_system.get_metrics()
        assert metrics["total_requests"] == 1
        assert metrics["successful_requests"] == 1
        assert metrics["failed_requests"] == 0
        assert metrics["success_rate"] == 100.0
        assert metrics["average_response_time"] == 1.5
    
    def test_update_metrics_with_failure(self, monitoring_system):
        """Test metrics with failed request"""
        monitoring_system.update_metrics(2.0, False, "TestError")
        
        metrics = monitoring_system.get_metrics()
        assert metrics["total_requests"] == 1
        assert metrics["successful_requests"] == 0
        assert metrics["failed_requests"] == 1
        assert metrics["success_rate"] == 0.0
        assert "TestError" in metrics["error_types"]
    
    def test_get_metrics_structure(self, monitoring_system):
        """Test metrics response structure"""
        metrics = monitoring_system.get_metrics()
        
        # Check all required fields exist
        required_fields = [
            "total_requests",
            "successful_requests",
            "failed_requests",
            "total_response_time",
            "average_response_time",
            "uptime_seconds",
            "uptime_hours",
            "success_rate",
            "error_types",
            "requests_by_hour"
        ]
        
        for field in required_fields:
            assert field in metrics


class TestQueryTracker:
    """Test QueryTracker context manager"""
    
    @pytest.fixture
    def monitoring_system(self, tmp_path):
        """Create a monitoring system with temporary log directory"""
        return MonitoringSystem(log_dir=str(tmp_path))
    
    def test_query_tracker_success(self, monitoring_system):
        """Test tracking a successful query"""
        with monitoring_system.track_query("test query") as tracker:
            tracker.set_response("test response")
        
        metrics = monitoring_system.get_metrics()
        assert metrics["total_requests"] == 1
        assert metrics["successful_requests"] == 1
    
    def test_query_tracker_failure(self, monitoring_system):
        """Test tracking a failed query"""
        try:
            with monitoring_system.track_query("test query"):
                raise ValueError("Test error")
        except ValueError:
            pass
        
        metrics = monitoring_system.get_metrics()
        assert metrics["total_requests"] == 1
        assert metrics["failed_requests"] == 1
        assert "ValueError" in metrics["error_types"]
    
    def test_query_tracker_response_length(self, monitoring_system):
        """Test response length tracking"""
        with monitoring_system.track_query("test") as tracker:
            response = "a" * 250
            tracker.set_response(response)
            assert tracker.response_length == 250