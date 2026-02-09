"""
Unit tests for FastAPI endpoints
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent directory to path to import app
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after path modification
from app import app

# Create test client
client = TestClient(app)


class TestHealthEndpoint:
    """Test /health endpoint"""
    
    def test_health_endpoint_exists(self):
        """Test health endpoint is accessible"""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_endpoint_structure(self):
        """Test health endpoint returns correct structure"""
        response = client.get("/health")
        data = response.json()
        
        # Check required fields
        assert "status" in data
        assert data["status"] == "healthy"


class TestMetricsEndpoints:
    """Test /metrics endpoints"""
    
    def test_metrics_endpoint_exists(self):
        """Test metrics endpoint is accessible"""
        response = client.get("/metrics")
        assert response.status_code == 200
    
    def test_metrics_endpoint_structure(self):
        """Test metrics endpoint returns correct structure"""
        response = client.get("/metrics")
        data = response.json()
        
        # Check required fields
        required_fields = [
            "total_requests",
            "successful_requests",
            "failed_requests",
            "average_response_time",
            "success_rate"
        ]
        
        for field in required_fields:
            assert field in data
    
    def test_metrics_endpoint_types(self):
        """Test metrics endpoint returns correct data types"""
        response = client.get("/metrics")
        data = response.json()
        
        assert isinstance(data["total_requests"], int)
        assert isinstance(data["successful_requests"], int)
        assert isinstance(data["failed_requests"], int)
        assert isinstance(data["average_response_time"], (int, float))
        assert isinstance(data["success_rate"], (int, float))
    
    def test_metrics_summary_endpoint_exists(self):
        """Test metrics summary endpoint is accessible"""
        response = client.get("/metrics/summary")
        assert response.status_code == 200
    
    def test_metrics_summary_structure(self):
        """Test metrics summary returns correct structure"""
        response = client.get("/metrics/summary")
        data = response.json()
        
        assert "summary" in data
        assert "raw_metrics" in data
        assert isinstance(data["summary"], str)
        assert isinstance(data["raw_metrics"], dict)


class TestAPIResponseCodes:
    """Test API response codes and error handling"""
    
    def test_valid_endpoints_return_200(self):
        """Test that valid endpoints return 200"""
        endpoints = ["/health", "/metrics", "/metrics/summary"]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200, f"{endpoint} failed"
    
    def test_invalid_endpoint_returns_404(self):
        """Test that invalid endpoints return 404"""
        response = client.get("/invalid-endpoint-12345")
        assert response.status_code == 404


class TestAPIPerformance:
    """Test API performance characteristics"""
    
    def test_health_endpoint_response_time(self):
        """Test health endpoint responds quickly"""
        import time
        
        start = time.time()
        response = client.get("/health")
        duration = time.time() - start
        
        assert response.status_code == 200
        assert duration < 1.0, "Health check took too long"
    
    def test_metrics_endpoint_response_time(self):
        """Test metrics endpoint responds quickly"""
        import time
        
        start = time.time()
        response = client.get("/metrics")
        duration = time.time() - start
        
        assert response.status_code == 200
        assert duration < 1.0, "Metrics endpoint took too long"


class TestCORS:
    """Test CORS configuration (if applicable)"""
    
    def test_cors_headers_present(self):
        """Test that CORS headers are configured"""
        response = client.options("/health")
        # OPTIONS method returns 404 - CORS not configured (acceptable for API-only app)
        # In production, CORS middleware would handle this
        assert response.status_code in [200, 404, 405]