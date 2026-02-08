"""
MLOps Monitoring System for HR RAG Application
Tracks requests, responses, performance metrics, and errors
"""

import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import threading
from collections import defaultdict

# Thread-safe metrics storage
_metrics_lock = threading.Lock()
_metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_response_time": 0.0,
    "average_response_time": 0.0,
    "requests_by_hour": defaultdict(int),
    "error_types": defaultdict(int),
    "uptime_start": datetime.now().isoformat(),
}

@dataclass
class RequestLog:
    """Structure for logging individual requests"""
    timestamp: str
    query: str
    response_time: float
    success: bool
    error_message: Optional[str] = None
    response_length: int = 0
    model_used: str = "groq-llama"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MonitoringSystem:
    """Centralized monitoring system for RAG application"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.requests_log_file = self.log_dir / f"requests_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
    def log_request(self, request_log: RequestLog):
        """Log individual request to JSONL file"""
        with open(self.requests_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(request_log.to_dict()) + '\n')
    
    def update_metrics(self, response_time: float, success: bool, error_type: Optional[str] = None):
        """Update global metrics (thread-safe)"""
        with _metrics_lock:
            _metrics["total_requests"] += 1
            _metrics["total_response_time"] += response_time
            
            if success:
                _metrics["successful_requests"] += 1
            else:
                _metrics["failed_requests"] += 1
                if error_type:
                    _metrics["error_types"][error_type] += 1
            
            # Update average
            _metrics["average_response_time"] = (
                _metrics["total_response_time"] / _metrics["total_requests"]
            )
            
            # Track requests by hour
            current_hour = datetime.now().strftime("%Y-%m-%d %H:00")
            _metrics["requests_by_hour"][current_hour] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        with _metrics_lock:
            # Calculate uptime
            uptime_start = datetime.fromisoformat(_metrics["uptime_start"])
            uptime_seconds = (datetime.now() - uptime_start).total_seconds()
            
            return {
                **_metrics,
                "uptime_seconds": uptime_seconds,
                "uptime_hours": round(uptime_seconds / 3600, 2),
                "success_rate": (
                    (_metrics["successful_requests"] / _metrics["total_requests"] * 100)
                    if _metrics["total_requests"] > 0 else 0
                ),
                "error_types": dict(_metrics["error_types"]),
                "requests_by_hour": dict(_metrics["requests_by_hour"]),
            }
    
    def track_query(self, query: str):
        """Context manager for tracking query execution"""
        return QueryTracker(self, query)


class QueryTracker:
    """Context manager for tracking individual queries"""
    
    def __init__(self, monitoring_system: MonitoringSystem, query: str):
        self.monitoring_system = monitoring_system
        self.query = query
        self.start_time = None
        self.response_time = 0.0
        self.success = False
        self.error_message = None
        self.response_length = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.response_time = time.time() - self.start_time
        
        if exc_type is not None:
            self.success = False
            self.error_message = str(exc_val)
            error_type = exc_type.__name__
        else:
            self.success = True
            error_type = None
        
        # Log request
        request_log = RequestLog(
            timestamp=datetime.now().isoformat(),
            query=self.query[:200],  # Truncate long queries
            response_time=round(self.response_time, 3),
            success=self.success,
            error_message=self.error_message,
            response_length=self.response_length,
        )
        self.monitoring_system.log_request(request_log)
        
        # Update metrics
        self.monitoring_system.update_metrics(
            self.response_time,
            self.success,
            error_type
        )
        
        # Don't suppress exceptions
        return False
    
    def set_response(self, response: str):
        """Set response length for metrics"""
        self.response_length = len(response)


# Global monitoring instance
monitoring = MonitoringSystem()