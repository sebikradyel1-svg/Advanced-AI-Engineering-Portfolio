"""
MLflow experiment tracking for HR RAG system
Tracks model parameters, metrics, and experiments
"""
import os
from pathlib import Path
import json
from typing import Dict, Any, Optional
import logging
import mlflow
import mlflow.pyfunc
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MLflowTracker:
    """MLflow experiment tracking wrapper"""
    
    def __init__(self, experiment_name: str = "hr-rag-system", tracking_uri: str = None):
        """
        Initialize MLflow tracker
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (default: local file storage)
        """
        # Set tracking URI (local file storage by default)
        if tracking_uri is None:
            # Use file-based storage (simpler, no SQL needed)
            mlruns_path = Path(__file__).parent / "mlruns"
            mlruns_path.mkdir(exist_ok=True)
            # Don't set tracking_uri - use default file store
            tracking_uri = None
        
        # Only set if not None
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created new MLflow experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {experiment_name}")
            
            mlflow.set_experiment(experiment_name)
            self.experiment_name = experiment_name
            self.experiment_id = experiment_id
            
        except Exception as e:
            logger.error(f"Failed to initialize MLflow experiment: {e}")
            raise
    
    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run
        
        Args:
            run_name: Optional name for the run
            
        Returns:
            Active MLflow run
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return mlflow.start_run(run_name=run_name)
    
    def log_system_params(self, params: Dict[str, Any]):
        """
        Log system parameters
        
        Args:
            params: Dictionary of parameters to log
        """
        try:
            for key, value in params.items():
                # Convert complex types to strings
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                mlflow.log_param(key, value)
            logger.info(f"Logged {len(params)} parameters")
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for time series metrics
        """
        try:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
            logger.info(f"Logged {len(metrics)} metrics")
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_model_config(self, config: Dict[str, Any]):
        """
        Log model configuration as artifact
        
        Args:
            config: Model configuration dictionary
        """
        try:
            config_path = Path("temp_model_config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            mlflow.log_artifact(str(config_path), artifact_path="config")
            config_path.unlink()  # Delete temp file
            logger.info("Logged model configuration")
        except Exception as e:
            logger.error(f"Failed to log model config: {e}")
    
    def log_query_metrics(self, query: str, response_time: float, 
                         success: bool, response_length: int):
        """
        Log metrics for a single query
        
        Args:
            query: User query
            response_time: Time to generate response (seconds)
            success: Whether query was successful
            response_length: Length of response in characters
        """
        metrics = {
            "response_time_seconds": response_time,
            "success": 1.0 if success else 0.0,
            "response_length_chars": float(response_length),
            "query_length_chars": float(len(query))
        }
        
        self.log_metrics(metrics)
    
    def end_run(self):
        """End the current MLflow run"""
        try:
            mlflow.end_run()
            logger.info("Ended MLflow run")
        except Exception as e:
            logger.error(f"Failed to end run: {e}")
    
    def get_experiment_info(self) -> Dict[str, Any]:
        """
        Get information about the current experiment
        
        Returns:
            Dictionary with experiment information
        """
        try:
            experiment = mlflow.get_experiment(self.experiment_id)
            runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
            
            return {
                "experiment_name": experiment.name,
                "experiment_id": experiment.experiment_id,
                "artifact_location": experiment.artifact_location,
                "total_runs": len(runs),
                "latest_run": runs.iloc[0].to_dict() if len(runs) > 0 else None
            }
        except Exception as e:
            logger.error(f"Failed to get experiment info: {e}")
            return {}


# Global tracker instance
_tracker: Optional[MLflowTracker] = None


def get_tracker() -> MLflowTracker:
    """Get or create global MLflow tracker instance"""
    global _tracker
    if _tracker is None:
        _tracker = MLflowTracker()
    return _tracker


def initialize_mlflow(model_params: Dict[str, Any]) -> str:
    """
    Initialize MLflow tracking with system parameters
    
    Args:
        model_params: Model configuration parameters
        
    Returns:
        Run ID
    """
    tracker = get_tracker()
    run = tracker.start_run()
    
    # Log system parameters
    tracker.log_system_params(model_params)
    tracker.log_model_config(model_params)
    
    logger.info(f"MLflow run started: {run.info.run_id}")
    return run.info.run_id