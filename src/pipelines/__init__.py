# src/pipelines/__init__.py

"""
ML Pipelines module for retail customer analytics.

This module contains end-to-end pipelines for:
- Data processing and transformation
- Model training workflows
- Prediction and inference
- Model evaluation and comparison
"""

# Import all pipeline classes
from .data_pipeline import DataPipeline
from .training_pipeline import TrainingPipeline
from .prediction_pipeline import PredictionPipeline
from .evaluation_pipeline import EvaluationPipeline

# Pipeline registry
PIPELINES = {
    'data_pipeline': DataPipeline,
    'training_pipeline': TrainingPipeline,
    'prediction_pipeline': PredictionPipeline,
    'evaluation_pipeline': EvaluationPipeline
}

def get_pipeline(pipeline_name: str):
    """
    Get a pipeline class by name
    
    Args:
        pipeline_name: Name of the pipeline
        
    Returns:
        Pipeline class
    """
    if pipeline_name not in PIPELINES:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")
    return PIPELINES[pipeline_name]

def list_pipelines():
    """List all available pipelines"""
    return list(PIPELINES.keys())

def create_complete_workflow(config_path: str = "config/config.yaml"):
    """
    Create a complete ML workflow with all pipelines
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary of initialized pipelines
    """
    return {
        'data': DataPipeline(config_path),
        'training': TrainingPipeline(config_path),
        'prediction': PredictionPipeline(config_path),
        'evaluation': EvaluationPipeline(config_path)
    }

__all__ = [
    'DataPipeline',
    'TrainingPipeline',
    'PredictionPipeline', 
    'EvaluationPipeline',
    'get_pipeline',
    'list_pipelines',
    'create_complete_workflow',
    'PIPELINES'
]