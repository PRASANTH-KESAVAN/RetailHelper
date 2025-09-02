# src/components/__init__.py

"""
Core components module for retail customer analytics.

This module contains the core analytical components including:
- Data ingestion and preprocessing
- Feature engineering 
- Customer segmentation
- Predictive modeling
- Recommendation engines
- Model evaluation
"""

# Import all components
from .data_ingestion import DataIngestion
from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineering
from .customer_segmentation import CustomerSegmentation
from .predictive_modeling import PredictiveModeling
from .recommendation_engine import RecommendationEngine
from .model_evaluation import ModelEvaluation

# Component registry
COMPONENTS = {
    'data_ingestion': DataIngestion,
    'data_preprocessing': DataPreprocessor,
    'feature_engineering': FeatureEngineering,
    'customer_segmentation': CustomerSegmentation,
    'predictive_modeling': PredictiveModeling,
    'recommendation_engine': RecommendationEngine,
    'model_evaluation': ModelEvaluation
}

def get_component(component_name: str):
    """
    Get a component class by name
    
    Args:
        component_name: Name of the component
        
    Returns:
        Component class
    """
    if component_name not in COMPONENTS:
        raise ValueError(f"Unknown component: {component_name}")
    return COMPONENTS[component_name]

def list_components():
    """List all available components"""
    return list(COMPONENTS.keys())

__all__ = [
    'DataIngestion',
    'DataPreprocessor', 
    'FeatureEngineering',
    'CustomerSegmentation',
    'PredictiveModeling',
    'RecommendationEngine',
    'ModelEvaluation',
    'get_component',
    'list_components',
    'COMPONENTS'
]