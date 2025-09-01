# src/__init__.py

"""
Retail Customer Analytics Package

This package provides comprehensive tools for retail customer analysis including:
- Data preprocessing and feature engineering
- Customer segmentation and clustering
- Predictive modeling for churn, CLV, and purchase behavior
- Recommendation systems
- Model evaluation and performance analysis
"""

__version__ = "1.0.0"
__author__ = "Prasanth"
__email__ = "team@retailanalytics.com"

# Package imports
from . import components
from . import pipelines
from . import utils

# Version info
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'release': 'stable'
}

def get_version():
    """Get the package version"""
    return __version__

def get_version_info():
    """Get detailed version information"""
    return VERSION_INFO

# Package metadata
__all__ = [
    'components',
    'pipelines', 
    'utils',
    'get_version',
    'get_version_info'
]