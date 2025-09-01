# config/__init__.py

"""
Configuration module for retail customer analytics project.

This module contains configuration utilities and settings for the entire project.
"""

__version__ = "1.0.0"
__author__ = "Prasanth "

from pathlib import Path
import yaml
import os
from typing import Dict, Any

# Get the config directory path
CONFIG_DIR = Path(__file__).parent

def load_main_config(config_file: str = "config.yaml") -> Dict[str, Any]:
    """
    Load the main configuration file
    
    Args:
        config_file: Name of the config file
        
    Returns:
        Configuration dictionary
    """
    config_path = CONFIG_DIR / config_file
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Replace environment variables
    config = _replace_env_vars(config)
    
    return config

def load_database_config(config_file: str = "database_config.yaml") -> Dict[str, Any]:
    """
    Load the database configuration file
    
    Args:
        config_file: Name of the database config file
        
    Returns:
        Database configuration dictionary
    """
    config_path = CONFIG_DIR / config_file
    
    if not config_path.exists():
        raise FileNotFoundError(f"Database configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Replace environment variables
    config = _replace_env_vars(config)
    
    return config

def _replace_env_vars(obj: Any) -> Any:
    """
    Recursively replace environment variable placeholders in configuration
    
    Args:
        obj: Configuration object (dict, list, or string)
        
    Returns:
        Object with environment variables replaced
    """
    if isinstance(obj, dict):
        return {k: _replace_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_replace_env_vars(item) for item in obj]
    elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
        # Extract environment variable name
        env_var = obj[2:-1]
        # Handle default values (e.g., ${VAR_NAME:default_value})
        if ':' in env_var:
            var_name, default_value = env_var.split(':', 1)
            return os.getenv(var_name, default_value)
        else:
            return os.getenv(env_var, obj)  # Return original if not found
    else:
        return obj

# Make configuration loading functions available at package level
__all__ = ['load_main_config', 'load_database_config', 'CONFIG_DIR']