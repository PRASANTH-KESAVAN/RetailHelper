# src/utils/common.py

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
import os

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise

def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory_path: Path to directory
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)

def load_sample_data() -> pd.DataFrame:
    """
    Load sample retail customer data for demonstration
    
    Returns:
        Sample dataframe
    """
    np.random.seed(42)
    n_customers = 1000
    
    # Generate realistic sample data
    data = {
        'Customer_ID': range(1, n_customers + 1),
        'Age': np.random.normal(40, 12, n_customers).astype(int),
        'Gender': np.random.choice(['Male', 'Female'], n_customers, p=[0.48, 0.52]),
        'Purchase_Amount': np.random.exponential(75, n_customers) + 20,
        'Purchase_Frequency': np.random.poisson(4, n_customers) + 1,
        'Category': np.random.choice(
            ['Electronics', 'Clothing', 'Books', 'Home', 'Sports'], 
            n_customers, 
            p=[0.25, 0.30, 0.15, 0.20, 0.10]
        ),
        'Rating': np.clip(np.random.normal(4.0, 0.8, n_customers), 1, 5).round(1),
        'Season': np.random.choice(
            ['Spring', 'Summer', 'Fall', 'Winter'], 
            n_customers,
            p=[0.25, 0.30, 0.28, 0.17]
        ),
        'Satisfaction_Score': np.clip(np.random.normal(7.5, 1.2, n_customers), 1, 10).round(1),
        'Previous_Purchases': np.random.poisson(8, n_customers),
        'Days_Since_Last_Purchase': np.random.exponential(30, n_customers).astype(int),
    }
    
    df = pd.DataFrame(data)
    
    # Add some realistic correlations
    # Higher age tends to have higher purchase amounts (slightly)
    age_factor = (df['Age'] - df['Age'].min()) / (df['Age'].max() - df['Age'].min())
    df['Purchase_Amount'] += age_factor * 50
    
    # More frequent purchasers tend to have higher satisfaction
    freq_factor = (df['Purchase_Frequency'] - df['Purchase_Frequency'].min()) / \
                  (df['Purchase_Frequency'].max() - df['Purchase_Frequency'].min())
    df['Satisfaction_Score'] += freq_factor * 1.5
    df['Satisfaction_Score'] = np.clip(df['Satisfaction_Score'], 1, 10)
    
    return df

def save_data(df: pd.DataFrame, file_path: str, format: str = 'csv') -> None:
    """
    Save dataframe to file
    
    Args:
        df: Dataframe to save
        file_path: Path to save file
        format: File format ('csv', 'parquet', 'json')
    """
    ensure_directory_exists(Path(file_path).parent)
    
    if format.lower() == 'csv':
        df.to_csv(file_path, index=False)
    elif format.lower() == 'parquet':
        df.to_parquet(file_path, index=False)
    elif format.lower() == 'json':
        df.to_json(file_path, orient='records', indent=2)
    else:
        raise ValueError(f"Unsupported file format: {format}")
    
    logger.info(f"Data saved to {file_path}")

def load_data(file_path: str, format: str = None) -> pd.DataFrame:
    """
    Load dataframe from file
    
    Args:
        file_path: Path to data file
        format: File format (auto-detected if None)
        
    Returns:
        Loaded dataframe
    """
    if format is None:
        # Auto-detect format from file extension
        format = Path(file_path).suffix.lower()[1:]
    
    if format == 'csv':
        df = pd.read_csv(file_path)
    elif format == 'parquet':
        df = pd.read_parquet(file_path)
    elif format == 'json':
        df = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {format}")
    
    logger.info(f"Data loaded from {file_path}. Shape: {df.shape}")
    return df

def calculate_business_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate key business metrics from customer data
    
    Args:
        df: Customer dataframe
        
    Returns:
        Dictionary of business metrics
    """
    metrics = {}
    
    # Customer metrics
    metrics['total_customers'] = len(df)
    metrics['avg_age'] = df['Age'].mean() if 'Age' in df.columns else 0
    
    # Purchase metrics
    if 'Purchase_Amount' in df.columns:
        metrics['total_revenue'] = df['Purchase_Amount'].sum()
        metrics['avg_order_value'] = df['Purchase_Amount'].mean()
        metrics['revenue_per_customer'] = metrics['total_revenue'] / metrics['total_customers']
    
    if 'Purchase_Frequency' in df.columns:
        metrics['avg_purchase_frequency'] = df['Purchase_Frequency'].mean()
        metrics['total_transactions'] = df['Purchase_Frequency'].sum()
    
    # Satisfaction metrics
    if 'Rating' in df.columns:
        metrics['avg_rating'] = df['Rating'].mean()
        metrics['high_satisfaction_rate'] = (df['Rating'] >= 4.0).mean() * 100
    
    if 'Satisfaction_Score' in df.columns:
        metrics['avg_satisfaction'] = df['Satisfaction_Score'].mean()
        metrics['satisfaction_std'] = df['Satisfaction_Score'].std()
    
    # Segmentation metrics
    if 'cluster_label' in df.columns:
        metrics['num_segments'] = df['cluster_label'].nunique()
        metrics['largest_segment_size'] = df['cluster_label'].value_counts().iloc[0]
        metrics['smallest_segment_size'] = df['cluster_label'].value_counts().iloc[-1]
    
    return metrics

def format_currency(amount: float, currency: str = 'USD') -> str:
    """
    Format amount as currency string
    
    Args:
        amount: Amount to format
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    if currency.upper() == 'USD':
        return f"${amount:,.2f}"
    elif currency.upper() == 'EUR':
        return f"€{amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"

def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format value as percentage string
    
    Args:
        value: Value to format (0-1 or 0-100)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    if value <= 1:
        value *= 100
    return f"{value:.{decimals}f}%"

def validate_dataframe_schema(df: pd.DataFrame, expected_columns: list) -> Dict[str, Any]:
    """
    Validate dataframe schema against expected columns
    
    Args:
        df: Dataframe to validate
        expected_columns: List of expected column names
        
    Returns:
        Validation results dictionary
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'extra_columns': [],
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict()
    }
    
    actual_columns = set(df.columns)
    expected_columns = set(expected_columns)
    
    validation_results['missing_columns'] = list(expected_columns - actual_columns)
    validation_results['extra_columns'] = list(actual_columns - expected_columns)
    
    if validation_results['missing_columns'] or validation_results['extra_columns']:
        validation_results['is_valid'] = False
    
    return validation_results

def create_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics for all features in dataframe
    
    Args:
        df: Input dataframe
        
    Returns:
        Summary dataframe
    """
    summary_data = []
    
    for column in df.columns:
        col_data = {
            'feature': column,
            'dtype': str(df[column].dtype),
            'missing_count': df[column].isnull().sum(),
            'missing_percentage': (df[column].isnull().sum() / len(df)) * 100,
            'unique_values': df[column].nunique(),
            'memory_usage': df[column].memory_usage(deep=True)
        }
        
        if df[column].dtype in ['int64', 'float64']:
            col_data.update({
                'mean': df[column].mean(),
                'std': df[column].std(),
                'min': df[column].min(),
                'max': df[column].max(),
                'median': df[column].median()
            })
        elif df[column].dtype == 'object':
            col_data.update({
                'most_frequent': df[column].mode().iloc[0] if not df[column].mode().empty else None,
                'frequency': df[column].value_counts().iloc[0] if not df[column].empty else 0
            })
        
        summary_data.append(col_data)
    
    return pd.DataFrame(summary_data)