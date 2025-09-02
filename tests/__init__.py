# tests/__init__.py

"""
Test suite for retail customer analytics project.

This package contains comprehensive tests for all components and pipelines.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Test configuration
TEST_DATA_SIZE = 100  # Number of sample records for tests
RANDOM_SEED = 42

# Test fixtures and utilities
def create_sample_customer_data(n_records: int = TEST_DATA_SIZE) -> pd.DataFrame:
    """
    Create sample customer data for testing
    
    Args:
        n_records: Number of records to generate
        
    Returns:
        Sample DataFrame with customer data
    """
    np.random.seed(RANDOM_SEED)
    
    data = {
        'Customer ID': [f'CUST_{i:05d}' for i in range(1, n_records + 1)],
        'Age': np.random.randint(18, 80, n_records),
        'Gender': np.random.choice(['Male', 'Female'], n_records),
        'Location': np.random.choice(['New York', 'California', 'Texas', 'Florida', 'Illinois'], n_records),
        'Category': np.random.choice(['Clothing', 'Electronics', 'Books', 'Home & Garden', 'Sports'], n_records),
        'Purchase Amount (USD)': np.random.exponential(50, n_records) + 10,
        'Purchase Date': pd.date_range('2023-01-01', '2024-12-31', periods=n_records),
        'Review Rating': np.random.uniform(1, 5, n_records),
        'Subscription Status': np.random.choice(['Yes', 'No'], n_records),
        'Payment Method': np.random.choice(['Credit Card', 'Debit Card', 'Cash', 'PayPal'], n_records),
        'Previous Purchases': np.random.poisson(3, n_records),
        'Promo Code Used': np.random.choice(['Yes', 'No'], n_records, p=[0.3, 0.7])
    }
    
    return pd.DataFrame(data)

def create_sample_processed_data(n_records: int = TEST_DATA_SIZE) -> pd.DataFrame:
    """
    Create sample processed data for testing advanced components
    
    Args:
        n_records: Number of records to generate
        
    Returns:
        Sample processed DataFrame
    """
    df = create_sample_customer_data(n_records)
    
    # Add derived features for testing
    df['Total_Spent'] = df['Purchase Amount (USD)'] * df['Previous Purchases']
    df['Avg_Rating'] = df['Review Rating']
    df['Is_Subscriber'] = (df['Subscription Status'] == 'Yes').astype(int)
    df['Days_Since_Last_Purchase'] = np.random.randint(1, 365, n_records)
    df['Purchase_Count'] = df['Previous Purchases']
    df['Recency'] = df['Days_Since_Last_Purchase']
    df['Frequency'] = df['Purchase_Count']
    df['Monetary'] = df['Total_Spent']
    
    return df

# Test configuration constants
TEST_CONFIG = {
    'random_state': RANDOM_SEED,
    'test_size': 0.2,
    'cv_folds': 3,
    'n_clusters': 3,
    'max_clusters': 5,
    'tolerance': 1e-6
}

__all__ = [
    'create_sample_customer_data',
    'create_sample_processed_data', 
    'TEST_CONFIG',
    'TEST_DATA_SIZE',
    'RANDOM_SEED'
]