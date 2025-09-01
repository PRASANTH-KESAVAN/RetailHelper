# src/components/data_ingestion.py

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import sqlite3
from sqlalchemy import create_engine
from loguru import logger
import requests
from pathlib import Path
import os

class DataIngestion:
    """
    Data ingestion component for loading data from various sources
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.supported_formats = ['csv', 'excel', 'json', 'parquet', 'sql']
        
    def load_from_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional parameters for pd.read_csv
            
        Returns:
            DataFrame with loaded data
        """
        try:
            logger.info(f"Loading data from CSV: {file_path}")
            
            # Default parameters
            default_params = {
                'encoding': 'utf-8',
                'low_memory': False
            }
            default_params.update(kwargs)
            
            df = pd.read_csv(file_path, **default_params)
            
            logger.info(f"Successfully loaded {len(df)} records from {file_path}")
            logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"File is empty: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise
    
    def load_from_excel(self, file_path: str, sheet_name: str = 0, **kwargs) -> pd.DataFrame:
        """
        Load data from Excel file
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name or index
            **kwargs: Additional parameters for pd.read_excel
            
        Returns:
            DataFrame with loaded data
        """
        try:
            logger.info(f"Loading data from Excel: {file_path}, Sheet: {sheet_name}")
            
            df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            
            logger.info(f"Successfully loaded {len(df)} records from Excel file")
            return df
            
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            raise
    
    def load_from_json(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from JSON file
        
        Args:
            file_path: Path to JSON file
            **kwargs: Additional parameters for pd.read_json
            
        Returns:
            DataFrame with loaded data
        """
        try:
            logger.info(f"Loading data from JSON: {file_path}")
            
            # Default parameters
            default_params = {
                'orient': 'records'
            }
            default_params.update(kwargs)
            
            df = pd.read_json(file_path, **default_params)
            
            logger.info(f"Successfully loaded {len(df)} records from JSON file")
            return df
            
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            raise
    
    def load_from_database(self, connection_string: str, query: str, **kwargs) -> pd.DataFrame:
        """
        Load data from database
        
        Args:
            connection_string: Database connection string
            query: SQL query to execute
            **kwargs: Additional parameters for pd.read_sql
            
        Returns:
            DataFrame with loaded data
        """
        try:
            logger.info("Loading data from database")
            
            engine = create_engine(connection_string)
            df = pd.read_sql(query, engine, **kwargs)
            
            logger.info(f"Successfully loaded {len(df)} records from database")
            return df
            
        except Exception as e:
            logger.error(f"Error loading from database: {e}")
            raise
    
    def download_kaggle_dataset(self, dataset_name: str, file_name: str = None) -> str:
        """
        Download dataset from Kaggle (requires kaggle API setup)
        
        Args:
            dataset_name: Kaggle dataset identifier (e.g., 'username/dataset-name')
            file_name: Specific file to download (optional)
            
        Returns:
            Path to downloaded file
        """
        try:
            import kaggle
            
            logger.info(f"Downloading Kaggle dataset: {dataset_name}")
            
            # Create download directory
            download_path = Path(self.config['data']['raw_data_path'])
            download_path.mkdir(parents=True, exist_ok=True)
            
            # Download dataset
            kaggle.api.dataset_download_files(
                dataset_name, 
                path=str(download_path), 
                unzip=True
            )
            
            # Find the downloaded file
            if file_name:
                file_path = download_path / file_name
            else:
                # Get the first CSV file in the directory
                csv_files = list(download_path.glob('*.csv'))
                if csv_files:
                    file_path = csv_files[0]
                else:
                    raise FileNotFoundError("No CSV files found in downloaded dataset")
            
            logger.info(f"Dataset downloaded successfully: {file_path}")
            return str(file_path)
            
        except ImportError:
            logger.error("Kaggle API not installed. Run: pip install kaggle")
            raise
        except Exception as e:
            logger.error(f"Error downloading Kaggle dataset: {e}")
            raise
    
    def load_sample_retail_data(self, n_customers: int = 1000) -> pd.DataFrame:
        """
        Generate sample retail customer data for testing
        
        Args:
            n_customers: Number of customers to generate
            
        Returns:
            DataFrame with sample retail data
        """
        logger.info(f"Generating sample retail data for {n_customers} customers")
        
        np.random.seed(42)
        
        # Customer demographics
        customer_ids = [f"CUST_{i:06d}" for i in range(1, n_customers + 1)]
        ages = np.random.normal(40, 12, n_customers).astype(int)
        ages = np.clip(ages, 18, 80)  # Reasonable age range
        
        genders = np.random.choice(['Male', 'Female', 'Other'], n_customers, p=[0.48, 0.50, 0.02])
        
        # Purchase behavior
        categories = ['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports', 'Beauty']
        item_purchased = np.random.choice(categories, n_customers)
        
        # Purchase amounts with realistic distribution
        base_amounts = np.random.exponential(50, n_customers) + 10
        # Add category-based pricing
        category_multipliers = {
            'Electronics': 2.5, 'Clothing': 1.2, 'Books': 0.8, 
            'Home & Garden': 1.8, 'Sports': 1.5, 'Beauty': 1.0
        }
        purchase_amounts = []
        for i, category in enumerate(item_purchased):
            purchase_amounts.append(base_amounts[i] * category_multipliers[category])
        
        purchase_amounts = np.array(purchase_amounts)
        
        # Review ratings
        review_ratings = np.random.normal(4.0, 0.8, n_customers)
        review_ratings = np.clip(review_ratings, 1, 5).round(1)
        
        # Purchase frequency and previous purchases
        purchase_frequencies = np.random.poisson(3, n_customers) + 1
        previous_purchases = np.random.poisson(8, n_customers)
        
        # Location data
        locations = ['California', 'Texas', 'New York', 'Florida', 'Illinois', 
                    'Pennsylvania', 'Ohio', 'Georgia', 'North Carolina', 'Michigan']
        customer_locations = np.random.choice(locations, n_customers)
        
        # Seasonal data
        seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        purchase_seasons = np.random.choice(seasons, n_customers, p=[0.25, 0.30, 0.28, 0.17])
        
        # Payment methods
        payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Cash', 'Bank Transfer']
        payment_method = np.random.choice(payment_methods, n_customers, p=[0.45, 0.25, 0.15, 0.10, 0.05])
        
        # Create synthetic dates
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2024-12-31')
        date_range = pd.date_range(start_date, end_date)
        purchase_dates = np.random.choice(date_range, n_customers)
        
        # Create DataFrame
        data = {
            'Customer ID': customer_ids,
            'Age': ages,
            'Gender': genders,
            'Item Purchased': item_purchased,
            'Category': item_purchased,  # Same as item for simplicity
            'Purchase Amount (USD)': np.round(purchase_amounts, 2),
            'Location': customer_locations,
            'Size': np.random.choice(['S', 'M', 'L', 'XL'], n_customers),
            'Color': np.random.choice(['Red', 'Blue', 'Green', 'Black', 'White'], n_customers),
            'Season': purchase_seasons,
            'Review Rating': review_ratings,
            'Subscription Status': np.random.choice(['Yes', 'No'], n_customers, p=[0.3, 0.7]),
            'Payment Method': payment_method,
            'Shipping Type': np.random.choice(['Standard', 'Express', 'Overnight'], n_customers, p=[0.6, 0.3, 0.1]),
            'Discount Applied': np.random.choice(['Yes', 'No'], n_customers, p=[0.4, 0.6]),
            'Promo Code Used': np.random.choice(['Yes', 'No'], n_customers, p=[0.2, 0.8]),
            'Previous Purchases': previous_purchases,
            'Purchase Date': purchase_dates,
            'Purchase Frequency': purchase_frequencies
        }
        
        df = pd.DataFrame(data)
        
        # Add some correlations to make it more realistic
        # Higher age -> slightly higher spending
        age_factor = (df['Age'] - 18) / (80 - 18)
        df['Purchase Amount (USD)'] += age_factor * 30
        
        # Subscription customers spend more
        subscription_mask = df['Subscription Status'] == 'Yes'
        df.loc[subscription_mask, 'Purchase Amount (USD)'] *= 1.3
        
        # Previous purchases correlate with higher ratings
        prev_purchase_factor = np.clip(df['Previous Purchases'] / 20, 0, 1)
        df['Review Rating'] += prev_purchase_factor * 0.5
        df['Review Rating'] = np.clip(df['Review Rating'], 1, 5).round(1)
        
        logger.info("Sample retail data generated successfully")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform basic data quality validation
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Data quality report
        """
        logger.info("Performing data quality validation")
        
        quality_report = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_records': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024**2), 2)
        }
        
        # Check for completely empty columns
        empty_columns = df.columns[df.isnull().all()].tolist()
        quality_report['empty_columns'] = empty_columns
        
        # Check for columns with single unique value
        single_value_columns = []
        for col in df.columns:
            if df[col].nunique() == 1:
                single_value_columns.append(col)
        quality_report['single_value_columns'] = single_value_columns
        
        # Numeric column statistics
        numeric_stats = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            numeric_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'zeros': (df[col] == 0).sum(),
                'negatives': (df[col] < 0).sum()
            }
        quality_report['numeric_stats'] = numeric_stats
        
        logger.info("Data quality validation completed")
        return quality_report
    
    def save_data(self, df: pd.DataFrame, file_path: str, format: str = 'csv'):
        """
        Save DataFrame to file
        
        Args:
            df: DataFrame to save
            file_path: Output file path
            format: Output format ('csv', 'excel', 'json', 'parquet')
        """
        try:
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'csv':
                df.to_csv(file_path, index=False)
            elif format.lower() == 'excel':
                df.to_excel(file_path, index=False)
            elif format.lower() == 'json':
                df.to_json(file_path, orient='records', indent=2)
            elif format.lower() == 'parquet':
                df.to_parquet(file_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Data saved successfully to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise