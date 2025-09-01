# src/components/data_preprocessing.py

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from loguru import logger
import yaml

class DataPreprocessor:
    """
    Data preprocessing component for retail customer analytics
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load raw data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        logger.info("Handling missing values...")
        
        # Check for missing values
        missing_stats = df.isnull().sum()
        logger.info(f"Missing values per column:\n{missing_stats[missing_stats > 0]}")
        
        # Numeric columns - use median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Categorical columns - use mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        logger.info("Missing values handled successfully")
        return df
    
    def encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        logger.info("Encoding categorical variables...")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            
            df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col])
        
        logger.info(f"Encoded {len(categorical_cols)} categorical variables")
        return df
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for analysis"""
        logger.info("Creating derived features...")
        
        # Purchase frequency features
        if 'Customer ID' in df.columns and 'Invoice Date' in df.columns:
            customer_stats = df.groupby('Customer ID').agg({
                'Invoice Date': 'count',
                'Quantity': ['sum', 'mean'],
                'Price': ['sum', 'mean'],
                'Age': 'first'
            }).round(2)
            
            customer_stats.columns = [
                'purchase_frequency', 'total_quantity', 'avg_quantity',
                'total_spent', 'avg_spent', 'age'
            ]
            
            # Customer lifetime value
            customer_stats['clv'] = customer_stats['total_spent'] * customer_stats['purchase_frequency']
            
            # Recency calculation (days since last purchase)
            if 'Invoice Date' in df.columns:
                df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])
                last_purchase = df.groupby('Customer ID')['Invoice Date'].max()
                reference_date = df['Invoice Date'].max()
                customer_stats['recency'] = (reference_date - last_purchase).dt.days
            
            df = df.merge(customer_stats, left_on='Customer ID', right_index=True, how='left')
        
        logger.info("Derived features created successfully")
        return df
    
    def normalize_features(self, df: pd.DataFrame, 
                          features_to_normalize: Optional[list] = None) -> pd.DataFrame:
        """Normalize numerical features"""
        logger.info("Normalizing features...")
        
        if features_to_normalize is None:
            # Auto-select numeric columns for normalization
            features_to_normalize = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude encoded categorical variables
            features_to_normalize = [col for col in features_to_normalize 
                                   if not col.endswith('_encoded')]
        
        df[features_to_normalize] = self.scaler.fit_transform(df[features_to_normalize])
        
        logger.info(f"Normalized {len(features_to_normalize)} features")
        return df
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Detect and handle outliers"""
        logger.info(f"Detecting outliers using {method} method...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_indices = set()
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
                outlier_indices.update(outliers)
        
        df['is_outlier'] = df.index.isin(outlier_indices)
        logger.info(f"Detected {len(outlier_indices)} outliers ({len(outlier_indices)/len(df)*100:.2f}%)")
        
        return df
    
    def preprocess_pipeline(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Complete preprocessing pipeline"""
        logger.info("Starting preprocessing pipeline...")
        
        preprocessing_stats = {
            'original_shape': df.shape,
            'missing_values_before': df.isnull().sum().sum()
        }
        
        # Step 1: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 2: Create derived features
        df = self.create_derived_features(df)
        
        # Step 3: Encode categorical variables
        df = self.encode_categorical_variables(df)
        
        # Step 4: Detect outliers
        df = self.detect_outliers(df)
        
        # Step 5: Normalize features (optional, for specific algorithms)
        # df = self.normalize_features(df)
        
        preprocessing_stats.update({
            'final_shape': df.shape,
            'missing_values_after': df.isnull().sum().sum(),
            'features_created': df.shape[1] - preprocessing_stats['original_shape'][1]
        })
        
        logger.info("Preprocessing pipeline completed successfully")
        logger.info(f"Final dataset shape: {df.shape}")
        
        return df, preprocessing_stats
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """Save processed data"""
        try:
            df.to_csv(output_path, index=False)
            logger.info(f"Processed data saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise