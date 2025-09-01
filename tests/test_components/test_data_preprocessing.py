# tests/test_components/test_data_preprocessing.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from components.data_preprocessing import DataPreprocessor
from tests import create_sample_customer_data, TEST_CONFIG

class TestDataPreprocessor:
    """Test suite for DataPreprocessor component"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return create_sample_customer_data(100)
    
    @pytest.fixture
    def preprocessor(self):
        """Create DataPreprocessor instance"""
        config_path = "config/config.yaml"
        with patch('yaml.safe_load') as mock_yaml:
            mock_yaml.return_value = {
                'preprocessing': {
                    'missing_threshold': 0.5,
                    'outlier_method': 'iqr',
                    'outlier_threshold': 1.5
                }
            }
            with patch('builtins.open', create=True):
                return DataPreprocessor(config_path)
    
    def test_initialization(self, preprocessor):
        """Test DataPreprocessor initialization"""
        assert preprocessor is not None
        assert hasattr(preprocessor, 'config')
        assert hasattr(preprocessor, 'processed_data')
    
    def test_handle_missing_values(self, preprocessor, sample_data):
        """Test missing value handling"""
        # Add some missing values
        sample_data.loc[0:5, 'Age'] = np.nan
        sample_data.loc[10:15, 'Purchase Amount (USD)'] = np.nan
        
        # Test missing value detection
        missing_info = preprocessor.analyze_missing_values(sample_data)
        
        assert 'Age' in missing_info['columns_with_missing']
        assert 'Purchase Amount (USD)' in missing_info['columns_with_missing']
        assert missing_info['total_missing'] > 0
    
    def test_missing_value_imputation(self, preprocessor, sample_data):
        """Test missing value imputation"""
        # Add missing values
        sample_data.loc[0:5, 'Age'] = np.nan
        sample_data.loc[10:15, 'Purchase Amount (USD)'] = np.nan
        
        # Impute missing values
        imputed_data = preprocessor.handle_missing_values(sample_data)
        
        # Check that missing values are filled
        assert imputed_data['Age'].isnull().sum() == 0
        assert imputed_data['Purchase Amount (USD)'].isnull().sum() == 0
    
    def test_outlier_detection_iqr(self, preprocessor, sample_data):
        """Test IQR-based outlier detection"""
        outliers = preprocessor.detect_outliers(
            sample_data, 
            columns=['Purchase Amount (USD)'], 
            method='iqr'
        )
        
        assert isinstance(outliers, dict)
        assert 'Purchase Amount (USD)' in outliers
        assert isinstance(outliers['Purchase Amount (USD)'], pd.Series)
    
    def test_outlier_detection_zscore(self, preprocessor, sample_data):
        """Test Z-score based outlier detection"""
        outliers = preprocessor.detect_outliers(
            sample_data,
            columns=['Purchase Amount (USD)'],
            method='zscore'
        )
        
        assert isinstance(outliers, dict)
        assert 'Purchase Amount (USD)' in outliers
    
    def test_data_type_conversion(self, preprocessor, sample_data):
        """Test data type conversion"""
        # Convert to string first
        sample_data['Age'] = sample_data['Age'].astype(str)
        
        converted_data = preprocessor.convert_data_types(sample_data)
        
        # Age should be converted back to numeric
        assert pd.api.types.is_numeric_dtype(converted_data['Age'])
    
    def test_date_parsing(self, preprocessor, sample_data):
        """Test date column parsing"""
        # Convert date to string
        sample_data['Purchase Date'] = sample_data['Purchase Date'].astype(str)
        
        parsed_data = preprocessor.parse_dates(sample_data, ['Purchase Date'])
        
        assert pd.api.types.is_datetime64_any_dtype(parsed_data['Purchase Date'])
    
    def test_duplicate_removal(self, preprocessor, sample_data):
        """Test duplicate record removal"""
        # Add duplicate rows
        duplicate_data = pd.concat([sample_data, sample_data.iloc[:5]], ignore_index=True)
        
        cleaned_data = preprocessor.remove_duplicates(duplicate_data)
        
        assert len(cleaned_data) < len(duplicate_data)
        assert not cleaned_data.duplicated().any()
    
    def test_categorical_encoding(self, preprocessor, sample_data):
        """Test categorical variable encoding"""
        encoded_data = preprocessor.encode_categorical_variables(sample_data)
        
        # Check that categorical columns are encoded
        assert 'Gender_Male' in encoded_data.columns or 'Gender_encoded' in encoded_data.columns
    
    def test_feature_scaling(self, preprocessor, sample_data):
        """Test feature scaling"""
        numeric_cols = ['Age', 'Purchase Amount (USD)', 'Previous Purchases']
        
        scaled_data, scaler = preprocessor.scale_features(sample_data, numeric_cols)
        
        # Check that features are scaled
        for col in numeric_cols:
            if col in scaled_data.columns:
                assert abs(scaled_data[col].mean()) < 1e-10  # Should be close to 0
                assert abs(scaled_data[col].std() - 1) < 1e-10  # Should be close to 1
    
    def test_preprocess_pipeline(self, preprocessor, sample_data):
        """Test complete preprocessing pipeline"""
        # Add some data quality issues
        sample_data.loc[0:2, 'Age'] = np.nan
        sample_data = pd.concat([sample_data, sample_data.iloc[:3]], ignore_index=True)
        
        processed_data, processing_report = preprocessor.preprocess_pipeline(sample_data)
        
        assert processed_data is not None
        assert isinstance(processing_report, dict)
        assert 'original_shape' in processing_report
        assert 'final_shape' in processing_report
        assert len(processed_data) <= len(sample_data)  # Should remove duplicates
    
    def test_data_validation(self, preprocessor, sample_data):
        """Test data validation"""
        validation_report = preprocessor.validate_data_quality(sample_data)
        
        assert isinstance(validation_report, dict)
        assert 'missing_values' in validation_report
        assert 'data_types' in validation_report
        assert 'duplicates' in validation_report
    
    def test_column_mapping(self, preprocessor):
        """Test column name standardization"""
        test_data = pd.DataFrame({
            'customer_id': ['C1', 'C2'],
            'PURCHASE_AMOUNT': [100, 200],
            'review.rating': [4.5, 3.0]
        })
        
        standardized_data = preprocessor.standardize_column_names(test_data)
        
        # Check that columns are standardized
        expected_cols = ['customer_id', 'purchase_amount', 'review_rating']
        for col in expected_cols:
            assert col in standardized_data.columns
    
    def test_error_handling(self, preprocessor):
        """Test error handling with invalid data"""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError):
            preprocessor.preprocess_pipeline(empty_df)
        
        # Test with invalid column names
        invalid_df = pd.DataFrame({'': [1, 2, 3]})
        
        result = preprocessor.validate_data_quality(invalid_df)
        assert 'errors' in result
    
    def test_custom_preprocessing_functions(self, preprocessor, sample_data):
        """Test custom preprocessing functions"""
        # Test age binning
        binned_data = preprocessor.create_age_bins(sample_data)
        assert 'Age_Bin' in binned_data.columns
        
        # Test purchase amount categorization
        categorized_data = preprocessor.categorize_purchase_amounts(sample_data)
        assert 'Purchase_Category' in categorized_data.columns
    
    def test_preprocessing_consistency(self, preprocessor, sample_data):
        """Test that preprocessing is consistent across runs"""
        # Run preprocessing twice
        result1, _ = preprocessor.preprocess_pipeline(sample_data.copy())
        result2, _ = preprocessor.preprocess_pipeline(sample_data.copy())
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_memory_efficiency(self, preprocessor):
        """Test memory efficiency with large datasets"""
        # Create larger dataset
        large_data = create_sample_customer_data(1000)
        
        # Check memory usage before and after
        memory_before = large_data.memory_usage(deep=True).sum()
        
        processed_data, _ = preprocessor.preprocess_pipeline(large_data)
        memory_after = processed_data.memory_usage(deep=True).sum()
        
        # Memory should not increase dramatically
        assert memory_after <= memory_before * 2
    
    @pytest.mark.parametrize("method", ["iqr", "zscore", "isolation_forest"])
    def test_outlier_methods(self, preprocessor, sample_data, method):
        """Test different outlier detection methods"""
        try:
            outliers = preprocessor.detect_outliers(
                sample_data,
                columns=['Purchase Amount (USD)'],
                method=method
            )
            assert isinstance(outliers, dict)
        except ImportError:
            # Skip if method not available
            pytest.skip(f"Method {method} not available")
    
    def test_preprocessing_config(self, preprocessor):
        """Test preprocessing configuration"""
        # Test config loading and validation
        assert preprocessor.config is not None
        
        # Test config parameters
        preprocessing_config = preprocessor.config.get('preprocessing', {})
        assert isinstance(preprocessing_config, dict)
    
    def test_data_integrity(self, preprocessor, sample_data):
        """Test data integrity after preprocessing"""
        original_customer_count = sample_data['Customer ID'].nunique()
        
        processed_data, _ = preprocessor.preprocess_pipeline(sample_data)
        
        # Customer count should not increase (may decrease due to deduplication)
        processed_customer_count = processed_data['Customer ID'].nunique()
        assert processed_customer_count <= original_customer_count