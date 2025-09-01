# tests/test_pipelines/test_data_pipeline.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pipelines.data_pipeline import DataPipeline
from tests import create_sample_customer_data, TEST_CONFIG

class TestDataPipeline:
    """Test suite for DataPipeline"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return create_sample_customer_data(100)
    
    @pytest.fixture
    def data_pipeline(self):
        """Create DataPipeline instance"""
        config_path = "config/config.yaml"
        with patch('yaml.safe_load') as mock_yaml:
            mock_yaml.return_value = {
                'data_sources': {
                    'csv_file': 'data/raw/customer_data.csv',
                    'database': {
                        'enabled': False
                    }
                },
                'preprocessing': {
                    'missing_threshold': 0.5,
                    'remove_duplicates': True,
                    'outlier_method': 'iqr'
                },
                'feature_engineering': {
                    'create_rfm': True,
                    'create_behavioral': True
                },
                'paths': {
                    'processed_data': 'data/processed/'
                }
            }
            with patch('builtins.open', create=True):
                return DataPipeline(config_path)
    
    def test_initialization(self, data_pipeline):
        """Test DataPipeline initialization"""
        assert data_pipeline is not None
        assert hasattr(data_pipeline, 'config')
        assert hasattr(data_pipeline, 'data_ingestion')
        assert hasattr(data_pipeline, 'preprocessor')
        assert hasattr(data_pipeline, 'feature_engineer')
    
    def test_load_data_from_csv(self, data_pipeline, sample_data, tmp_path):
        """Test loading data from CSV file"""
        # Create temporary CSV file
        csv_file = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_file, index=False)
        
        with patch.object(data_pipeline.data_ingestion, 'load_from_csv') as mock_load:
            mock_load.return_value = sample_data
            
            loaded_data = data_pipeline.load_data(str(csv_file))
            
            assert loaded_data is not None
            assert len(loaded_data) == len(sample_data)
            mock_load.assert_called_once()
    
    def test_data_preprocessing_step(self, data_pipeline, sample_data):
        """Test data preprocessing step"""
        with patch.object(data_pipeline.preprocessor, 'preprocess_pipeline') as mock_preprocess:
            mock_preprocess.return_value = (sample_data, {'status': 'success'})
            
            processed_data, report = data_pipeline.preprocess_data(sample_data)
            
            assert processed_data is not None
            assert isinstance(report, dict)
            mock_preprocess.assert_called_once()
    
    def test_feature_engineering_step(self, data_pipeline, sample_data):
        """Test feature engineering step"""
        with patch.object(data_pipeline.feature_engineer, 'create_all_features') as mock_features:
            mock_features.return_value = sample_data
            
            feature_data = data_pipeline.create_features(sample_data)
            
            assert feature_data is not None
            mock_features.assert_called_once()
    
    def test_complete_pipeline_execution(self, data_pipeline, sample_data, tmp_path):
        """Test complete pipeline execution"""
        # Create temporary CSV file
        csv_file = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_file, index=False)
        
        # Mock all dependencies
        with patch.object(data_pipeline.data_ingestion, 'load_from_csv') as mock_load, \
             patch.object(data_pipeline.preprocessor, 'preprocess_pipeline') as mock_preprocess, \
             patch.object(data_pipeline.feature_engineer, 'create_all_features') as mock_features:
            
            mock_load.return_value = sample_data
            mock_preprocess.return_value = (sample_data, {'status': 'success'})
            mock_features.return_value = sample_data
            
            final_data, pipeline_report = data_pipeline.run_complete_pipeline(
                data_source=str(csv_file)
            )
            
            assert final_data is not None
            assert isinstance(pipeline_report, dict)
            assert 'execution_time' in pipeline_report
            assert 'steps_completed' in pipeline_report
    
    def test_pipeline_error_handling(self, data_pipeline):
        """Test pipeline error handling"""
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            data_pipeline.load_data("non_existent_file.csv")
    
    def test_pipeline_with_database_source(self, data_pipeline, sample_data):
        """Test pipeline with database data source"""
        with patch.object(data_pipeline.data_ingestion, 'load_from_database') as mock_db_load:
            mock_db_load.return_value = sample_data
            
            loaded_data = data_pipeline.load_from_database("SELECT * FROM customers")
            
            assert loaded_data is not None
            mock_db_load.assert_called_once()
    
    def test_data_validation_in_pipeline(self, data_pipeline, sample_data):
        """Test data validation within pipeline"""
        # Add data quality issues
        corrupted_data = sample_data.copy()
        corrupted_data.loc[0:5, 'Age'] = -1  # Invalid ages
        
        with patch.object(data_pipeline.preprocessor, 'validate_data_quality') as mock_validate:
            mock_validate.return_value = {
                'is_valid': False,
                'errors': ['Invalid age values']
            }
            
            validation_result = data_pipeline.validate_data(corrupted_data)
            
            assert 'is_valid' in validation_result
            assert 'errors' in validation_result
    
    def test_pipeline_configuration_loading(self, data_pipeline):
        """Test pipeline configuration loading"""
        assert data_pipeline.config is not None
        assert 'data_sources' in data_pipeline.config
        assert 'preprocessing' in data_pipeline.config
    
    def test_save_processed_data(self, data_pipeline, sample_data, tmp_path):
        """Test saving processed data"""
        output_file = tmp_path / "processed_data.csv"
        
        success = data_pipeline.save_processed_data(sample_data, str(output_file))
        
        assert success
        assert output_file.exists()
        
        # Verify saved data
        saved_data = pd.read_csv(output_file)
        assert len(saved_data) == len(sample_data)
    
    def test_pipeline_performance_monitoring(self, data_pipeline, sample_data):
        """Test pipeline performance monitoring"""
        with patch('time.time') as mock_time:
            mock_time.side_effect = [0, 10, 20, 30]  # Simulate time progression
            
            final_data, report = data_pipeline.run_complete_pipeline(
                data_source=sample_data  # Pass DataFrame directly
            )
            
            assert 'execution_time' in report
            assert report['execution_time'] > 0
    
    def test_pipeline_with_different_data_sources(self, data_pipeline, sample_data):
        """Test pipeline with different data source types"""
        # Test with DataFrame
        result1 = data_pipeline.process_dataframe(sample_data)
        assert result1 is not None
        
        # Test with dictionary
        data_dict = sample_data.to_dict('records')
        with patch.object(data_pipeline, 'process_dictionary') as mock_process:
            mock_process.return_value = sample_data
            result2 = data_pipeline.process_dictionary(data_dict)
            assert result2 is not None
    
    def test_pipeline_memory_management(self, data_pipeline):
        """Test pipeline memory management with large datasets"""
        # Create larger dataset
        large_data = create_sample_customer_data(1000)
        
        with patch.object(data_pipeline.preprocessor, 'preprocess_pipeline') as mock_preprocess, \
             patch.object(data_pipeline.feature_engineer, 'create_all_features') as mock_features:
            
            mock_preprocess.return_value = (large_data, {'status': 'success'})
            mock_features.return_value = large_data
            
            final_data, report = data_pipeline.run_complete_pipeline(
                data_source=large_data
            )
            
            assert final_data is not None
            assert len(final_data) == len(large_data)
    
    def test_pipeline_step_dependencies(self, data_pipeline, sample_data):
        """Test pipeline step dependencies and execution order"""
        execution_order = []
        
        def track_preprocessing(*args, **kwargs):
            execution_order.append('preprocessing')
            return sample_data, {'status': 'success'}
        
        def track_feature_engineering(*args, **kwargs):
            execution_order.append('feature_engineering')
            return sample_data
        
        with patch.object(data_pipeline.preprocessor, 'preprocess_pipeline', side_effect=track_preprocessing), \
             patch.object(data_pipeline.feature_engineer, 'create_all_features', side_effect=track_feature_engineering):
            
            data_pipeline.run_complete_pipeline(data_source=sample_data)
            
            # Verify correct execution order
            assert execution_order == ['preprocessing', 'feature_engineering']
    
    def test_pipeline_rollback_mechanism(self, data_pipeline, sample_data):
        """Test pipeline rollback on failure"""
        with patch.object(data_pipeline.feature_engineer, 'create_all_features') as mock_features:
            mock_features.side_effect = Exception("Feature engineering failed")
            
            with pytest.raises(Exception):
                data_pipeline.run_complete_pipeline(data_source=sample_data)
    
    def test_pipeline_caching(self, data_pipeline, sample_data):
        """Test pipeline caching mechanism"""
        # First run
        with patch.object(data_pipeline.preprocessor, 'preprocess_pipeline') as mock_preprocess:
            mock_preprocess.return_value = (sample_data, {'status': 'success'})
            
            result1, report1 = data_pipeline.run_complete_pipeline(
                data_source=sample_data,
                use_cache=True
            )
            
            # Second run (should use cache)
            result2, report2 = data_pipeline.run_complete_pipeline(
                data_source=sample_data,
                use_cache=True
            )
            
            assert result1 is not None
            assert result2 is not None
    
    def test_pipeline_parallel_processing(self, data_pipeline, sample_data):
        """Test pipeline parallel processing capabilities"""
        with patch.object(data_pipeline, 'enable_parallel_processing') as mock_parallel:
            mock_parallel.return_value = True
            
            final_data, report = data_pipeline.run_complete_pipeline(
                data_source=sample_data,
                parallel=True
            )
            
            assert final_data is not None
            assert 'parallel_processing' in report
    
    @pytest.mark.parametrize("data_format", ["csv", "json", "parquet"])
    def test_pipeline_with_different_formats(self, data_pipeline, sample_data, tmp_path, data_format):
        """Test pipeline with different data formats"""
        if data_format == "csv":
            file_path = tmp_path / "test.csv"
            sample_data.to_csv(file_path, index=False)
        elif data_format == "json":
            file_path = tmp_path / "test.json"
            sample_data.to_json(file_path, orient='records')
        elif data_format == "parquet":
            pytest.skip("Parquet format requires pyarrow")
        
        with patch.object(data_pipeline.data_ingestion, f'load_from_{data_format}') as mock_load:
            mock_load.return_value = sample_data
            
            loaded_data = data_pipeline.load_data(str(file_path))
            assert loaded_data is not None
    
    def test_pipeline_data_lineage(self, data_pipeline, sample_data):
        """Test pipeline data lineage tracking"""
        final_data, report = data_pipeline.run_complete_pipeline(
            data_source=sample_data,
            track_lineage=True
        )
        
        assert 'data_lineage' in report
        assert 'transformations' in report['data_lineage']
    
    def test_pipeline_incremental_processing(self, data_pipeline, sample_data):
        """Test pipeline incremental processing"""
        # Process initial data
        result1 = data_pipeline.run_complete_pipeline(data_source=sample_data)
        
        # Process incremental data
        incremental_data = create_sample_customer_data(50)
        result2 = data_pipeline.run_incremental_pipeline(
            new_data=incremental_data,
            existing_data=result1[0]
        )
        
        assert result2 is not None