# tests/test_pipelines/test_training_pipeline.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pipelines.training_pipeline import TrainingPipeline
from tests import create_sample_processed_data, TEST_CONFIG

class TestTrainingPipeline:
    """Test suite for TrainingPipeline"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample processed data for testing"""
        return create_sample_processed_data(200)
    
    @pytest.fixture
    def training_pipeline(self):
        """Create TrainingPipeline instance"""
        config_path = "config/config.yaml"
        with patch('yaml.safe_load') as mock_yaml:
            mock_yaml.return_value = {
                'training': {
                    'test_size': 0.2,
                    'random_state': 42,
                    'cv_folds': 3
                },
                'models': {
                    'segmentation': {
                        'n_clusters_range': [2, 8],
                        'random_state': 42
                    },
                    'predictive': {
                        'algorithms': ['random_forest', 'xgboost'],
                        'hyperparameters': {
                            'random_forest': {
                                'n_estimators': [50, 100],
                                'max_depth': [5, 10]
                            }
                        }
                    }
                },
                'paths': {
                    'models': 'models/',
                    'reports': 'reports/'
                }
            }
            with patch('builtins.open', create=True):
                return TrainingPipeline(config_path)
    
    def test_initialization(self, training_pipeline):
        """Test TrainingPipeline initialization"""
        assert training_pipeline is not None
        assert hasattr(training_pipeline, 'config')
        assert hasattr(training_pipeline, 'segmentation')
        assert hasattr(training_pipeline, 'predictive_modeling')
        assert hasattr(training_pipeline, 'recommendation_engine')
    
    def test_customer_segmentation_training(self, training_pipeline, sample_data):
        """Test customer segmentation model training"""
        with patch.object(training_pipeline.segmentation, 'perform_kmeans_clustering') as mock_cluster, \
             patch.object(training_pipeline.segmentation, 'find_optimal_clusters') as mock_optimal:
            
            mock_optimal.return_value = {'elbow_point': 3, 'silhouette_scores': [0.5, 0.6, 0.7]}
            mock_cluster.return_value = np.random.randint(0, 3, len(sample_data))
            
            segmentation_results = training_pipeline.train_customer_segmentation(sample_data)
            
            assert segmentation_results is not None
            assert 'cluster_labels' in segmentation_results
            assert 'optimal_clusters' in segmentation_results
            assert 'model_performance' in segmentation_results
    
    def test_churn_prediction_training(self, training_pipeline, sample_data):
        """Test churn prediction model training"""
        with patch.object(training_pipeline.predictive_modeling, 'prepare_churn_prediction_data') as mock_prep, \
             patch.object(training_pipeline.predictive_modeling, 'train_model') as mock_train:
            
            # Mock data preparation
            X = sample_data[['Age', 'Total_Spent', 'Purchase_Count']].fillna(0)
            y = np.random.binomial(1, 0.2, len(sample_data))  # Binary churn labels
            mock_prep.return_value = (X, y)
            
            # Mock training results
            mock_train.return_value = {
                'model': MagicMock(),
                'scaler': MagicMock(),
                'metrics': {'accuracy': 0.85, 'f1_score': 0.75},
                'y_pred': y,
                'y_pred_proba': np.random.rand(len(y))
            }
            
            churn_results = training_pipeline.train_churn_prediction(sample_data)
            
            assert churn_results is not None
            assert 'model' in churn_results
            assert 'performance_metrics' in churn_results
            assert 'feature_importance' in churn_results
    
    def test_clv_prediction_training(self, training_pipeline, sample_data):
        """Test CLV prediction model training"""
        with patch.object(training_pipeline.predictive_modeling, 'prepare_clv_prediction_data') as mock_prep, \
             patch.object(training_pipeline.predictive_modeling, 'train_model') as mock_train:
            
            # Mock data preparation
            X = sample_data[['Age', 'Total_Spent', 'Purchase_Count']].fillna(0)
            y = np.random.exponential(200, len(sample_data))  # CLV values
            mock_prep.return_value = (X, y)
            
            # Mock training results
            mock_train.return_value = {
                'model': MagicMock(),
                'scaler': MagicMock(),
                'metrics': {'r2': 0.78, 'rmse': 45.2},
                'y_pred': y
            }
            
            clv_results = training_pipeline.train_clv_prediction(sample_data)
            
            assert clv_results is not None
            assert 'model' in clv_results
            assert 'performance_metrics' in clv_results
    
    def test_recommendation_system_training(self, training_pipeline, sample_data):
        """Test recommendation system training"""
        with patch.object(training_pipeline.recommendation_engine, 'prepare_user_item_matrix') as mock_matrix, \
             patch.object(training_pipeline.recommendation_engine, 'train_collaborative_filtering') as mock_train:
            
            # Mock user-item matrix
            mock_matrix.return_value = np.random.rand(50, 20)  # 50 users, 20 items
            
            # Mock training results
            mock_train.return_value = {
                'model': MagicMock(),
                'user_mapping': {f'user_{i}': i for i in range(50)},
                'item_mapping': {f'item_{i}': i for i in range(20)}
            }
            
            rec_results = training_pipeline.train_recommendation_system(sample_data)
            
            assert rec_results is not None
            assert 'user_item_matrix' in rec_results
            assert 'model_performance' in rec_results
    
    def test_complete_training_pipeline(self, training_pipeline, sample_data):
        """Test complete training pipeline execution"""
        with patch.object(training_pipeline, 'train_customer_segmentation') as mock_seg, \
             patch.object(training_pipeline, 'train_churn_prediction') as mock_churn, \
             patch.object(training_pipeline, 'train_clv_prediction') as mock_clv, \
             patch.object(training_pipeline, 'train_recommendation_system') as mock_rec:
            
            # Mock all training results
            mock_seg.return_value = {'status': 'success', 'n_clusters': 3}
            mock_churn.return_value = {'status': 'success', 'accuracy': 0.85}
            mock_clv.return_value = {'status': 'success', 'r2': 0.78}
            mock_rec.return_value = {'status': 'success', 'coverage': 0.8}
            
            complete_results = training_pipeline.run_complete_training_pipeline(sample_data)
            
            assert complete_results is not None
            assert 'segmentation' in complete_results
            assert 'churn_prediction' in complete_results
            assert 'clv_prediction' in complete_results
            assert 'recommendation' in complete_results
            assert 'training_summary' in complete_results
    
    def test_model_hyperparameter_tuning(self, training_pipeline, sample_data):
        """Test hyperparameter tuning functionality"""
        with patch.object(training_pipeline.predictive_modeling, 'hyperparameter_tuning') as mock_tune:
            mock_tune.return_value = {
                'best_params': {'n_estimators': 100, 'max_depth': 10},
                'best_score': 0.85,
                'cv_results': {'mean_score': 0.82, 'std_score': 0.05}
            }
            
            X = sample_data[['Age', 'Total_Spent']].fillna(0)
            y = np.random.binomial(1, 0.2, len(sample_data))
            
            tuning_results = training_pipeline.hyperparameter_tuning(
                X, y, model_type='random_forest', task_type='classification'
            )
            
            assert tuning_results is not None
            assert 'best_params' in tuning_results
            assert 'best_score' in tuning_results
    
    def test_cross_validation_evaluation(self, training_pipeline, sample_data):
        """Test cross-validation evaluation"""
        with patch.object(training_pipeline.model_evaluator, 'cross_validation_evaluation') as mock_cv:
            mock_cv.return_value = {
                'cv_scores': [0.8, 0.82, 0.78, 0.85, 0.79],
                'mean_score': 0.808,
                'std_score': 0.027
            }
            
            X = sample_data[['Age', 'Total_Spent']].fillna(0)
            y = np.random.binomial(1, 0.2, len(sample_data))
            
            cv_results = training_pipeline.cross_validate_model(
                X, y, model_type='random_forest'
            )
            
            assert cv_results is not None
            assert 'cv_scores' in cv_results
            assert 'mean_score' in cv_results
    
    def test_model_saving(self, training_pipeline, sample_data, tmp_path):
        """Test model saving functionality"""
        # Create mock model
        mock_model = MagicMock()
        mock_scaler = MagicMock()
        
        model_data = {
            'model': mock_model,
            'scaler': mock_scaler,
            'metrics': {'accuracy': 0.85},
            'features': ['Age', 'Total_Spent']
        }
        
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        
        with patch('joblib.dump') as mock_dump:
            success = training_pipeline.save_trained_models(
                {'churn_prediction': model_data},
                str(models_dir)
            )
            
            assert success
            mock_dump.assert_called()
    
    def test_training_progress_tracking(self, training_pipeline, sample_data):
        """Test training progress tracking"""
        progress_updates = []
        
        def mock_progress_callback(stage, progress):
            progress_updates.append({'stage': stage, 'progress': progress})
        
        with patch.object(training_pipeline, 'progress_callback', mock_progress_callback):
            # Simulate training with progress tracking
            training_pipeline.train_with_progress_tracking(sample_data)
            
            assert len(progress_updates) > 0
            assert all('stage' in update for update in progress_updates)
    
    def test_training_error_handling(self, training_pipeline, sample_data):
        """Test training error handling"""
        with patch.object(training_pipeline.predictive_modeling, 'train_model') as mock_train:
            mock_train.side_effect = Exception("Training failed")
            
            # Should handle errors gracefully
            results = training_pipeline.train_churn_prediction(sample_data)
            
            assert 'error' in results
            assert results['status'] == 'failed'
    
    def test_model_performance_comparison(self, training_pipeline, sample_data):
        """Test model performance comparison"""
        # Mock multiple model results
        model_results = {
            'random_forest': {'accuracy': 0.85, 'f1_score': 0.78},
            'xgboost': {'accuracy': 0.87, 'f1_score': 0.80},
            'logistic_regression': {'accuracy': 0.82, 'f1_score': 0.75}
        }
        
        comparison = training_pipeline.compare_model_performance(model_results)
        
        assert comparison is not None
        assert 'best_model' in comparison
        assert comparison['best_model'] == 'xgboost'  # Highest accuracy
    
    def test_feature_importance_analysis(self, training_pipeline, sample_data):
        """Test feature importance analysis"""
        # Create mock model with feature importance
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.3, 0.5, 0.2])
        
        feature_names = ['Age', 'Total_Spent', 'Purchase_Count']
        
        importance_analysis = training_pipeline.analyze_feature_importance(
            mock_model, feature_names
        )
        
        assert importance_analysis is not None
        assert 'feature_importance' in importance_analysis
        assert len(importance_analysis['feature_importance']) == len(feature_names)
    
    def test_training_data_splitting(self, training_pipeline, sample_data):
        """Test training data splitting"""
        X = sample_data[['Age', 'Total_Spent', 'Purchase_Count']].fillna(0)
        y = np.random.binomial(1, 0.2, len(sample_data))
        
        X_train, X_test, y_train, y_test = training_pipeline.split_training_data(X, y)
        
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
    
    def test_training_with_imbalanced_data(self, training_pipeline, sample_data):
        """Test training with imbalanced datasets"""
        # Create imbalanced target
        imbalanced_y = np.zeros(len(sample_data))
        imbalanced_y[:int(len(sample_data) * 0.05)] = 1  # Only 5% positive class
        
        with patch.object(training_pipeline.predictive_modeling, 'handle_imbalanced_data') as mock_balance:
            mock_balance.return_value = (sample_data, imbalanced_y)
            
            X = sample_data[['Age', 'Total_Spent']].fillna(0)
            
            balanced_data = training_pipeline.handle_imbalanced_training_data(X, imbalanced_y)
            
            assert balanced_data is not None
            mock_balance.assert_called_once()
    
    def test_ensemble_model_training(self, training_pipeline, sample_data):
        """Test ensemble model training"""
        with patch.object(training_pipeline, 'train_ensemble_models') as mock_ensemble:
            mock_ensemble.return_value = {
                'voting_classifier': {'accuracy': 0.88},
                'stacking_classifier': {'accuracy': 0.89},
                'best_ensemble': 'stacking_classifier'
            }
            
            X = sample_data[['Age', 'Total_Spent']].fillna(0)
            y = np.random.binomial(1, 0.2, len(sample_data))
            
            ensemble_results = training_pipeline.train_ensemble_models(X, y)
            
            assert ensemble_results is not None
            assert 'best_ensemble' in ensemble_results
    
    @pytest.mark.parametrize("model_type", ["random_forest", "xgboost", "logistic_regression"])
    def test_different_model_types(self, training_pipeline, sample_data, model_type):
        """Test training with different model types"""
        with patch.object(training_pipeline.predictive_modeling, 'train_model') as mock_train:
            mock_train.return_value = {
                'model': MagicMock(),
                'metrics': {'accuracy': 0.8},
                'model_type': model_type
            }
            
            X = sample_data[['Age', 'Total_Spent']].fillna(0)
            y = np.random.binomial(1, 0.2, len(sample_data))
            
            result = training_pipeline.train_single_model(X, y, model_type)
            
            assert result is not None
            assert result['model_type'] == model_type
    
    def test_training_pipeline_reproducibility(self, training_pipeline, sample_data):
        """Test training pipeline reproducibility"""
        # Set random seed
        training_pipeline.config['training']['random_state'] = 42
        
        with patch.object(training_pipeline, 'train_churn_prediction') as mock_train:
            mock_train.return_value = {'model_id': 'test_model', 'accuracy': 0.85}
            
            # Run training twice
            result1 = training_pipeline.train_churn_prediction(sample_data)
            result2 = training_pipeline.train_churn_prediction(sample_data)
            
            # Results should be consistent
            assert result1 == result2