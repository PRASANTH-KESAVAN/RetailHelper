# src/pipelines/training_pipeline.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import yaml
from pathlib import Path
from loguru import logger
import joblib
import os

from components.data_preprocessing import DataPreprocessor
from components.feature_engineering import FeatureEngineering
from components.customer_segmentation import CustomerSegmentation
from components.predictive_modeling import PredictiveModeling
from components.recommendation_engine import RecommendationEngine
from components.model_evaluation import ModelEvaluation
from utils.common import ensure_directory_exists

class TrainingPipeline:
    """
    Training pipeline for all machine learning models
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the training pipeline"""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.preprocessor = DataPreprocessor(config_path)
        self.feature_engineer = FeatureEngineering(self.config)
        self.segmentation = CustomerSegmentation(self.config)
        self.predictive_modeling = PredictiveModeling(self.config)
        self.recommendation_engine = RecommendationEngine(self.config)
        self.model_evaluator = ModelEvaluation(self.config)
        
        self.trained_models = {}
        self.training_results = {}
        
        logger.info("Training Pipeline initialized successfully")
    
    def _load_config(self) -> Dict:
        """Load configuration file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def train_segmentation_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train customer segmentation models
        
        Args:
            df: Input dataframe with customer features
            
        Returns:
            Training results for segmentation models
        """
        logger.info("=== TRAINING CUSTOMER SEGMENTATION MODELS ===")
        
        try:
            segmentation_results = {}
            
            # Prepare features for clustering
            clustering_features = self.segmentation.select_features_for_clustering(df)
            
            # Find optimal number of clusters
            optimal_analysis = self.segmentation.find_optimal_clusters(clustering_features, max_clusters=10)
            segmentation_results['optimal_clusters_analysis'] = optimal_analysis
            
            # Train K-means with different cluster numbers
            cluster_options = [optimal_analysis['elbow_point'], optimal_analysis['best_silhouette_k']]
            cluster_options = list(set([c for c in cluster_options if c is not None]))
            
            if not cluster_options:
                cluster_options = [5]  # Default fallback
            
            best_score = -1
            best_model_config = None
            
            for n_clusters in cluster_options:
                try:
                    # Train K-means model
                    cluster_labels = self.segmentation.perform_kmeans_clustering(
                        clustering_features, n_clusters=n_clusters
                    )
                    
                    # Analyze segments
                    segment_analysis = self.segmentation.analyze_segments(df, cluster_labels)
                    segment_profiles = self.segmentation.create_segment_profiles(segment_analysis)
                    
                    # Calculate silhouette score as quality metric
                    from sklearn.metrics import silhouette_score
                    score = silhouette_score(clustering_features, cluster_labels)
                    
                    config_key = f"kmeans_{n_clusters}_clusters"
                    segmentation_results[config_key] = {
                        'model': self.segmentation.model,
                        'cluster_labels': cluster_labels,
                        'segment_analysis': segment_analysis,
                        'segment_profiles': segment_profiles,
                        'silhouette_score': score,
                        'n_clusters': n_clusters,
                        'feature_names': self.segmentation.feature_names
                    }
                    
                    if score > best_score:
                        best_score = score
                        best_model_config = config_key
                    
                    logger.info(f"K-means with {n_clusters} clusters trained. Silhouette Score: {score:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training K-means with {n_clusters} clusters: {e}")
                    continue
            
            if best_model_config:
                segmentation_results['best_model'] = best_model_config
                segmentation_results['best_score'] = best_score
                logger.info(f"Best segmentation model: {best_model_config} (Score: {best_score:.4f})")
            
            self.trained_models['segmentation'] = segmentation_results
            return segmentation_results
            
        except Exception as e:
            logger.error(f"Error training segmentation models: {e}")
            raise
    
    def train_predictive_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train predictive models (churn, CLV, purchase prediction)
        
        Args:
            df: Input dataframe with customer features
            
        Returns:
            Training results for predictive models
        """
        logger.info("=== TRAINING PREDICTIVE MODELS ===")
        
        try:
            predictive_results = {}
            
            # 1. Train Churn Prediction Model
            logger.info("Training churn prediction model...")
            try:
                X_churn, y_churn = self.predictive_modeling.prepare_churn_prediction_data(df)
                
                # Train multiple models and compare
                churn_comparison = self.predictive_modeling.compare_models(
                    X_churn, y_churn, 
                    model_types=['random_forest', 'logistic_regression', 'xgboost'],
                    task_type='classification'
                )
                
                # Get best model
                best_churn_model = churn_comparison.iloc[0]['model_type']
                
                # Train final model with hyperparameter tuning
                churn_results = self.predictive_modeling.hyperparameter_tuning(
                    X_churn, y_churn, 
                    model_type=best_churn_model,
                    task_type='classification'
                )
                
                # Evaluate model
                y_pred_churn = churn_results['best_model'].predict(
                    churn_results['scaler'].transform(X_churn)
                )
                churn_evaluation = self.model_evaluator.evaluate_classification_model(
                    y_churn.values, y_pred_churn, model_name="churn_prediction"
                )
                
                predictive_results['churn_prediction'] = {
                    'model': churn_results['best_model'],
                    'scaler': churn_results['scaler'],
                    'best_params': churn_results['best_params'],
                    'features': X_churn.columns.tolist(),
                    'model_comparison': churn_comparison,
                    'evaluation_metrics': churn_evaluation,
                    'training_score': churn_results['best_score']
                }
                
                logger.info(f"Churn prediction model trained successfully. Best model: {best_churn_model}")
                
            except Exception as e:
                logger.error(f"Error training churn prediction model: {e}")
                predictive_results['churn_prediction'] = {'error': str(e)}
            
            # 2. Train Customer Lifetime Value (CLV) Prediction Model
            logger.info("Training CLV prediction model...")
            try:
                X_clv, y_clv = self.predictive_modeling.prepare_clv_prediction_data(df)
                
                # Train multiple models and compare
                clv_comparison = self.predictive_modeling.compare_models(
                    X_clv, y_clv,
                    model_types=['random_forest', 'linear_regression', 'xgboost'],
                    task_type='regression'
                )
                
                # Get best model
                best_clv_model = clv_comparison.iloc[0]['model_type']
                
                # Train final model
                clv_results = self.predictive_modeling.hyperparameter_tuning(
                    X_clv, y_clv,
                    model_type=best_clv_model,
                    task_type='regression'
                )
                
                # Evaluate model
                y_pred_clv = clv_results['best_model'].predict(
                    clv_results['scaler'].transform(X_clv)
                )
                clv_evaluation = self.model_evaluator.evaluate_regression_model(
                    y_clv.values, y_pred_clv, model_name="clv_prediction"
                )
                
                predictive_results['clv_prediction'] = {
                    'model': clv_results['best_model'],
                    'scaler': clv_results['scaler'],
                    'best_params': clv_results['best_params'],
                    'features': X_clv.columns.tolist(),
                    'model_comparison': clv_comparison,
                    'evaluation_metrics': clv_evaluation,
                    'training_score': clv_results['best_score']
                }
                
                logger.info(f"CLV prediction model trained successfully. Best model: {best_clv_model}")
                
            except Exception as e:
                logger.error(f"Error training CLV prediction model: {e}")
                predictive_results['clv_prediction'] = {'error': str(e)}
            
            # 3. Train Purchase Prediction Model
            logger.info("Training purchase prediction model...")
            try:
                X_purchase, y_purchase = self.predictive_modeling.prepare_purchase_prediction_data(df)
                
                # Train multiple models and compare
                purchase_comparison = self.predictive_modeling.compare_models(
                    X_purchase, y_purchase,
                    model_types=['random_forest', 'logistic_regression', 'gradient_boosting'],
                    task_type='classification'
                )
                
                # Get best model
                best_purchase_model = purchase_comparison.iloc[0]['model_type']
                
                # Train final model
                purchase_results = self.predictive_modeling.train_model(
                    X_purchase, y_purchase,
                    model_type=best_purchase_model,
                    task_type='classification',
                    cross_validation=True
                )
                
                predictive_results['purchase_prediction'] = {
                    'model': purchase_results['model'],
                    'scaler': purchase_results['scaler'],
                    'features': X_purchase.columns.tolist(),
                    'model_comparison': purchase_comparison,
                    'evaluation_metrics': purchase_results['metrics'],
                    'cv_scores': purchase_results['cv_scores']
                }
                
                logger.info(f"Purchase prediction model trained successfully. Best model: {best_purchase_model}")
                
            except Exception as e:
                logger.error(f"Error training purchase prediction model: {e}")
                predictive_results['purchase_prediction'] = {'error': str(e)}
            
            self.trained_models['predictive'] = predictive_results
            return predictive_results
            
        except Exception as e:
            logger.error(f"Error training predictive models: {e}")
            raise
    
    def train_recommendation_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train recommendation system models
        
        Args:
            df: Input dataframe
            
        Returns:
            Training results for recommendation models
        """
        logger.info("=== TRAINING RECOMMENDATION MODELS ===")
        
        try:
            recommendation_results = {}
            
            # Prepare user-item interaction matrix
            user_item_matrix = self.recommendation_engine.prepare_user_item_matrix(
                df, user_col='Customer ID', item_col='Category', rating_col='Review Rating'
            )
            
            # Train collaborative filtering models
            logger.info("Training collaborative filtering models...")
            
            # Get sample users for evaluation
            sample_users = list(self.recommendation_engine.user_mapping.keys())[:50]
            
            # Evaluate collaborative filtering approaches
            cf_evaluation_results = {}
            
            try:
                # User-based collaborative filtering
                user_cf_metrics = self.recommendation_engine.evaluate_recommendations(
                    df, test_users=sample_users[:10], 
                    recommendation_method='collaborative_filtering'
                )
                cf_evaluation_results['user_based'] = user_cf_metrics
                
            except Exception as e:
                logger.warning(f"Error evaluating user-based CF: {e}")
                cf_evaluation_results['user_based'] = {'error': str(e)}
            
            try:
                # Content-based filtering
                content_metrics = self.recommendation_engine.evaluate_recommendations(
                    df, test_users=sample_users[:10],
                    recommendation_method='content_based'
                )
                cf_evaluation_results['content_based'] = content_metrics
                
            except Exception as e:
                logger.warning(f"Error evaluating content-based filtering: {e}")
                cf_evaluation_results['content_based'] = {'error': str(e)}
            
            try:
                # Hybrid approach
                hybrid_metrics = self.recommendation_engine.evaluate_recommendations(
                    df, test_users=sample_users[:10],
                    recommendation_method='hybrid'
                )
                cf_evaluation_results['hybrid'] = hybrid_metrics
                
            except Exception as e:
                logger.warning(f"Error evaluating hybrid approach: {e}")
                cf_evaluation_results['hybrid'] = {'error': str(e)}
            
            recommendation_results['collaborative_filtering'] = {
                'user_item_matrix_shape': user_item_matrix.shape,
                'n_users': len(self.recommendation_engine.user_mapping),
                'n_items': len(self.recommendation_engine.item_mapping),
                'evaluation_results': cf_evaluation_results,
                'user_mapping': self.recommendation_engine.user_mapping,
                'item_mapping': self.recommendation_engine.item_mapping
            }
            
            # Train matrix factorization model
            logger.info("Training matrix factorization model...")
            try:
                # Test matrix factorization with different components
                mf_results = {}
                for n_components in [10, 25, 50]:
                    try:
                        # Train on a sample user to test
                        test_user = sample_users[0]
                        mf_recommendations = self.recommendation_engine.matrix_factorization_recommendations(
                            test_user, n_recommendations=5, n_components=n_components
                        )
                        mf_results[f'n_components_{n_components}'] = {
                            'n_recommendations': len(mf_recommendations),
                            'sample_recommendations': mf_recommendations[:3]
                        }
                    except Exception as e:
                        logger.warning(f"Error testing matrix factorization with {n_components} components: {e}")
                        continue
                
                recommendation_results['matrix_factorization'] = mf_results
                
            except Exception as e:
                logger.error(f"Error training matrix factorization: {e}")
                recommendation_results['matrix_factorization'] = {'error': str(e)}
            
            self.trained_models['recommendation'] = recommendation_results
            logger.info("Recommendation models training completed")
            return recommendation_results
            
        except Exception as e:
            logger.error(f"Error training recommendation models: {e}")
            raise
    
    def run_complete_training_pipeline(self, df: pd.DataFrame, 
                                     output_dir: str = None,
                                     save_models: bool = True) -> Dict[str, Any]:
        """
        Run the complete training pipeline
        
        Args:
            df: Input dataframe with customer data
            output_dir: Directory to save trained models
            save_models: Whether to save trained models to disk
            
        Returns:
            Complete training results
        """
        logger.info("🚀 STARTING COMPLETE TRAINING PIPELINE")
        training_start_time = pd.Timestamp.now()
        
        try:
            complete_results = {
                'training_start_time': training_start_time,
                'input_data_shape': df.shape
            }
            
            # Step 1: Train Segmentation Models
            segmentation_results = self.train_segmentation_models(df)
            complete_results['segmentation'] = segmentation_results
            
            # Step 2: Train Predictive Models
            predictive_results = self.train_predictive_models(df)
            complete_results['predictive'] = predictive_results
            
            # Step 3: Train Recommendation Models
            recommendation_results = self.train_recommendation_models(df)
            complete_results['recommendation'] = recommendation_results
            
            # Calculate training summary
            training_end_time = pd.Timestamp.now()
            complete_results['training_summary'] = {
                'end_time': training_end_time,
                'total_duration': str(training_end_time - training_start_time),
                'models_trained': {
                    'segmentation_models': len([k for k in segmentation_results.keys() if 'kmeans' in k]),
                    'predictive_models': len([k for k, v in predictive_results.items() if 'error' not in v]),
                    'recommendation_models': len(recommendation_results.keys())
                },
                'training_success': True
            }
            
            # Step 4: Save Models (if requested)
            if save_models:
                self.save_trained_models(complete_results, output_dir)
            
            # Store results
            self.training_results = complete_results
            
            logger.info("✅ COMPLETE TRAINING PIPELINE FINISHED SUCCESSFULLY")
            logger.info(f"Total training time: {training_end_time - training_start_time}")
            
            return complete_results
            
        except Exception as e:
            logger.error(f"❌ TRAINING PIPELINE FAILED: {e}")
            complete_results['training_summary'] = {
                'end_time': pd.Timestamp.now(),
                'training_success': False,
                'error': str(e)
            }
            raise
    
    def save_trained_models(self, training_results: Dict[str, Any], 
                          output_dir: str = None):
        """
        Save trained models to disk
        
        Args:
            training_results: Complete training results
            output_dir: Directory to save models
        """
        logger.info("Saving trained models...")
        
        try:
            if output_dir is None:
                output_dir = Path(self.config['paths']['models'])
            else:
                output_dir = Path(output_dir)
            
            # Create directories
            ensure_directory_exists(output_dir)
            ensure_directory_exists(output_dir / "segmentation")
            ensure_directory_exists(output_dir / "predictive")
            ensure_directory_exists(output_dir / "recommendation")
            
            # Save segmentation models
            if 'segmentation' in training_results:
                seg_results = training_results['segmentation']
                if 'best_model' in seg_results:
                    best_seg_model = seg_results[seg_results['best_model']]
                    seg_model_path = output_dir / "segmentation" / "best_segmentation_model.pkl"
                    
                    model_data = {
                        'model': best_seg_model['model'],
                        'segment_analysis': best_seg_model['segment_analysis'],
                        'segment_profiles': best_seg_model['segment_profiles'],
                        'feature_names': best_seg_model['feature_names'],
                        'n_clusters': best_seg_model['n_clusters'],
                        'silhouette_score': best_seg_model['silhouette_score']
                    }
                    
                    joblib.dump(model_data, seg_model_path)
                    logger.info(f"Segmentation model saved to {seg_model_path}")
            
            # Save predictive models
            if 'predictive' in training_results:
                pred_results = training_results['predictive']
                
                for model_name, model_data in pred_results.items():
                    if 'error' not in model_data and 'model' in model_data:
                        model_path = output_dir / "predictive" / f"{model_name}_model.pkl"
                        joblib.dump(model_data, model_path)
                        logger.info(f"{model_name} model saved to {model_path}")
            
            # Save recommendation models
            if 'recommendation' in training_results:
                rec_results = training_results['recommendation']
                rec_model_path = output_dir / "recommendation" / "recommendation_system.pkl"
                
                # Save recommendation system data
                rec_data = {
                    'user_item_matrix': self.recommendation_engine.user_item_matrix,
                    'user_mapping': self.recommendation_engine.user_mapping,
                    'item_mapping': self.recommendation_engine.item_mapping,
                    'reverse_user_mapping': self.recommendation_engine.reverse_user_mapping,
                    'reverse_item_mapping': self.recommendation_engine.reverse_item_mapping,
                    'models': self.recommendation_engine.models,
                    'similarity_matrices': self.recommendation_engine.similarity_matrices,
                    'evaluation_results': rec_results
                }
                
                joblib.dump(rec_data, rec_model_path)
                logger.info(f"Recommendation system saved to {rec_model_path}")
            
            # Save training metadata
            metadata_path = output_dir / "training_metadata.pkl"
            metadata = {
                'training_results': training_results,
                'config': self.config,
                'training_timestamp': pd.Timestamp.now()
            }
            joblib.dump(metadata, metadata_path)
            
            logger.info("All trained models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving trained models: {e}")
            raise
    
    def load_trained_models(self, models_dir: str):
        """
        Load previously trained models
        
        Args:
            models_dir: Directory containing saved models
        """
        logger.info(f"Loading trained models from {models_dir}")
        
        try:
            models_path = Path(models_dir)
            
            # Load segmentation model
            seg_model_path = models_path / "segmentation" / "best_segmentation_model.pkl"
            if seg_model_path.exists():
                seg_data = joblib.load(seg_model_path)
                self.trained_models['segmentation'] = seg_data
                logger.info("Segmentation model loaded")
            
            # Load predictive models
            pred_dir = models_path / "predictive"
            if pred_dir.exists():
                pred_models = {}
                for model_file in pred_dir.glob("*.pkl"):
                    model_name = model_file.stem.replace('_model', '')
                    pred_models[model_name] = joblib.load(model_file)
                    logger.info(f"Predictive model loaded: {model_name}")
                self.trained_models['predictive'] = pred_models
            
            # Load recommendation system
            rec_model_path = models_path / "recommendation" / "recommendation_system.pkl"
            if rec_model_path.exists():
                rec_data = joblib.load(rec_model_path)
                
                # Restore recommendation engine state
                self.recommendation_engine.user_item_matrix = rec_data.get('user_item_matrix')
                self.recommendation_engine.user_mapping = rec_data.get('user_mapping', {})
                self.recommendation_engine.item_mapping = rec_data.get('item_mapping', {})
                self.recommendation_engine.reverse_user_mapping = rec_data.get('reverse_user_mapping', {})
                self.recommendation_engine.reverse_item_mapping = rec_data.get('reverse_item_mapping', {})
                self.recommendation_engine.models = rec_data.get('models', {})
                self.recommendation_engine.similarity_matrices = rec_data.get('similarity_matrices', {})
                
                self.trained_models['recommendation'] = rec_data
                logger.info("Recommendation system loaded")
            
            logger.info("All trained models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading trained models: {e}")
            raise
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training results"""
        return self.training_results.get('training_summary', {})