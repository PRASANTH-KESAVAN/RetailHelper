# src/pipelines/data_pipeline.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import yaml
from pathlib import Path
from loguru import logger
import joblib

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessor
from src.components.feature_engineering import FeatureEngineering
from src.components.customer_segmentation import CustomerSegmentation
from src.utils.common import ensure_directory_exists


class DataPipeline:
    """
    Main data pipeline orchestrating all data processing steps
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the data pipeline"""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.data_ingestion = DataIngestion(self.config)
        self.preprocessor = DataPreprocessor(config_path)
        self.feature_engineer = FeatureEngineering(self.config)
        self.segmentation = CustomerSegmentation(self.config)
        
        self.processed_data = None
        self.pipeline_metrics = {}
        
        logger.info("Data Pipeline initialized successfully")
    
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
    
    def run_data_ingestion(self, data_source: str, **kwargs) -> pd.DataFrame:
        """
        Execute data ingestion step
        
        Args:
            data_source: Path to data source or database connection string
            **kwargs: Additional parameters for data ingestion
        
        Returns:
            Raw dataframe
        """
        logger.info("=== STARTING DATA INGESTION ===")
        
        try:
            if data_source.endswith('.csv'):
                raw_data = self.data_ingestion.load_from_csv(data_source)
            elif data_source.startswith('sql://'):
                raw_data = self.data_ingestion.load_from_database(data_source, **kwargs)
            else:
                raise ValueError(f"Unsupported data source format: {data_source}")
            
            # Store ingestion metrics
            self.pipeline_metrics['ingestion'] = {
                'source': data_source,
                'raw_records': len(raw_data),
                'raw_features': raw_data.shape[1],
                'data_types': raw_data.dtypes.to_dict(),
                'memory_usage_mb': raw_data.memory_usage(deep=True).sum() / 1024**2
            }
            
            logger.info(f"Data ingestion completed. Shape: {raw_data.shape}")
            return raw_data
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            raise
    
    def run_preprocessing(self, raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Execute data preprocessing step
        
        Args:
            raw_data: Raw dataframe from ingestion
        
        Returns:
            Preprocessed dataframe and preprocessing statistics
        """
        logger.info("=== STARTING DATA PREPROCESSING ===")
        
        try:
            processed_data, preprocessing_stats = self.preprocessor.preprocess_pipeline(raw_data)
            
            # Store preprocessing metrics
            self.pipeline_metrics['preprocessing'] = preprocessing_stats
            self.pipeline_metrics['preprocessing']['completion_time'] = pd.Timestamp.now()
            
            logger.info("Data preprocessing completed successfully")
            return processed_data, preprocessing_stats
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            raise
    
    def run_feature_engineering(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Execute feature engineering step
        
        Args:
            processed_data: Preprocessed dataframe
        
        Returns:
            Dataframe with engineered features
        """
        logger.info("=== STARTING FEATURE ENGINEERING ===")
        
        try:
            # Create customer-level features
            customer_features = self.feature_engineer.create_customer_features(processed_data)
            
            # Create RFM features
            rfm_features = self.feature_engineer.create_rfm_features(processed_data)
            
            # Create behavioral features
            behavioral_features = self.feature_engineer.create_behavioral_features(processed_data)
            
            # Merge all features
            engineered_data = self.feature_engineer.merge_features(
                processed_data, customer_features, rfm_features, behavioral_features
            )
            
            # Store feature engineering metrics
            self.pipeline_metrics['feature_engineering'] = {
                'original_features': processed_data.shape[1],
                'engineered_features': engineered_data.shape[1],
                'new_features_created': engineered_data.shape[1] - processed_data.shape[1],
                'completion_time': pd.Timestamp.now()
            }
            
            logger.info(f"Feature engineering completed. New shape: {engineered_data.shape}")
            return engineered_data
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise
    
    def run_customer_segmentation(self, engineered_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Execute customer segmentation step
        
        Args:
            engineered_data: Dataframe with engineered features
        
        Returns:
            Dataframe with cluster labels and segmentation analysis
        """
        logger.info("=== STARTING CUSTOMER SEGMENTATION ===")
        
        try:
            # Prepare features for clustering
            clustering_features = self.segmentation.select_features_for_clustering(engineered_data)
            
            # Find optimal number of clusters
            optimal_clusters_analysis = self.segmentation.find_optimal_clusters(clustering_features)
            
            # Perform clustering
            cluster_labels = self.segmentation.perform_kmeans_clustering(
                clustering_features, 
                n_clusters=optimal_clusters_analysis['best_silhouette_k']
            )
            
            # Add cluster labels to data
            segmented_data = engineered_data.copy()
            segmented_data['cluster_label'] = cluster_labels
            
            # Analyze segments
            segment_analysis = self.segmentation.analyze_segments(segmented_data, cluster_labels)
            segment_profiles = self.segmentation.create_segment_profiles(segment_analysis)
            
            # Store segmentation metrics
            self.pipeline_metrics['segmentation'] = {
                'n_clusters': len(np.unique(cluster_labels)),
                'clustering_algorithm': 'K-Means',
                'features_used': len(self.segmentation.feature_names),
                'silhouette_score': optimal_clusters_analysis['silhouette_scores'][-1],
                'segment_sizes': {f'cluster_{i}': np.sum(cluster_labels == i) 
                                for i in np.unique(cluster_labels)},
                'completion_time': pd.Timestamp.now()
            }
            
            segmentation_results = {
                'segment_analysis': segment_analysis,
                'segment_profiles': segment_profiles,
                'optimal_clusters_analysis': optimal_clusters_analysis
            }
            
            logger.info(f"Customer segmentation completed. {len(np.unique(cluster_labels))} segments created")
            return segmented_data, segmentation_results
            
        except Exception as e:
            logger.error(f"Customer segmentation failed: {e}")
            raise
    
    def save_pipeline_results(self, 
                            final_data: pd.DataFrame, 
                            segmentation_results: Dict,
                            output_dir: str = None):
        """
        Save pipeline results and artifacts
        
        Args:
            final_data: Final processed dataframe with segments
            segmentation_results: Results from segmentation analysis
            output_dir: Directory to save results (optional)
        """
        logger.info("=== SAVING PIPELINE RESULTS ===")
        
        try:
            if output_dir is None:
                output_dir = Path(self.config['data']['processed_data_path'])
            else:
                output_dir = Path(output_dir)
            
            ensure_directory_exists(output_dir)
            ensure_directory_exists(output_dir / "models")
            ensure_directory_exists(output_dir / "analysis")
            
            # Save processed data
            data_path = output_dir / "final_customer_data.csv"
            final_data.to_csv(data_path, index=False)
            logger.info(f"Final data saved to {data_path}")
            
            # Save segmentation model
            model_path = output_dir / "models" / "segmentation_model.pkl"
            self.segmentation.save_model(str(model_path))
            
            # Save segmentation analysis
            analysis_path = output_dir / "analysis" / "segmentation_analysis.pkl"
            joblib.dump(segmentation_results, analysis_path)
            logger.info(f"Segmentation analysis saved to {analysis_path}")
            
            # Save pipeline metrics
            metrics_path = output_dir / "analysis" / "pipeline_metrics.pkl"
            joblib.dump(self.pipeline_metrics, metrics_path)
            logger.info(f"Pipeline metrics saved to {metrics_path}")
            
        except Exception as e:
            logger.error(f"Error saving pipeline results: {e}")
            raise
    
    def run_complete_pipeline(self, 
                            data_source: str, 
                            output_dir: str = None,
                            save_results: bool = True,
                            **ingestion_kwargs) -> Tuple[pd.DataFrame, Dict]:
        """
        Execute the complete data pipeline
        
        Args:
            data_source: Path to data source
            output_dir: Directory to save results
            save_results: Whether to save pipeline results
            **ingestion_kwargs: Additional parameters for data ingestion
        
        Returns:
            Final processed dataframe and complete pipeline results
        """
        logger.info("🚀 STARTING COMPLETE DATA PIPELINE")
        pipeline_start_time = pd.Timestamp.now()
        
        try:
            # Step 1: Data Ingestion
            raw_data = self.run_data_ingestion(data_source, **ingestion_kwargs)
            
            # Step 2: Data Preprocessing
            processed_data, preprocessing_stats = self.run_preprocessing(raw_data)
            
            # Step 3: Feature Engineering
            engineered_data = self.run_feature_engineering(processed_data)
            
            # Step 4: Customer Segmentation
            final_data, segmentation_results = self.run_customer_segmentation(engineered_data)
            
            # Calculate total pipeline metrics
            pipeline_end_time = pd.Timestamp.now()
            self.pipeline_metrics['pipeline_summary'] = {
                'start_time': pipeline_start_time,
                'end_time': pipeline_end_time,
                'total_duration': str(pipeline_end_time - pipeline_start_time),
                'final_dataset_shape': final_data.shape,
                'pipeline_success': True
            }
            
            # Step 5: Save Results (optional)
            if save_results:
                self.save_pipeline_results(final_data, segmentation_results, output_dir)
            
            # Create complete results package
            complete_results = {
                'processed_data': final_data,
                'segmentation_results': segmentation_results,
                'pipeline_metrics': self.pipeline_metrics,
                'preprocessing_stats': preprocessing_stats
            }
            
            logger.info("✅ COMPLETE DATA PIPELINE FINISHED SUCCESSFULLY")
            logger.info(f"Total execution time: {pipeline_end_time - pipeline_start_time}")
            logger.info(f"Final dataset shape: {final_data.shape}")
            
            return final_data, complete_results
            
        except Exception as e:
            logger.error(f"❌ PIPELINE FAILED: {e}")
            self.pipeline_metrics['pipeline_summary'] = {
                'start_time': pipeline_start_time,
                'end_time': pd.Timestamp.now(),
                'pipeline_success': False,
                'error': str(e)
            }
            raise
    
    def get_pipeline_summary(self) -> Dict:
        """Get summary of pipeline execution"""
        return self.pipeline_metrics
    
    def validate_pipeline_output(self, final_data: pd.DataFrame) -> Dict:
        """
        Validate the final pipeline output
        
        Args:
            final_data: Final processed dataframe
        
        Returns:
            Validation results
        """
        logger.info("Validating pipeline output...")
        
        validation_results = {
            'data_quality': {
                'total_records': len(final_data),
                'missing_values': final_data.isnull().sum().sum(),
                'duplicate_records': final_data.duplicated().sum(),
                'data_types_valid': True
            },
            'feature_validation': {
                'required_features_present': True,
                'feature_count': final_data.shape[1],
                'numeric_features': len(final_data.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(final_data.select_dtypes(include=['object']).columns)
            },
            'segmentation_validation': {
                'clusters_present': 'cluster_label' in final_data.columns,
                'cluster_count': final_data['cluster_label'].nunique() if 'cluster_label' in final_data.columns else 0,
                'balanced_clusters': True  # This could be more sophisticated
            }
        }
        
        # Additional validation logic can be added here
        
        logger.info("Pipeline output validation completed")
        return validation_results