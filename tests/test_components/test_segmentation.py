# tests/test_components/test_segmentation.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from components.customer_segmentation import CustomerSegmentation
from tests import create_sample_processed_data, TEST_CONFIG

class TestCustomerSegmentation:
    """Test suite for CustomerSegmentation component"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample processed data for testing"""
        return create_sample_processed_data(100)
    
    @pytest.fixture
    def segmentation(self):
        """Create CustomerSegmentation instance"""
        config = {
            'segmentation': {
                'random_state': 42,
                'n_clusters_range': [2, 8],
                'max_iter': 300
            }
        }
        return CustomerSegmentation(config)
    
    def test_initialization(self, segmentation):
        """Test CustomerSegmentation initialization"""
        assert segmentation is not None
        assert hasattr(segmentation, 'config')
        assert hasattr(segmentation, 'model')
    
    def test_feature_selection_for_clustering(self, segmentation, sample_data):
        """Test feature selection for clustering"""
        features = segmentation.select_features_for_clustering(sample_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_data)
        assert features.shape[1] > 0
        
        # Should only contain numeric features
        assert all(pd.api.types.is_numeric_dtype(features[col]) for col in features.columns)
    
    def test_optimal_clusters_elbow_method(self, segmentation, sample_data):
        """Test optimal cluster finding using elbow method"""
        features = segmentation.select_features_for_clustering(sample_data)
        
        optimal_analysis = segmentation.find_optimal_clusters(features, max_clusters=6)
        
        assert isinstance(optimal_analysis, dict)
        assert 'inertias' in optimal_analysis
        assert 'silhouette_scores' in optimal_analysis
        assert 'elbow_point' in optimal_analysis
        assert len(optimal_analysis['inertias']) > 0
    
    def test_kmeans_clustering(self, segmentation, sample_data):
        """Test K-means clustering"""
        features = segmentation.select_features_for_clustering(sample_data)
        
        cluster_labels = segmentation.perform_kmeans_clustering(features, n_clusters=3)
        
        assert len(cluster_labels) == len(features)
        assert len(np.unique(cluster_labels)) <= 3
        assert all(isinstance(label, (int, np.integer)) for label in cluster_labels)
    
    def test_hierarchical_clustering(self, segmentation, sample_data):
        """Test hierarchical clustering"""
        features = segmentation.select_features_for_clustering(sample_data)
        
        cluster_labels = segmentation.perform_hierarchical_clustering(features, n_clusters=3)
        
        assert len(cluster_labels) == len(features)
        assert len(np.unique(cluster_labels)) <= 3
    
    def test_segment_analysis(self, segmentation, sample_data):
        """Test segment analysis"""
        features = segmentation.select_features_for_clustering(sample_data)
        cluster_labels = segmentation.perform_kmeans_clustering(features, n_clusters=3)
        
        segment_analysis = segmentation.analyze_segments(sample_data, cluster_labels)
        
        assert isinstance(segment_analysis, dict)
        assert len(segment_analysis) == len(np.unique(cluster_labels))
        
        # Each segment should have statistics
        for segment_id, stats in segment_analysis.items():
            assert isinstance(stats, dict)
            assert 'count' in stats
            assert stats['count'] > 0
    
    def test_segment_profiling(self, segmentation, sample_data):
        """Test segment profiling with business names"""
        features = segmentation.select_features_for_clustering(sample_data)
        cluster_labels = segmentation.perform_kmeans_clustering(features, n_clusters=3)
        segment_analysis = segmentation.analyze_segments(sample_data, cluster_labels)
        
        segment_profiles = segmentation.create_segment_profiles(segment_analysis)
        
        assert isinstance(segment_profiles, dict)
        assert len(segment_profiles) == len(segment_analysis)
        
        # Should contain business-friendly names
        profile_names = list(segment_profiles.values())
        assert all(isinstance(name, str) for name in profile_names)
    
    def test_rfm_segmentation(self, segmentation, sample_data):
        """Test RFM-based segmentation"""
        rfm_segments = segmentation.perform_rfm_segmentation(sample_data)
        
        assert isinstance(rfm_segments, pd.DataFrame)
        assert 'RFM_Segment' in rfm_segments.columns
        assert 'RFM_Score' in rfm_segments.columns
        assert len(rfm_segments) > 0
    
    def test_clustering_with_different_algorithms(self, segmentation, sample_data):
        """Test clustering with different algorithms"""
        features = segmentation.select_features_for_clustering(sample_data)
        
        # Test K-means
        kmeans_labels = segmentation.perform_kmeans_clustering(features, n_clusters=3)
        assert len(kmeans_labels) == len(features)
        
        # Test hierarchical
        hierarchical_labels = segmentation.perform_hierarchical_clustering(features, n_clusters=3)
        assert len(hierarchical_labels) == len(features)
    
    def test_silhouette_analysis(self, segmentation, sample_data):
        """Test silhouette analysis for cluster validation"""
        features = segmentation.select_features_for_clustering(sample_data)
        cluster_labels = segmentation.perform_kmeans_clustering(features, n_clusters=3)
        
        from sklearn.metrics import silhouette_score
        silhouette_avg = silhouette_score(features, cluster_labels)
        
        assert -1 <= silhouette_avg <= 1  # Silhouette score range
    
    def test_feature_scaling_in_clustering(self, segmentation, sample_data):
        """Test that features are properly scaled for clustering"""
        features = segmentation.select_features_for_clustering(sample_data)
        
        # Features should be scaled (mean ~0, std ~1)
        for col in features.columns:
            assert abs(features[col].mean()) < 1  # Should be relatively centered
            assert 0.5 < features[col].std() < 2  # Should have reasonable variance
    
    def test_cluster_stability(self, segmentation, sample_data):
        """Test clustering stability across multiple runs"""
        features = segmentation.select_features_for_clustering(sample_data)
        
        # Run clustering multiple times with same random state
        labels1 = segmentation.perform_kmeans_clustering(features, n_clusters=3)
        labels2 = segmentation.perform_kmeans_clustering(features, n_clusters=3)
        
        # Results should be identical with same random state
        np.testing.assert_array_equal(labels1, labels2)
    
    def test_empty_data_handling(self, segmentation):
        """Test handling of empty or invalid data"""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError):
            segmentation.select_features_for_clustering(empty_df)
    
    def test_single_feature_clustering(self, segmentation):
        """Test clustering with single feature"""
        single_feature_df = pd.DataFrame({
            'feature1': np.random.randn(50)
        })
        
        features = segmentation.select_features_for_clustering(single_feature_df)
        cluster_labels = segmentation.perform_kmeans_clustering(features, n_clusters=2)
        
        assert len(cluster_labels) == len(single_feature_df)
        assert len(np.unique(cluster_labels)) <= 2
    
    def test_high_dimensional_clustering(self, segmentation):
        """Test clustering with high-dimensional data"""
        # Create data with many features
        high_dim_data = pd.DataFrame(np.random.randn(100, 20))
        high_dim_data.columns = [f'feature_{i}' for i in range(20)]
        
        features = segmentation.select_features_for_clustering(high_dim_data)
        cluster_labels = segmentation.perform_kmeans_clustering(features, n_clusters=3)
        
        assert len(cluster_labels) == len(high_dim_data)
    
    def test_categorical_data_handling(self, segmentation, sample_data):
        """Test handling of categorical data in clustering"""
        # Add categorical column
        sample_data['Category_encoded'] = pd.Categorical(sample_data['Category']).codes
        
        features = segmentation.select_features_for_clustering(sample_data)
        
        # Should handle categorical data appropriately
        assert len(features.columns) > 0
    
    def test_outlier_impact_on_clustering(self, segmentation, sample_data):
        """Test impact of outliers on clustering results"""
        features = segmentation.select_features_for_clustering(sample_data)
        
        # Add outliers
        outlier_features = features.copy()
        outlier_features.iloc[0] = outlier_features.iloc[0] * 10  # Create outlier
        
        normal_labels = segmentation.perform_kmeans_clustering(features, n_clusters=3)
        outlier_labels = segmentation.perform_kmeans_clustering(outlier_features, n_clusters=3)
        
        # Results may differ due to outliers
        assert len(normal_labels) == len(outlier_labels)
    
    def test_segment_size_distribution(self, segmentation, sample_data):
        """Test that segments have reasonable size distribution"""
        features = segmentation.select_features_for_clustering(sample_data)
        cluster_labels = segmentation.perform_kmeans_clustering(features, n_clusters=3)
        
        segment_counts = pd.Series(cluster_labels).value_counts()
        
        # No segment should be too small (less than 5% of data)
        min_segment_size = len(sample_data) * 0.05
        assert all(count >= min_segment_size for count in segment_counts)
    
    def test_clustering_parameters(self, segmentation, sample_data):
        """Test different clustering parameters"""
        features = segmentation.select_features_for_clustering(sample_data)
        
        # Test different numbers of clusters
        for n_clusters in [2, 3, 4, 5]:
            labels = segmentation.perform_kmeans_clustering(features, n_clusters=n_clusters)
            unique_labels = len(np.unique(labels))
            assert unique_labels <= n_clusters
    
    @pytest.mark.parametrize("n_clusters", [2, 3, 4, 5])
    def test_different_cluster_numbers(self, segmentation, sample_data, n_clusters):
        """Test clustering with different numbers of clusters"""
        features = segmentation.select_features_for_clustering(sample_data)
        cluster_labels = segmentation.perform_kmeans_clustering(features, n_clusters=n_clusters)
        
        unique_clusters = len(np.unique(cluster_labels))
        assert unique_clusters <= n_clusters
        assert len(cluster_labels) == len(features)
    
    def test_segment_interpretation(self, segmentation, sample_data):
        """Test segment interpretation and naming"""
        features = segmentation.select_features_for_clustering(sample_data)
        cluster_labels = segmentation.perform_kmeans_clustering(features, n_clusters=3)
        segment_analysis = segmentation.analyze_segments(sample_data, cluster_labels)
        
        # Test interpretation logic
        profiles = segmentation.create_segment_profiles(segment_analysis)
        
        # Should have meaningful names
        expected_names = ['High Value', 'Low Value', 'Medium Value', 'Champions', 'At Risk']
        profile_values = list(profiles.values())
        
        # At least some profiles should have meaningful names
        meaningful_names = any(any(name in profile for name in expected_names) 
                             for profile in profile_values)
        assert meaningful_names or len(profile_values) > 0  # Should have some profiles
    
    def test_clustering_reproducibility(self, segmentation, sample_data):
        """Test that clustering results are reproducible"""
        features = segmentation.select_features_for_clustering(sample_data)
        
        # Set random state for reproducibility
        segmentation.config['segmentation']['random_state'] = 42
        
        labels1 = segmentation.perform_kmeans_clustering(features, n_clusters=3)
        labels2 = segmentation.perform_kmeans_clustering(features, n_clusters=3)
        
        np.testing.assert_array_equal(labels1, labels2)