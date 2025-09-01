# src/components/customer_segmentation.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import joblib

class CustomerSegmentation:
    """
    Customer segmentation component using various clustering algorithms
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        
    def prepare_rfm_features(self, df: pd.DataFrame, 
                           customer_id_col: str = 'Customer ID',
                           date_col: str = 'Invoice Date',
                           amount_col: str = 'Price') -> pd.DataFrame:
        """
        Prepare RFM (Recency, Frequency, Monetary) features for segmentation
        """
        logger.info("Preparing RFM features...")
        
        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        reference_date = df[date_col].max() + pd.Timedelta(days=1)
        
        # Calculate RFM metrics
        rfm = df.groupby(customer_id_col).agg({
            date_col: lambda x: (reference_date - x.max()).days,  # Recency
            customer_id_col: 'count',  # Frequency
            amount_col: 'sum'  # Monetary
        }).reset_index()
        
        rfm.columns = [customer_id_col, 'Recency', 'Frequency', 'Monetary']
        
        # Create RFM scores (1-5 scale)
        rfm['R_Score'] = pd.qcut(rfm['Recency'].rank(method='first'), 5, labels=[5,4,3,2,1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5])
        
        # Combine RFM scores
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        logger.info(f"RFM features prepared for {len(rfm)} customers")
        return rfm
    
    def create_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive customer features for segmentation
        """
        logger.info("Creating customer features...")
        
        customer_features = df.groupby('Customer ID').agg({
            'Age': 'first',
            'Gender': 'first',
            'Item Purchased': 'count',  # Purchase frequency
            'Category': lambda x: x.mode().iloc[0],  # Most frequent category
            'Purchase Amount (USD)': ['sum', 'mean', 'std'],  # Monetary features
            'Review Rating': 'mean',  # Average rating
            'Previous Purchases': 'first',
            'Season': lambda x: x.mode().iloc[0]  # Most frequent season
        }).reset_index()
        
        # Flatten column names
        customer_features.columns = [
            'Customer ID', 'Age', 'Gender', 'Purchase_Frequency', 
            'Preferred_Category', 'Total_Spent', 'Avg_Spent', 'Spending_Std',
            'Avg_Rating', 'Previous_Purchases', 'Preferred_Season'
        ]
        
        # Create derived features
        customer_features['Spending_Consistency'] = customer_features['Spending_Std'] / customer_features['Avg_Spent']
        customer_features['Customer_Tenure'] = customer_features['Previous_Purchases'] + customer_features['Purchase_Frequency']
        
        logger.info(f"Customer features created for {len(customer_features)} customers")
        return customer_features
    
    def select_features_for_clustering(self, df: pd.DataFrame, 
                                     feature_columns: Optional[List[str]] = None) -> np.ndarray:
        """
        Select and prepare features for clustering
        """
        if feature_columns is None:
            # Default feature selection
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in numeric_columns if col != 'Customer ID']
        
        self.feature_names = feature_columns
        features = df[feature_columns].fillna(0)  # Handle any remaining NaN values
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        logger.info(f"Selected {len(feature_columns)} features for clustering")
        logger.info(f"Features: {feature_columns}")
        
        return features_scaled
    
    def find_optimal_clusters(self, X: np.ndarray, max_clusters: int = 10) -> Dict:
        """
        Find optimal number of clusters using elbow method and silhouette analysis
        """
        logger.info("Finding optimal number of clusters...")
        
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        cluster_range = range(2, max_clusters + 1)
        
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, cluster_labels))
            calinski_scores.append(calinski_harabasz_score(X, cluster_labels))
        
        # Find elbow point (simplified approach)
        # Calculate the rate of change
        deltas = np.diff(inertias)
        elbow_point = cluster_range[np.argmax(deltas[:-1] - deltas[1:]) + 1]
        
        # Best silhouette score
        best_silhouette_k = cluster_range[np.argmax(silhouette_scores)]
        
        results = {
            'cluster_range': list(cluster_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'calinski_scores': calinski_scores,
            'elbow_point': elbow_point,
            'best_silhouette_k': best_silhouette_k
        }
        
        logger.info(f"Elbow method suggests: {elbow_point} clusters")
        logger.info(f"Best silhouette score at: {best_silhouette_k} clusters")
        
        return results
    
    def perform_kmeans_clustering(self, X: np.ndarray, n_clusters: int = None) -> np.ndarray:
        """
        Perform K-means clustering
        """
        if n_clusters is None:
            n_clusters = self.config['modeling']['segmentation']['n_clusters']
        
        logger.info(f"Performing K-means clustering with {n_clusters} clusters...")
        
        self.model = KMeans(
            n_clusters=n_clusters, 
            random_state=self.config['modeling']['segmentation']['random_state'],
            n_init=10
        )
        
        cluster_labels = self.model.fit_predict(X)
        
        # Calculate clustering metrics
        silhouette = silhouette_score(X, cluster_labels)
        calinski = calinski_harabasz_score(X, cluster_labels)
        
        logger.info(f"Clustering completed. Silhouette Score: {silhouette:.3f}")
        logger.info(f"Calinski-Harabasz Score: {calinski:.3f}")
        
        return cluster_labels
    
    def analyze_segments(self, df: pd.DataFrame, cluster_labels: np.ndarray) -> Dict:
        """
        Analyze the characteristics of each customer segment
        """
        logger.info("Analyzing customer segments...")
        
        # Add cluster labels to dataframe
        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = cluster_labels
        
        # Calculate segment statistics
        segment_stats = {}
        
        for cluster_id in np.unique(cluster_labels):
            cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
            
            stats = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df_with_clusters) * 100,
            }
            
            # Calculate statistics for numeric features
            for feature in self.feature_names:
                if feature in df_with_clusters.columns:
                    stats[f'{feature}_mean'] = cluster_data[feature].mean()
                    stats[f'{feature}_median'] = cluster_data[feature].median()
                    stats[f'{feature}_std'] = cluster_data[feature].std()
            
            segment_stats[f'Cluster_{cluster_id}'] = stats
        
        logger.info(f"Segment analysis completed for {len(segment_stats)} clusters")
        return segment_stats
    
    def create_segment_profiles(self, segment_stats: Dict) -> Dict:
        """
        Create interpretable segment profiles with business names
        """
        segment_profiles = {}
        
        for cluster_name, stats in segment_stats.items():
            profile = {
                'size': stats['size'],
                'percentage': stats['percentage'],
                'characteristics': {}
            }
            
            # Analyze key characteristics
            if 'Total_Spent_mean' in stats:
                if stats['Total_Spent_mean'] > 1000:
                    profile['characteristics']['spending_level'] = 'High'
                elif stats['Total_Spent_mean'] > 500:
                    profile['characteristics']['spending_level'] = 'Medium'
                else:
                    profile['characteristics']['spending_level'] = 'Low'
            
            if 'Purchase_Frequency_mean' in stats:
                if stats['Purchase_Frequency_mean'] > 10:
                    profile['characteristics']['engagement'] = 'Highly Engaged'
                elif stats['Purchase_Frequency_mean'] > 5:
                    profile['characteristics']['engagement'] = 'Moderately Engaged'
                else:
                    profile['characteristics']['engagement'] = 'Low Engagement'
            
            if 'Avg_Rating_mean' in stats:
                if stats['Avg_Rating_mean'] > 4.0:
                    profile['characteristics']['satisfaction'] = 'Highly Satisfied'
                elif stats['Avg_Rating_mean'] > 3.5:
                    profile['characteristics']['satisfaction'] = 'Satisfied'
                else:
                    profile['characteristics']['satisfaction'] = 'Needs Attention'
            
            segment_profiles[cluster_name] = profile
        
        return segment_profiles
    
    def save_model(self, model_path: str):
        """Save the trained clustering model"""
        try:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, model_path: str):
        """Load a trained clustering model"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise