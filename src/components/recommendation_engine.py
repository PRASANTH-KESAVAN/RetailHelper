# src/components/recommendation_engine.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class RecommendationEngine:
    """
    Recommendation engine for product and customer recommendations
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.similarity_matrices = {}
        self.item_features = None
        self.user_features = None
        self.user_item_matrix = None
        
    def prepare_user_item_matrix(self, df: pd.DataFrame, 
                                user_col: str = 'Customer ID',
                                item_col: str = 'Category',
                                rating_col: str = 'Review Rating') -> csr_matrix:
        """
        Prepare user-item interaction matrix
        
        Args:
            df: Input dataframe
            user_col: Column name for users
            item_col: Column name for items
            rating_col: Column name for ratings/interactions
            
        Returns:
            Sparse user-item matrix
        """
        logger.info("Preparing user-item interaction matrix...")
        
        try:
            # Create user-item matrix
            user_item_df = df.groupby([user_col, item_col])[rating_col].mean().reset_index()
            user_item_pivot = user_item_df.pivot(index=user_col, columns=item_col, values=rating_col)
            user_item_pivot = user_item_pivot.fillna(0)
            
            # Convert to sparse matrix for efficiency
            self.user_item_matrix = csr_matrix(user_item_pivot.values)
            self.user_mapping = {user: idx for idx, user in enumerate(user_item_pivot.index)}
            self.item_mapping = {item: idx for idx, item in enumerate(user_item_pivot.columns)}
            self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
            self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
            
            logger.info(f"User-item matrix created: {self.user_item_matrix.shape}")
            return self.user_item_matrix
            
        except Exception as e:
            logger.error(f"Error preparing user-item matrix: {e}")
            raise
    
    def collaborative_filtering_user_based(self, user_id: str, 
                                         n_recommendations: int = 5,
                                         similarity_threshold: float = 0.1) -> List[Tuple[str, float]]:
        """
        Generate recommendations using user-based collaborative filtering
        
        Args:
            user_id: ID of the user to recommend for
            n_recommendations: Number of recommendations to generate
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of (item, score) tuples
        """
        logger.info(f"Generating user-based collaborative filtering recommendations for user {user_id}...")
        
        try:
            if self.user_item_matrix is None:
                raise ValueError("User-item matrix not prepared. Call prepare_user_item_matrix first.")
            
            if user_id not in self.user_mapping:
                logger.warning(f"User {user_id} not found in training data")
                return []
            
            user_idx = self.user_mapping[user_id]
            
            # Calculate user similarities
            user_similarities = cosine_similarity(
                self.user_item_matrix[user_idx].reshape(1, -1),
                self.user_item_matrix
            )[0]
            
            # Get target user's ratings
            user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
            
            # Find items the user hasn't rated
            unrated_items = np.where(user_ratings == 0)[0]
            
            if len(unrated_items) == 0:
                logger.info("User has rated all items")
                return []
            
            # Calculate weighted ratings for unrated items
            recommendations = []
            
            for item_idx in unrated_items:
                # Get users who rated this item
                item_raters = np.where(self.user_item_matrix[:, item_idx] > 0)[0]
                
                if len(item_raters) == 0:
                    continue
                
                # Calculate weighted average rating
                similar_users = item_raters[user_similarities[item_raters] > similarity_threshold]
                
                if len(similar_users) > 0:
                    weights = user_similarities[similar_users]
                    ratings = self.user_item_matrix[similar_users, item_idx].toarray().flatten()
                    
                    if np.sum(weights) > 0:
                        weighted_rating = np.average(ratings, weights=weights)
                        item_name = self.reverse_item_mapping[item_idx]
                        recommendations.append((item_name, weighted_rating))
            
            # Sort by predicted rating and return top N
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Generated {len(recommendations[:n_recommendations])} recommendations")
            return recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error in user-based collaborative filtering: {e}")
            raise
    
    def collaborative_filtering_item_based(self, user_id: str,
                                         n_recommendations: int = 5) -> List[Tuple[str, float]]:
        """
        Generate recommendations using item-based collaborative filtering
        
        Args:
            user_id: ID of the user to recommend for
            n_recommendations: Number of recommendations to generate
            
        Returns:
            List of (item, score) tuples
        """
        logger.info(f"Generating item-based collaborative filtering recommendations for user {user_id}...")
        
        try:
            if self.user_item_matrix is None:
                raise ValueError("User-item matrix not prepared.")
            
            if user_id not in self.user_mapping:
                logger.warning(f"User {user_id} not found in training data")
                return []
            
            user_idx = self.user_mapping[user_id]
            user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
            
            # Find items the user has rated
            rated_items = np.where(user_ratings > 0)[0]
            unrated_items = np.where(user_ratings == 0)[0]
            
            if len(rated_items) == 0 or len(unrated_items) == 0:
                return []
            
            # Calculate item similarities if not cached
            if 'item_similarity' not in self.similarity_matrices:
                logger.info("Calculating item similarity matrix...")
                self.similarity_matrices['item_similarity'] = cosine_similarity(
                    self.user_item_matrix.T
                )
            
            item_similarity = self.similarity_matrices['item_similarity']
            
            # Generate recommendations
            recommendations = []
            
            for unrated_item in unrated_items:
                # Calculate similarity-weighted rating
                similarities = item_similarity[unrated_item, rated_items]
                ratings = user_ratings[rated_items]
                
                if np.sum(np.abs(similarities)) > 0:
                    predicted_rating = np.average(ratings, weights=np.abs(similarities))
                    item_name = self.reverse_item_mapping[unrated_item]
                    recommendations.append((item_name, predicted_rating))
            
            # Sort by predicted rating
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Generated {len(recommendations[:n_recommendations])} recommendations")
            return recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error in item-based collaborative filtering: {e}")
            raise
    
    def matrix_factorization_recommendations(self, user_id: str,
                                           n_recommendations: int = 5,
                                           n_components: int = 50) -> List[Tuple[str, float]]:
        """
        Generate recommendations using matrix factorization (SVD)
        
        Args:
            user_id: ID of the user to recommend for
            n_recommendations: Number of recommendations to generate
            n_components: Number of latent factors
            
        Returns:
            List of (item, score) tuples
        """
        logger.info(f"Generating matrix factorization recommendations for user {user_id}...")
        
        try:
            if self.user_item_matrix is None:
                raise ValueError("User-item matrix not prepared.")
            
            if user_id not in self.user_mapping:
                logger.warning(f"User {user_id} not found in training data")
                return []
            
            # Perform SVD if not cached
            if 'svd_model' not in self.models:
                logger.info("Training SVD model...")
                svd = TruncatedSVD(n_components=n_components, random_state=42)
                self.models['svd_model'] = svd
                self.models['user_factors'] = svd.fit_transform(self.user_item_matrix)
                self.models['item_factors'] = svd.components_
            
            user_idx = self.user_mapping[user_id]
            user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
            
            # Generate predictions for all items
            user_factor = self.models['user_factors'][user_idx]
            predicted_ratings = np.dot(user_factor, self.models['item_factors'])
            
            # Find items the user hasn't rated
            unrated_items = np.where(user_ratings == 0)[0]
            
            # Get recommendations for unrated items
            recommendations = []
            for item_idx in unrated_items:
                item_name = self.reverse_item_mapping[item_idx]
                predicted_rating = predicted_ratings[item_idx]
                recommendations.append((item_name, predicted_rating))
            
            # Sort by predicted rating
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Generated {len(recommendations[:n_recommendations])} recommendations")
            return recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error in matrix factorization recommendations: {e}")
            raise
    
    def content_based_recommendations(self, df: pd.DataFrame, user_id: str,
                                    n_recommendations: int = 5,
                                    content_features: List[str] = None) -> List[Tuple[str, float]]:
        """
        Generate content-based recommendations
        
        Args:
            df: Input dataframe with item features
            user_id: ID of the user to recommend for
            n_recommendations: Number of recommendations to generate
            content_features: List of content feature columns
            
        Returns:
            List of (item, score) tuples
        """
        logger.info(f"Generating content-based recommendations for user {user_id}...")
        
        try:
            if content_features is None:
                content_features = ['Category', 'Season', 'Color', 'Size']
            
            # Filter features that exist in the dataframe
            available_features = [f for f in content_features if f in df.columns]
            
            if len(available_features) == 0:
                logger.error("No content features available")
                return []
            
            # Get user's purchase history
            user_history = df[df['Customer ID'] == user_id]
            
            if user_history.empty:
                logger.warning(f"No purchase history found for user {user_id}")
                return []
            
            # Create content feature vectors
            def create_content_vector(row):
                return ' '.join([str(row[feature]) for feature in available_features])
            
            df['content_features'] = df.apply(create_content_vector, axis=1)
            user_content = user_history.apply(create_content_vector, axis=1)
            
            # Create TF-IDF vectors
            tfidf = TfidfVectorizer()
            content_matrix = tfidf.fit_transform(df['content_features'].unique())
            
            # Create user profile (average of user's item vectors)
            user_items_content = user_history['content_features'].values
            user_tfidf = tfidf.transform(user_items_content)
            user_profile = np.mean(user_tfidf.toarray(), axis=0).reshape(1, -1)
            
            # Calculate similarities with all items
            unique_content = df['content_features'].unique()
            unique_tfidf = tfidf.transform(unique_content)
            similarities = cosine_similarity(user_profile, unique_tfidf)[0]
            
            # Get items the user hasn't purchased
            user_purchased_categories = set(user_history['Category'].values)
            all_categories = set(df['Category'].values)
            unpurchased_categories = all_categories - user_purchased_categories
            
            # Generate recommendations
            recommendations = []
            content_to_category = dict(zip(df['content_features'], df['Category']))
            
            for i, content in enumerate(unique_content):
                category = content_to_category.get(content)
                if category in unpurchased_categories:
                    recommendations.append((category, similarities[i]))
            
            # Sort by similarity score
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Generated {len(recommendations[:n_recommendations])} recommendations")
            return recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error in content-based recommendations: {e}")
            raise
    
    def hybrid_recommendations(self, df: pd.DataFrame, user_id: str,
                             n_recommendations: int = 5,
                             cf_weight: float = 0.6,
                             content_weight: float = 0.4) -> List[Tuple[str, float]]:
        """
        Generate hybrid recommendations combining collaborative filtering and content-based
        
        Args:
            df: Input dataframe
            user_id: ID of the user to recommend for
            n_recommendations: Number of recommendations to generate
            cf_weight: Weight for collaborative filtering
            content_weight: Weight for content-based filtering
            
        Returns:
            List of (item, score) tuples
        """
        logger.info(f"Generating hybrid recommendations for user {user_id}...")
        
        try:
            # Get collaborative filtering recommendations
            cf_recommendations = self.collaborative_filtering_item_based(user_id, n_recommendations * 2)
            
            # Get content-based recommendations
            content_recommendations = self.content_based_recommendations(df, user_id, n_recommendations * 2)
            
            # Combine recommendations
            cf_dict = {item: score for item, score in cf_recommendations}
            content_dict = {item: score for item, score in content_recommendations}
            
            # Normalize scores to 0-1 range
            if cf_dict:
                max_cf_score = max(cf_dict.values())
                min_cf_score = min(cf_dict.values())
                cf_dict = {item: (score - min_cf_score) / (max_cf_score - min_cf_score + 1e-6) 
                          for item, score in cf_dict.items()}
            
            if content_dict:
                max_content_score = max(content_dict.values())
                min_content_score = min(content_dict.values())
                content_dict = {item: (score - min_content_score) / (max_content_score - min_content_score + 1e-6)
                              for item, score in content_dict.items()}
            
            # Combine scores
            all_items = set(cf_dict.keys()) | set(content_dict.keys())
            hybrid_scores = {}
            
            for item in all_items:
                cf_score = cf_dict.get(item, 0)
                content_score = content_dict.get(item, 0)
                hybrid_score = cf_weight * cf_score + content_weight * content_score
                hybrid_scores[item] = hybrid_score
            
            # Sort by hybrid score
            recommendations = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
            
            logger.info(f"Generated {len(recommendations[:n_recommendations])} hybrid recommendations")
            return recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error in hybrid recommendations: {e}")
            raise
    
    def get_similar_users(self, user_id: str, n_users: int = 5) -> List[Tuple[str, float]]:
        """
        Find similar users based on purchase behavior
        
        Args:
            user_id: ID of the target user
            n_users: Number of similar users to return
            
        Returns:
            List of (user_id, similarity_score) tuples
        """
        logger.info(f"Finding similar users for user {user_id}...")
        
        try:
            if self.user_item_matrix is None:
                raise ValueError("User-item matrix not prepared.")
            
            if user_id not in self.user_mapping:
                logger.warning(f"User {user_id} not found")
                return []
            
            user_idx = self.user_mapping[user_id]
            
            # Calculate user similarities
            user_similarities = cosine_similarity(
                self.user_item_matrix[user_idx].reshape(1, -1),
                self.user_item_matrix
            )[0]
            
            # Get similar users (excluding the user themselves)
            similar_users = []
            for i, similarity in enumerate(user_similarities):
                if i != user_idx and similarity > 0:
                    similar_user_id = self.reverse_user_mapping[i]
                    similar_users.append((similar_user_id, similarity))
            
            # Sort by similarity
            similar_users.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Found {len(similar_users[:n_users])} similar users")
            return similar_users[:n_users]
            
        except Exception as e:
            logger.error(f"Error finding similar users: {e}")
            raise
    
    def get_item_recommendations_for_segment(self, df: pd.DataFrame, 
                                           segment_label: int,
                                           n_recommendations: int = 5) -> List[Tuple[str, float]]:
        """
        Get popular item recommendations for a customer segment
        
        Args:
            df: Input dataframe with customer segments
            segment_label: Target customer segment
            n_recommendations: Number of recommendations to generate
            
        Returns:
            List of (item, popularity_score) tuples
        """
        logger.info(f"Generating segment-based recommendations for segment {segment_label}...")
        
        try:
            # Filter data for the target segment
            segment_data = df[df['cluster_label'] == segment_label] if 'cluster_label' in df.columns else df
            
            if segment_data.empty:
                logger.warning(f"No data found for segment {segment_label}")
                return []
            
            # Calculate item popularity within the segment
            item_popularity = segment_data.groupby('Category').agg({
                'Review Rating': 'mean',
                'Purchase Amount (USD)': 'mean',
                'Customer ID': 'count'
            }).reset_index()
            
            item_popularity.columns = ['Category', 'Avg_Rating', 'Avg_Amount', 'Purchase_Count']
            
            # Calculate popularity score (weighted combination)
            item_popularity['Popularity_Score'] = (
                0.4 * (item_popularity['Avg_Rating'] / 5.0) +
                0.3 * (item_popularity['Purchase_Count'] / item_popularity['Purchase_Count'].max()) +
                0.3 * (item_popularity['Avg_Amount'] / item_popularity['Avg_Amount'].max())
            )
            
            # Sort by popularity score
            item_popularity = item_popularity.sort_values('Popularity_Score', ascending=False)
            
            recommendations = [(row['Category'], row['Popularity_Score']) 
                             for _, row in item_popularity.head(n_recommendations).iterrows()]
            
            logger.info(f"Generated {len(recommendations)} segment-based recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in segment-based recommendations: {e}")
            raise
    
    def evaluate_recommendations(self, df: pd.DataFrame, 
                               test_users: List[str] = None,
                               recommendation_method: str = 'collaborative_filtering') -> Dict[str, float]:
        """
        Evaluate recommendation system performance
        
        Args:
            df: Input dataframe
            test_users: List of users to test on
            recommendation_method: Method to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {recommendation_method} recommendations...")
        
        try:
            if test_users is None:
                # Use a sample of users for evaluation
                all_users = list(self.user_mapping.keys())
                test_users = np.random.choice(all_users, min(50, len(all_users)), replace=False)
            
            precision_scores = []
            recall_scores = []
            
            for user_id in test_users:
                try:
                    # Get user's actual purchases
                    user_purchases = set(df[df['Customer ID'] == user_id]['Category'].values)
                    
                    if len(user_purchases) < 2:
                        continue
                    
                    # Get recommendations
                    if recommendation_method == 'collaborative_filtering':
                        recommendations = self.collaborative_filtering_item_based(user_id, 5)
                    elif recommendation_method == 'content_based':
                        recommendations = self.content_based_recommendations(df, user_id, 5)
                    elif recommendation_method == 'hybrid':
                        recommendations = self.hybrid_recommendations(df, user_id, 5)
                    else:
                        continue
                    
                    if not recommendations:
                        continue
                    
                    recommended_items = set([item for item, _ in recommendations])
                    
                    # Calculate precision and recall
                    true_positives = len(recommended_items & user_purchases)
                    precision = true_positives / len(recommended_items) if recommended_items else 0
                    recall = true_positives / len(user_purchases) if user_purchases else 0
                    
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    
                except Exception:
                    continue
            
            # Calculate average metrics
            avg_precision = np.mean(precision_scores) if precision_scores else 0
            avg_recall = np.mean(recall_scores) if recall_scores else 0
            f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
            
            metrics = {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1_score': f1_score,
                'users_evaluated': len(precision_scores)
            }
            
            logger.info(f"Evaluation completed: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error in recommendation evaluation: {e}")
            raise