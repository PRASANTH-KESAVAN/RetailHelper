# src/components/predictive_modeling.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, mean_absolute_error, mean_squared_error, r2_score)
from loguru import logger
import joblib
import warnings
warnings.filterwarnings('ignore')

class PredictiveModeling:
    """
    Predictive modeling component for various ML tasks
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.model_performance = {}
        
        # Initialize available models
        self.available_models = {
            'classification': {
                'random_forest': RandomForestClassifier(random_state=42),
                'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
                'gradient_boosting': GradientBoostingClassifier(random_state=42),
                'xgboost': XGBClassifier(random_state=42, eval_metric='logloss'),
                'decision_tree': DecisionTreeClassifier(random_state=42),
                'svm': SVC(random_state=42, probability=True),
                'naive_bayes': GaussianNB(),
                'knn': KNeighborsClassifier()
            },
            'regression': {
                'random_forest': RandomForestRegressor(random_state=42),
                'linear_regression': LinearRegression(),
                'xgboost': XGBRegressor(random_state=42),
                'svr': SVR()
            }
        }
    
    def prepare_churn_prediction_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for churn prediction
        
        Args:
            df: Input dataframe with customer features
            
        Returns:
            Feature matrix and target variable
        """
        logger.info("Preparing data for churn prediction...")
        
        try:
            # Create churn target based on business rules
            # Customers are considered churned if:
            # 1. Low recent activity (high recency)
            # 2. Low purchase frequency
            # 3. Low satisfaction scores
            
            df_churn = df.copy()
            
            # Calculate churn score based on multiple factors
            churn_indicators = []
            
            # Recency indicator (high recency = higher churn risk)
            if 'Recency' in df_churn.columns:
                recency_threshold = df_churn['Recency'].quantile(0.75)
                churn_indicators.append(df_churn['Recency'] > recency_threshold)
            
            # Frequency indicator (low frequency = higher churn risk)
            if 'Frequency' in df_churn.columns:
                frequency_threshold = df_churn['Frequency'].quantile(0.25)
                churn_indicators.append(df_churn['Frequency'] < frequency_threshold)
            
            # Rating indicator (low ratings = higher churn risk)
            if 'Avg_Rating' in df_churn.columns:
                rating_threshold = df_churn['Avg_Rating'].quantile(0.25)
                churn_indicators.append(df_churn['Avg_Rating'] < rating_threshold)
            
            # Monetary indicator (low spending = higher churn risk)
            if 'Monetary' in df_churn.columns:
                monetary_threshold = df_churn['Monetary'].quantile(0.25)
                churn_indicators.append(df_churn['Monetary'] < monetary_threshold)
            
            # Combine indicators (customer churns if 2 or more indicators are true)
            if churn_indicators:
                churn_score = sum(churn_indicators)
                df_churn['churn'] = (churn_score >= 2).astype(int)
            else:
                # Fallback: random churn labels for demonstration
                np.random.seed(42)
                df_churn['churn'] = np.random.binomial(1, 0.2, len(df_churn))
            
            # Select features for modeling
            feature_columns = [col for col in df_churn.columns 
                             if col not in ['Customer_ID', 'churn'] and 
                             df_churn[col].dtype in [np.int64, np.float64]]
            
            X = df_churn[feature_columns]
            y = df_churn['churn']
            
            # Handle missing values
            X = X.fillna(X.median())
            
            logger.info(f"Churn prediction data prepared: {X.shape}, Churn rate: {y.mean():.2%}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing churn prediction data: {e}")
            raise
    
    def prepare_clv_prediction_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for Customer Lifetime Value prediction
        
        Args:
            df: Input dataframe with customer features
            
        Returns:
            Feature matrix and target variable (CLV)
        """
        logger.info("Preparing data for CLV prediction...")
        
        try:
            df_clv = df.copy()
            
            # Calculate CLV as target variable
            # CLV = Average Order Value * Purchase Frequency * Customer Lifespan
            if 'Estimated_CLV' in df_clv.columns:
                target_col = 'Estimated_CLV'
            elif 'Monetary' in df_clv.columns and 'Frequency' in df_clv.columns:
                # Simple CLV calculation
                df_clv['CLV'] = df_clv['Monetary'] * df_clv['Frequency'] * 2  # Assume 2-year lifespan
                target_col = 'CLV'
            else:
                # Fallback CLV calculation
                if 'Total_Spent' in df_clv.columns and 'Purchase_Count' in df_clv.columns:
                    df_clv['CLV'] = df_clv['Total_Spent'] * 2  # Simple estimation
                    target_col = 'CLV'
                else:
                    raise ValueError("Insufficient data to calculate CLV")
            
            # Select features for modeling (exclude CLV-related columns)
            exclude_cols = ['Customer_ID', target_col, 'Estimated_CLV', 'CLV', 'Total_Spent']
            feature_columns = [col for col in df_clv.columns 
                             if col not in exclude_cols and 
                             df_clv[col].dtype in [np.int64, np.float64]]
            
            X = df_clv[feature_columns]
            y = df_clv[target_col]
            
            # Handle missing values and outliers
            X = X.fillna(X.median())
            
            # Remove extreme outliers in target
            y_q1 = y.quantile(0.01)
            y_q99 = y.quantile(0.99)
            mask = (y >= y_q1) & (y <= y_q99)
            X = X[mask]
            y = y[mask]
            
            logger.info(f"CLV prediction data prepared: {X.shape}, Mean CLV: ${y.mean():.2f}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing CLV prediction data: {e}")
            raise
    
    def prepare_purchase_prediction_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for next purchase prediction
        
        Args:
            df: Input dataframe with customer features
            
        Returns:
            Feature matrix and target variable (will purchase in next period)
        """
        logger.info("Preparing data for purchase prediction...")
        
        try:
            df_purchase = df.copy()
            
            # Create purchase likelihood target based on RFM and behavioral features
            purchase_indicators = []
            
            # Recent activity indicator
            if 'Recency' in df_purchase.columns:
                recency_threshold = df_purchase['Recency'].quantile(0.5)
                purchase_indicators.append(df_purchase['Recency'] <= recency_threshold)
            
            # High frequency indicator
            if 'Frequency' in df_purchase.columns:
                frequency_threshold = df_purchase['Frequency'].quantile(0.6)
                purchase_indicators.append(df_purchase['Frequency'] >= frequency_threshold)
            
            # High satisfaction indicator
            if 'Avg_Rating' in df_purchase.columns:
                rating_threshold = df_purchase['Avg_Rating'].quantile(0.6)
                purchase_indicators.append(df_purchase['Avg_Rating'] >= rating_threshold)
            
            # Subscription indicator
            if 'Is_Subscriber' in df_purchase.columns:
                purchase_indicators.append(df_purchase['Is_Subscriber'] == 1)
            
            # Combine indicators
            if purchase_indicators:
                purchase_score = sum(purchase_indicators)
                df_purchase['will_purchase'] = (purchase_score >= 2).astype(int)
            else:
                # Fallback: random purchase labels
                np.random.seed(42)
                df_purchase['will_purchase'] = np.random.binomial(1, 0.6, len(df_purchase))
            
            # Select features
            feature_columns = [col for col in df_purchase.columns 
                             if col not in ['Customer_ID', 'will_purchase'] and 
                             df_purchase[col].dtype in [np.int64, np.float64]]
            
            X = df_purchase[feature_columns]
            y = df_purchase['will_purchase']
            
            # Handle missing values
            X = X.fillna(X.median())
            
            logger.info(f"Purchase prediction data prepared: {X.shape}, Purchase rate: {y.mean():.2%}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing purchase prediction data: {e}")
            raise
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   model_type: str = 'random_forest', 
                   task_type: str = 'classification',
                   test_size: float = 0.2,
                   cross_validation: bool = True) -> Dict[str, Any]:
        """
        Train a machine learning model
        
        Args:
            X: Feature matrix
            y: Target variable
            model_type: Type of model to train
            task_type: 'classification' or 'regression'
            test_size: Proportion of data for testing
            cross_validation: Whether to perform cross-validation
            
        Returns:
            Dictionary with model, performance metrics, and other results
        """
        logger.info(f"Training {model_type} model for {task_type}...")
        
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y if task_type == 'classification' else None
            )
            
            # Scale features
            scaler_name = f"{task_type}_{model_type}"
            self.scalers[scaler_name] = StandardScaler()
            X_train_scaled = self.scalers[scaler_name].fit_transform(X_train)
            X_test_scaled = self.scalers[scaler_name].transform(X_test)
            
            # Get the model
            model = self.available_models[task_type][model_type]
            
            # Train the model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            if task_type == 'classification' and hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_pred_proba = None
            
            # Calculate performance metrics
            if task_type == 'classification':
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                }
                
                if y_pred_proba is not None and len(np.unique(y_test)) == 2:
                    metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba)
                    
            else:  # regression
                metrics = {
                    'mae': mean_absolute_error(y_test, y_pred),
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'r2': r2_score(y_test, y_pred)
                }
            
            # Cross-validation
            cv_scores = None
            if cross_validation:
                cv_scoring = 'accuracy' if task_type == 'classification' else 'r2'
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring=cv_scoring)
                metrics[f'cv_{cv_scoring}_mean'] = cv_scores.mean()
                metrics[f'cv_{cv_scoring}_std'] = cv_scores.std()
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
            elif hasattr(model, 'coef_'):
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'coefficient': model.coef_.flatten() if len(model.coef_.shape) > 1 else model.coef_
                })
            
            # Store the model
            model_name = f"{task_type}_{model_type}"
            self.models[model_name] = model
            self.feature_names = X.columns.tolist()
            
            results = {
                'model': model,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'cv_scores': cv_scores,
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'scaler': self.scalers[scaler_name]
            }
            
            # Store performance
            self.model_performance[model_name] = metrics
            
            logger.info(f"Model training completed. Performance: {metrics}")
            return results
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, 
                            model_type: str = 'random_forest',
                            task_type: str = 'classification') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            X: Feature matrix
            y: Target variable
            model_type: Type of model to tune
            task_type: 'classification' or 'regression'
            
        Returns:
            Results with best model and parameters
        """
        logger.info(f"Performing hyperparameter tuning for {model_type}...")
        
        try:
            # Define parameter grids
            param_grids = {
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'xgboost': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0]
                },
                'logistic_regression': {
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['liblinear', 'lbfgs'],
                    'penalty': ['l1', 'l2']
                }
            }
            
            if model_type not in param_grids:
                logger.warning(f"No parameter grid defined for {model_type}, using default parameters")
                return self.train_model(X, y, model_type, task_type)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, 
                stratify=y if task_type == 'classification' else None
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Get base model
            model = self.available_models[task_type][model_type]
            
            # Perform grid search
            scoring = 'accuracy' if task_type == 'classification' else 'r2'
            grid_search = GridSearchCV(
                model, param_grids[model_type], 
                cv=3, scoring=scoring, n_jobs=-1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            
            # Make predictions
            y_pred = best_model.predict(X_test_scaled)
            
            # Calculate metrics
            if task_type == 'classification':
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                }
            else:
                metrics = {
                    'mae': mean_absolute_error(y_test, y_pred),
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'r2': r2_score(y_test, y_pred)
                }
            
            results = {
                'best_model': best_model,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'metrics': metrics,
                'grid_search_results': grid_search.cv_results_,
                'scaler': scaler
            }
            
            logger.info(f"Hyperparameter tuning completed. Best score: {grid_search.best_score_:.4f}")
            logger.info(f"Best parameters: {grid_search.best_params_}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {e}")
            raise
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using a trained model
        
        Args:
            model_name: Name of the trained model
            X: Feature matrix
            
        Returns:
            Predictions array
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            
            model = self.models[model_name]
            scaler = self.scalers[model_name]
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Make predictions
            predictions = model.predict(X_scaled)
            
            logger.info(f"Predictions made using {model_name}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def save_model(self, model_name: str, file_path: str):
        """
        Save a trained model to file
        
        Args:
            model_name: Name of the model to save
            file_path: Path to save the model
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model_data = {
                'model': self.models[model_name],
                'scaler': self.scalers[model_name],
                'feature_names': self.feature_names,
                'performance': self.model_performance.get(model_name, {})
            }
            
            joblib.dump(model_data, file_path)
            logger.info(f"Model saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, model_name: str, file_path: str):
        """
        Load a trained model from file
        
        Args:
            model_name: Name to assign to the loaded model
            file_path: Path to the model file
        """
        try:
            model_data = joblib.load(file_path)
            
            self.models[model_name] = model_data['model']
            self.scalers[model_name] = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.model_performance[model_name] = model_data.get('performance', {})
            
            logger.info(f"Model loaded from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def compare_models(self, X: pd.DataFrame, y: pd.Series, 
                      model_types: List[str] = None,
                      task_type: str = 'classification') -> pd.DataFrame:
        """
        Compare performance of multiple models
        
        Args:
            X: Feature matrix
            y: Target variable
            model_types: List of model types to compare
            task_type: 'classification' or 'regression'
            
        Returns:
            DataFrame with model comparison results
        """
        logger.info(f"Comparing models for {task_type}...")
        
        try:
            if model_types is None:
                model_types = ['random_forest', 'logistic_regression', 'xgboost'] if task_type == 'classification' else ['random_forest', 'linear_regression', 'xgboost']
            
            comparison_results = []
            
            for model_type in model_types:
                try:
                    results = self.train_model(X, y, model_type, task_type, cross_validation=True)
                    
                    result_dict = {'model_type': model_type}
                    result_dict.update(results['metrics'])
                    comparison_results.append(result_dict)
                    
                except Exception as e:
                    logger.error(f"Error training {model_type}: {e}")
                    continue
            
            comparison_df = pd.DataFrame(comparison_results)
            
            # Sort by primary metric
            primary_metric = 'accuracy' if task_type == 'classification' else 'r2'
            if primary_metric in comparison_df.columns:
                comparison_df = comparison_df.sort_values(primary_metric, ascending=False)
            
            logger.info("Model comparison completed")
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error in model comparison: {e}")
            raise