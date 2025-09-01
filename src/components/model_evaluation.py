# src/components/model_evaluation.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, mean_absolute_error, 
    mean_squared_error, r2_score, mean_absolute_percentage_error
)
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluation:
    """
    Model evaluation and performance analysis component
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.evaluation_results = {}
        
    def evaluate_classification_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    y_pred_proba: Optional[np.ndarray] = None,
                                    model_name: str = "model") -> Dict[str, Any]:
        """
        Comprehensive evaluation for classification models
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating classification model: {model_name}")
        
        try:
            # Basic metrics
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
                'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
                'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
                'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
                'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
                'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
                'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
            
            # AUC-ROC for binary classification
            if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
                metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm
            
            # Classification report
            metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
            
            # Per-class metrics
            unique_classes = np.unique(y_true)
            for i, class_label in enumerate(unique_classes):
                class_mask = (y_true == class_label)
                if np.sum(class_mask) > 0:
                    metrics[f'class_{class_label}_precision'] = precision_score(
                        y_true == class_label, y_pred == class_label, zero_division=0
                    )
                    metrics[f'class_{class_label}_recall'] = recall_score(
                        y_true == class_label, y_pred == class_label, zero_division=0
                    )
                    metrics[f'class_{class_label}_f1'] = f1_score(
                        y_true == class_label, y_pred == class_label, zero_division=0
                    )
            
            # Store results
            self.evaluation_results[f"{model_name}_classification"] = metrics
            
            logger.info(f"Classification evaluation completed for {model_name}")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"F1-Score (weighted): {metrics['f1_weighted']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in classification evaluation: {e}")
            raise
    
    def evaluate_regression_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                                model_name: str = "model") -> Dict[str, Any]:
        """
        Comprehensive evaluation for regression models
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating regression model: {model_name}")
        
        try:
            # Basic metrics
            metrics = {
                'mae': mean_absolute_error(y_true, y_pred),
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score(y_true, y_pred)
            }
            
            # Mean Absolute Percentage Error
            try:
                metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100
            except:
                # Calculate MAPE manually if sklearn doesn't have it
                mask = y_true != 0
                if np.sum(mask) > 0:
                    metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                else:
                    metrics['mape'] = float('inf')
            
            # Adjusted R-squared (requires number of features)
            n = len(y_true)
            if hasattr(self, 'n_features'):
                p = self.n_features
                if n > p + 1:
                    metrics['adj_r2'] = 1 - (1 - metrics['r2']) * (n - 1) / (n - p - 1)
            
            # Residual analysis
            residuals = y_true - y_pred
            metrics['residual_mean'] = np.mean(residuals)
            metrics['residual_std'] = np.std(residuals)
            metrics['residual_skewness'] = pd.Series(residuals).skew()
            metrics['residual_kurtosis'] = pd.Series(residuals).kurtosis()
            
            # Quantile-based metrics
            metrics['q25_error'] = np.percentile(np.abs(residuals), 25)
            metrics['q50_error'] = np.percentile(np.abs(residuals), 50)  # Median absolute error
            metrics['q75_error'] = np.percentile(np.abs(residuals), 75)
            metrics['q95_error'] = np.percentile(np.abs(residuals), 95)
            
            # Store results
            self.evaluation_results[f"{model_name}_regression"] = metrics
            
            logger.info(f"Regression evaluation completed for {model_name}")
            logger.info(f"R²: {metrics['r2']:.4f}")
            logger.info(f"RMSE: {metrics['rmse']:.4f}")
            logger.info(f"MAPE: {metrics['mape']:.2f}%")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in regression evaluation: {e}")
            raise
    
    def cross_validation_evaluation(self, model: Any, X: pd.DataFrame, y: pd.Series,
                                  cv_folds: int = 5, 
                                  scoring: str = 'accuracy',
                                  model_name: str = "model") -> Dict[str, Any]:
        """
        Perform cross-validation evaluation
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target variable
            cv_folds: Number of CV folds
            scoring: Scoring metric
            model_name: Name of the model
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation for {model_name}")
        
        try:
            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
            
            results = {
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_min': cv_scores.min(),
                'cv_max': cv_scores.max(),
                'scoring_metric': scoring,
                'cv_folds': cv_folds
            }
            
            # Store results
            self.evaluation_results[f"{model_name}_cv"] = results
            
            logger.info(f"CV Mean {scoring}: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            raise
    
    def learning_curve_analysis(self, model: Any, X: pd.DataFrame, y: pd.Series,
                              train_sizes: np.ndarray = None,
                              cv_folds: int = 3,
                              scoring: str = 'accuracy',
                              model_name: str = "model") -> Dict[str, Any]:
        """
        Analyze learning curves to assess model performance vs training data size
        
        Args:
            model: Model to analyze
            X: Feature matrix
            y: Target variable
            train_sizes: Array of training sizes to evaluate
            cv_folds: Number of CV folds
            scoring: Scoring metric
            model_name: Name of the model
            
        Returns:
            Learning curve results
        """
        logger.info(f"Analyzing learning curves for {model_name}")
        
        try:
            if train_sizes is None:
                train_sizes = np.linspace(0.1, 1.0, 10)
            
            # Generate learning curve
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X, y, train_sizes=train_sizes, cv=cv_folds, 
                scoring=scoring, n_jobs=-1
            )
            
            results = {
                'train_sizes': train_sizes_abs,
                'train_scores_mean': np.mean(train_scores, axis=1),
                'train_scores_std': np.std(train_scores, axis=1),
                'val_scores_mean': np.mean(val_scores, axis=1),
                'val_scores_std': np.std(val_scores, axis=1),
                'scoring_metric': scoring
            }
            
            # Identify overfitting
            final_train_score = results['train_scores_mean'][-1]
            final_val_score = results['val_scores_mean'][-1]
            overfitting_gap = final_train_score - final_val_score
            
            results['overfitting_assessment'] = {
                'final_train_score': final_train_score,
                'final_val_score': final_val_score,
                'overfitting_gap': overfitting_gap,
                'is_overfitting': overfitting_gap > 0.1  # Threshold can be adjusted
            }
            
            # Store results
            self.evaluation_results[f"{model_name}_learning_curve"] = results
            
            logger.info(f"Learning curve analysis completed for {model_name}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in learning curve analysis: {e}")
            raise
    
    def feature_importance_analysis(self, model: Any, feature_names: List[str],
                                  model_name: str = "model",
                                  top_k: int = 20) -> pd.DataFrame:
        """
        Analyze feature importance
        
        Args:
            model: Trained model
            feature_names: List of feature names
            model_name: Name of the model
            top_k: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        logger.info(f"Analyzing feature importance for {model_name}")
        
        try:
            importance_df = None
            
            # Tree-based models
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_,
                    'importance_type': 'tree_importance'
                })
            
            # Linear models
            elif hasattr(model, 'coef_'):
                coef = model.coef_.flatten() if len(model.coef_.shape) > 1 else model.coef_
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': np.abs(coef),
                    'coefficient': coef,
                    'importance_type': 'coefficient'
                })
            
            if importance_df is not None:
                # Sort by importance
                importance_df = importance_df.sort_values('importance', ascending=False)
                
                # Add relative importance
                total_importance = importance_df['importance'].sum()
                if total_importance > 0:
                    importance_df['relative_importance'] = importance_df['importance'] / total_importance * 100
                
                # Store results
                self.evaluation_results[f"{model_name}_feature_importance"] = importance_df
                
                logger.info(f"Feature importance analysis completed for {model_name}")
                logger.info(f"Top 3 features: {importance_df['feature'].head(3).tolist()}")
                
                return importance_df.head(top_k)
            else:
                logger.warning(f"Model {model_name} does not support feature importance analysis")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error in feature importance analysis: {e}")
            raise
    
    def compare_models(self, models_results: Dict[str, Dict[str, Any]],
                      metric: str = 'accuracy',
                      task_type: str = 'classification') -> pd.DataFrame:
        """
        Compare multiple models based on specified metric
        
        Args:
            models_results: Dictionary of model results
            metric: Metric to compare models on
            task_type: 'classification' or 'regression'
            
        Returns:
            DataFrame with model comparison
        """
        logger.info(f"Comparing models based on {metric}")
        
        try:
            comparison_data = []
            
            for model_name, results in models_results.items():
                if metric in results:
                    comparison_data.append({
                        'model': model_name,
                        'metric': metric,
                        'score': results[metric],
                        'task_type': task_type
                    })
                    
                    # Add additional relevant metrics
                    if task_type == 'classification':
                        if 'f1_weighted' in results:
                            comparison_data[-1]['f1_score'] = results['f1_weighted']
                        if 'precision_weighted' in results:
                            comparison_data[-1]['precision'] = results['precision_weighted']
                        if 'recall_weighted' in results:
                            comparison_data[-1]['recall'] = results['recall_weighted']
                    else:  # regression
                        if 'rmse' in results:
                            comparison_data[-1]['rmse'] = results['rmse']
                        if 'mae' in results:
                            comparison_data[-1]['mae'] = results['mae']
                        if 'mape' in results:
                            comparison_data[-1]['mape'] = results['mape']
            
            comparison_df = pd.DataFrame(comparison_data)
            
            if not comparison_df.empty:
                # Sort by the comparison metric
                ascending = metric.lower() in ['mae', 'mse', 'rmse', 'mape']  # Lower is better
                comparison_df = comparison_df.sort_values('score', ascending=ascending)
                
                # Add ranking
                comparison_df['rank'] = range(1, len(comparison_df) + 1)
            
            logger.info("Model comparison completed")
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error in model comparison: {e}")
            raise
    
    def generate_evaluation_report(self, model_name: str) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report for a model
        
        Args:
            model_name: Name of the model to report on
            
        Returns:
            Comprehensive evaluation report
        """
        logger.info(f"Generating evaluation report for {model_name}")
        
        try:
            report = {
                'model_name': model_name,
                'report_timestamp': pd.Timestamp.now(),
                'sections': {}
            }
            
            # Collect all evaluation results for this model
            for key, results in self.evaluation_results.items():
                if model_name in key:
                    section_name = key.replace(f"{model_name}_", "")
                    report['sections'][section_name] = results
            
            # Generate summary statistics
            if 'classification' in report['sections']:
                class_results = report['sections']['classification']
                report['summary'] = {
                    'task_type': 'classification',
                    'accuracy': class_results.get('accuracy', 0),
                    'f1_score': class_results.get('f1_weighted', 0),
                    'precision': class_results.get('precision_weighted', 0),
                    'recall': class_results.get('recall_weighted', 0)
                }
            elif 'regression' in report['sections']:
                reg_results = report['sections']['regression']
                report['summary'] = {
                    'task_type': 'regression',
                    'r2_score': reg_results.get('r2', 0),
                    'rmse': reg_results.get('rmse', float('inf')),
                    'mae': reg_results.get('mae', float('inf')),
                    'mape': reg_results.get('mape', float('inf'))
                }
            
            # Add recommendations
            report['recommendations'] = self._generate_model_recommendations(report)
            
            logger.info(f"Evaluation report generated for {model_name}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating evaluation report: {e}")
            raise
    
    def _generate_model_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on evaluation results
        
        Args:
            report: Evaluation report
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        try:
            if 'summary' in report:
                summary = report['summary']
                
                if summary['task_type'] == 'classification':
                    accuracy = summary.get('accuracy', 0)
                    f1_score = summary.get('f1_score', 0)
                    
                    if accuracy < 0.7:
                        recommendations.append("Consider collecting more training data or feature engineering")
                    if f1_score < 0.6:
                        recommendations.append("Model shows poor F1 score - check class imbalance")
                        
                elif summary['task_type'] == 'regression':
                    r2_score = summary.get('r2_score', 0)
                    mape = summary.get('mape', float('inf'))
                    
                    if r2_score < 0.5:
                        recommendations.append("Low R² score - consider more complex model or additional features")
                    if mape > 20:
                        recommendations.append("High prediction error - review feature selection and data quality")
            
            # Check for overfitting
            if 'learning_curve' in report.get('sections', {}):
                lc_results = report['sections']['learning_curve']
                if lc_results.get('overfitting_assessment', {}).get('is_overfitting', False):
                    recommendations.append("Model shows signs of overfitting - consider regularization or more data")
            
            # Default recommendation
            if not recommendations:
                recommendations.append("Model performance appears satisfactory - consider deployment")
                
        except Exception:
            recommendations.append("Unable to generate specific recommendations")
        
        return recommendations
    
    def save_evaluation_results(self, file_path: str):
        """
        Save all evaluation results to file
        
        Args:
            file_path: Path to save results
        """
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in self.evaluation_results.items():
                if isinstance(value, dict):
                    serializable_value = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            serializable_value[k] = v.tolist()
                        elif isinstance(v, pd.DataFrame):
                            serializable_value[k] = v.to_dict()
                        else:
                            serializable_value[k] = v
                    serializable_results[key] = serializable_value
                else:
                    serializable_results[key] = value
            
            # Save to file
            import json
            with open(file_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"Evaluation results saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
            raise