"""
Model Evaluation Utilities for Predictive Modeling Project

This module provides comprehensive evaluation metrics for machine learning models,
including both classification and regression tasks. It contains the calculate_model_metrics
function that can be imported into your predictive modeling scripts.

Usage:
    from utils.evaluation import calculate_model_metrics
    
    # For classification
    metrics = calculate_model_metrics(y_true, y_pred, y_pred_proba, task_type='classification')
    
    # For regression
    metrics = calculate_model_metrics(y_true, y_pred, task_type='regression')
    
    # Auto-detect task type
    metrics = calculate_model_metrics(y_true, y_pred)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, mean_absolute_error, 
    mean_squared_error, r2_score, mean_absolute_percentage_error,
    log_loss, matthews_corrcoef
)
import warnings
from typing import Dict, Any, Optional, Union, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_model_metrics(y_true: np.ndarray, 
                          y_pred: np.ndarray, 
                          y_pred_proba: Optional[np.ndarray] = None,
                          task_type: str = 'auto', 
                          average: str = 'weighted',
                          verbose: bool = True) -> Dict[str, Any]:
    """
    Calculate comprehensive model evaluation metrics for both classification and regression tasks.
    This is the main function for model evaluation in your predictive modeling pipeline.
    
    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted values
        y_pred_proba (np.ndarray, optional): Predicted probabilities (for classification only)
        task_type (str): 'classification', 'regression', or 'auto' to detect automatically
        average (str): Averaging strategy for multi-class classification metrics 
                      ('weighted', 'macro', 'micro')
        verbose (bool): Whether to print metrics summary
        
    Returns:
        Dict[str, Any]: Dictionary containing all relevant metrics
        
    Examples:
        >>> # Classification example
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.model_selection import train_test_split
        >>> 
        >>> X, y = make_classification(n_samples=1000, n_features=20, n_classes=3)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        >>> 
        >>> model = RandomForestClassifier()
        >>> model.fit(X_train, y_train)
        >>> y_pred = model.predict(X_test)
        >>> y_pred_proba = model.predict_proba(X_test)
        >>> 
        >>> metrics = calculate_model_metrics(y_test, y_pred, y_pred_proba, task_type='classification')
        
        >>> # Regression example
        >>> from sklearn.datasets import make_regression
        >>> from sklearn.linear_model import LinearRegression
        >>> 
        >>> X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        >>> 
        >>> model = LinearRegression()
        >>> model.fit(X_train, y_train)
        >>> y_pred = model.predict(X_test)
        >>> 
        >>> metrics = calculate_model_metrics(y_test, y_pred, task_type='regression')
    """
    
    # Convert inputs to numpy arrays if they aren't already
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_pred_proba is not None:
        y_pred_proba = np.asarray(y_pred_proba)
    
    # Auto-detect task type if needed
    if task_type == 'auto':
        task_type = _detect_task_type(y_true)
        if verbose:
            print(f"🔍 Auto-detected task type: {task_type}")
    
    # Initialize metrics dictionary
    metrics = {
        'task_type': task_type,
        'sample_size': len(y_true)
    }
    
    # Calculate metrics based on task type
    if task_type == 'classification':
        metrics.update(_calculate_classification_metrics(y_true, y_pred, y_pred_proba, average))
        if verbose:
            _print_classification_summary(metrics)
            
    elif task_type == 'regression':
        metrics.update(_calculate_regression_metrics(y_true, y_pred))
        if verbose:
            _print_regression_summary(metrics)
    
    else:
        raise ValueError("task_type must be 'classification', 'regression', or 'auto'")
    
    return metrics


def _detect_task_type(y_true: np.ndarray) -> str:
    """
    Automatically detect if the task is classification or regression.
    
    Args:
        y_true: True target values
        
    Returns:
        str: 'classification' or 'regression'
    """
    unique_values = np.unique(y_true)
    
    # If there are fewer than 20 unique values and they're all integers, assume classification
    if len(unique_values) < 20 and np.all(y_true == y_true.astype(int)):
        return 'classification'
    else:
        return 'regression'


def _calculate_classification_metrics(y_true: np.ndarray, 
                                    y_pred: np.ndarray, 
                                    y_pred_proba: Optional[np.ndarray] = None,
                                    average: str = 'weighted') -> Dict[str, Any]:
    """Calculate comprehensive classification metrics."""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    # Matthews Correlation Coefficient
    try:
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
    except Exception:
        metrics['matthews_corrcoef'] = None
    
    # Classification report
    try:
        metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    except Exception:
        metrics['classification_report'] = None
    
    # ROC AUC and Log Loss (if probabilities are provided)
    if y_pred_proba is not None:
        try:
            unique_classes = len(np.unique(y_true))
            if unique_classes == 2:  # Binary classification
                proba_positive = y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba
                metrics['roc_auc'] = roc_auc_score(y_true, proba_positive)
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            else:  # Multi-class classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average=average)
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
        except Exception as e:
            warnings.warn(f"Could not calculate ROC AUC or Log Loss: {e}")
            metrics['roc_auc'] = None
            metrics['log_loss'] = None
    else:
        metrics['roc_auc'] = None
        metrics['log_loss'] = None
    
    # Per-class metrics for multi-class problems
    unique_labels = np.unique(y_true)
    if len(unique_labels) > 2:
        try:
            metrics['per_class_precision'] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
            metrics['per_class_recall'] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
            metrics['per_class_f1'] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
            metrics['class_labels'] = unique_labels.tolist()
        except Exception:
            pass
    
    return metrics


def _calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Calculate comprehensive regression metrics."""
    metrics = {}
    
    # Basic metrics
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['r2_score'] = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    try:
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100  # Convert to percentage
    except Exception:
        # Calculate manually if sklearn version doesn't have MAPE
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            metrics['mape'] = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        else:
            metrics['mape'] = float('inf')
    
    # Additional useful metrics
    residuals = y_true - y_pred
    metrics['max_error'] = np.max(np.abs(residuals))
    metrics['mean_error'] = np.mean(residuals)  # Bias
    metrics['std_error'] = np.std(residuals)
    metrics['median_absolute_error'] = np.median(np.abs(residuals))
    
    # Explained variance score
    try:
        metrics['explained_variance'] = 1 - np.var(residuals) / np.var(y_true)
    except Exception:
        metrics['explained_variance'] = None
    
    # Additional statistical measures
    metrics['mean_true'] = np.mean(y_true)
    metrics['std_true'] = np.std(y_true)
    metrics['mean_pred'] = np.mean(y_pred)
    metrics['std_pred'] = np.std(y_pred)
    
    return metrics


def _print_classification_summary(metrics: Dict[str, Any]) -> None:
    """Print a formatted summary of classification metrics."""
    print("\n" + "="*60)
    print("🎯 CLASSIFICATION MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"📊 Sample Size: {metrics['sample_size']:,}")
    print(f"🎯 Accuracy: {metrics['accuracy']:.4f}")
    print(f"🎯 Precision: {metrics['precision']:.4f}")
    print(f"🎯 Recall: {metrics['recall']:.4f}")
    print(f"🎯 F1-Score: {metrics['f1_score']:.4f}")
    
    if metrics.get('roc_auc') is not None:
        print(f"📈 ROC AUC: {metrics['roc_auc']:.4f}")
    if metrics.get('log_loss') is not None:
        print(f"📉 Log Loss: {metrics['log_loss']:.4f}")
    if metrics.get('matthews_corrcoef') is not None:
        print(f"🔄 Matthews Correlation: {metrics['matthews_corrcoef']:.4f}")
    
    print("="*60)


def _print_regression_summary(metrics: Dict[str, Any]) -> None:
    """Print a formatted summary of regression metrics."""
    print("\n" + "="*60)
    print("📈 REGRESSION MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"📊 Sample Size: {metrics['sample_size']:,}")
    print(f"📏 MAE (Mean Absolute Error): {metrics['mae']:.4f}")
    print(f"📏 MSE (Mean Squared Error): {metrics['mse']:.4f}")
    print(f"📏 RMSE (Root Mean Squared Error): {metrics['rmse']:.4f}")
    print(f"📊 R² Score: {metrics['r2_score']:.4f}")
    print(f"📈 MAPE: {metrics['mape']:.2f}%")
    print(f"📐 Max Error: {metrics['max_error']:.4f}")
    print(f"⚖️  Mean Error (Bias): {metrics['mean_error']:.4f}")
    print("="*60)


# Additional utility functions that might be useful in your project

def compare_models(model_results: Dict[str, Dict[str, Any]], 
                  metric: str = 'accuracy') -> pd.DataFrame:
    """
    Compare multiple models based on a specific metric.
    
    Args:
        model_results: Dictionary where keys are model names and values are metric dictionaries
        metric: The metric to compare models on
        
    Returns:
        DataFrame with model comparison
    """
    comparison_data = []
    
    for model_name, metrics in model_results.items():
        if metric in metrics:
            comparison_data.append({
                'Model': model_name,
                'Metric': metric,
                'Value': metrics[metric],
                'Task_Type': metrics.get('task_type', 'Unknown'),
                'Sample_Size': metrics.get('sample_size', 'Unknown')
            })
    
    df = pd.DataFrame(comparison_data)
    if not df.empty:
        df = df.sort_values('Value', ascending=False)
    
    return df


def export_metrics_to_csv(metrics: Dict[str, Any], 
                         filepath: str = 'model_metrics.csv') -> None:
    """
    Export metrics to a CSV file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save the CSV file
    """
    # Flatten the metrics dictionary for CSV export
    flattened_metrics = {}
    
    for key, value in metrics.items():
        if isinstance(value, (list, np.ndarray)):
            if key == 'confusion_matrix':
                # Handle confusion matrix specially
                cm = np.array(value)
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        flattened_metrics[f'{key}_{i}_{j}'] = cm[i, j]
            else:
                # Handle other lists/arrays
                for i, item in enumerate(value):
                    flattened_metrics[f'{key}_{i}'] = item
        elif isinstance(value, dict):
            # Skip nested dictionaries for CSV export
            continue
        else:
            flattened_metrics[key] = value
    
    # Create DataFrame and save
    df = pd.DataFrame([flattened_metrics])
    df.to_csv(filepath, index=False)
    print(f"✅ Metrics exported to {filepath}")


def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         normalize: bool = False,
                         figsize: Tuple[int, int] = (8, 6),
                         class_names: Optional[List[str]] = None) -> None:
    """
    Plot confusion matrix with nice formatting.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        normalize: Whether to normalize the confusion matrix
        figsize: Figure size for the plot
        class_names: Optional list of class names for labels
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
        fmt = '.2f'
    else:
        title = 'Confusion Matrix'
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()


def plot_regression_diagnostics(y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              figsize: Tuple[int, int] = (15, 5)) -> None:
    """
    Plot regression diagnostic plots.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        figsize: Figure size for the plots
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Actual vs Predicted
    axes[0].scatter(y_true, y_pred, alpha=0.6, color='blue')
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('True Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title('Actual vs Predicted')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals plot
    axes[1].scatter(y_pred, residuals, alpha=0.6, color='green')
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residuals vs Predicted')
    axes[1].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[2].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[2].set_xlabel('Residuals')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Residuals Distribution')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Quick testing function
def test_evaluation_module():
    """Test the evaluation module with sample data."""
    print("🧪 Testing evaluation module...")
    
    # Test classification
    print("\n1️⃣ Testing Classification Metrics:")
    np.random.seed(42)
    y_true_class = np.random.randint(0, 3, 100)
    y_pred_class = y_true_class.copy()
    # Add some prediction errors
    error_indices = np.random.choice(100, 20, replace=False)
    y_pred_class[error_indices] = np.random.randint(0, 3, 20)
    
    classification_metrics = calculate_model_metrics(
        y_true_class, y_pred_class, task_type='classification', verbose=True
    )
    
    # Test regression
    print("\n2️⃣ Testing Regression Metrics:")
    y_true_reg = np.random.normal(50, 15, 100)
    y_pred_reg = y_true_reg + np.random.normal(0, 5, 100)
    
    regression_metrics = calculate_model_metrics(
        y_true_reg, y_pred_reg, task_type='regression', verbose=True
    )
    
    print("\n✅ All tests passed! The evaluation module is working correctly.")
    return classification_metrics, regression_metrics


if __name__ == "__main__":
    # Run tests if this file is executed directly
    test_evaluation_module()