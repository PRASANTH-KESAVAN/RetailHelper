# src/utils/visualization.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Set default style
plt.style.use('default')
sns.set_palette("husl")

def set_plot_style(style: str = "whitegrid"):
    """
    Set the default plotting style
    
    Args:
        style: Style name ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
    """
    sns.set_style(style)
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

def create_distribution_plot(data: pd.Series, title: str = "Distribution Plot", 
                           bins: int = 30, color: str = "skyblue") -> plt.Figure:
    """
    Create a distribution plot
    
    Args:
        data: Data series to plot
        title: Plot title
        bins: Number of bins for histogram
        color: Plot color
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    ax1.hist(data.dropna(), bins=bins, color=color, alpha=0.7, edgecolor='black')
    ax1.set_title(f'{title} - Histogram')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    
    # Box plot
    ax2.boxplot(data.dropna(), vert=True)
    ax2.set_title(f'{title} - Box Plot')
    ax2.set_ylabel('Value')
    
    plt.tight_layout()
    return fig

def create_correlation_heatmap(data: pd.DataFrame, title: str = "Correlation Heatmap",
                             figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Create a correlation heatmap
    
    Args:
        data: DataFrame with numerical data
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        logger.warning("No numeric columns found for correlation heatmap")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No numeric data available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=16)
        return fig
    
    # Calculate correlation matrix
    corr_matrix = numeric_data.corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title(title, fontsize=16, pad=20)
    
    plt.tight_layout()
    return fig

def create_segmentation_plot(data: pd.DataFrame, x_col: str, y_col: str, 
                           cluster_col: str, title: str = "Customer Segmentation") -> go.Figure:
    """
    Create an interactive segmentation scatter plot
    
    Args:
        data: DataFrame with customer data
        x_col: X-axis column name
        y_col: Y-axis column name  
        cluster_col: Cluster label column name
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = px.scatter(data, x=x_col, y=y_col, color=cluster_col,
                    title=title, hover_data=['Customer_ID'] if 'Customer_ID' in data.columns else None)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_x=0.5
    )
    
    return fig

def create_rfm_analysis_plot(rfm_data: pd.DataFrame) -> go.Figure:
    """
    Create RFM analysis visualization
    
    Args:
        rfm_data: DataFrame with RFM scores
        
    Returns:
        Plotly figure with subplots
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Recency Distribution', 'Frequency Distribution', 
                       'Monetary Distribution', 'RFM Segments'),
        specs=[[{'type': 'histogram'}, {'type': 'histogram'}],
               [{'type': 'histogram'}, {'type': 'bar'}]]
    )
    
    # Recency histogram
    fig.add_trace(go.Histogram(x=rfm_data['Recency'], name='Recency', 
                              marker_color='lightblue'), row=1, col=1)
    
    # Frequency histogram
    fig.add_trace(go.Histogram(x=rfm_data['Frequency'], name='Frequency',
                              marker_color='lightgreen'), row=1, col=2)
    
    # Monetary histogram
    fig.add_trace(go.Histogram(x=rfm_data['Monetary'], name='Monetary',
                              marker_color='lightcoral'), row=2, col=1)
    
    # RFM Segments bar chart
    if 'RFM_Segment' in rfm_data.columns:
        segment_counts = rfm_data['RFM_Segment'].value_counts()
        fig.add_trace(go.Bar(x=segment_counts.index, y=segment_counts.values,
                           name='Segments', marker_color='gold'), row=2, col=2)
    
    fig.update_layout(height=800, showlegend=False, title_text="RFM Analysis Dashboard")
    return fig

def create_customer_journey_plot(transaction_data: pd.DataFrame, customer_id: str) -> go.Figure:
    """
    Create customer journey visualization
    
    Args:
        transaction_data: DataFrame with transaction data
        customer_id: Specific customer ID
        
    Returns:
        Plotly figure
    """
    # Filter data for specific customer
    customer_data = transaction_data[transaction_data['Customer ID'] == customer_id].copy()
    
    if customer_data.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"No data found for customer {customer_id}",
                         xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Sort by date
    customer_data['Purchase Date'] = pd.to_datetime(customer_data['Purchase Date'])
    customer_data = customer_data.sort_values('Purchase Date')
    
    # Create timeline plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=customer_data['Purchase Date'],
        y=customer_data['Purchase Amount (USD)'],
        mode='lines+markers',
        name='Purchase Amount',
        marker=dict(size=8, color=customer_data['Review Rating'], 
                   colorscale='RdYlGn', showscale=True),
        text=customer_data['Category'],
        hovertemplate='<b>Date:</b> %{x}<br><b>Amount:</b> $%{y}<br><b>Category:</b> %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Customer Journey - {customer_id}',
        xaxis_title='Purchase Date',
        yaxis_title='Purchase Amount (USD)',
        height=500
    )
    
    return fig

def create_churn_analysis_plot(data: pd.DataFrame, prediction_col: str = 'churn_probability') -> go.Figure:
    """
    Create churn analysis visualization
    
    Args:
        data: DataFrame with churn predictions
        prediction_col: Column with churn probabilities
        
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Churn Probability Distribution', 'Churn by Age Group',
                       'Churn by Purchase Frequency', 'Risk Categories'),
        specs=[[{'type': 'histogram'}, {'type': 'bar'}],
               [{'type': 'scatter'}, {'type': 'pie'}]]
    )
    
    # Churn probability distribution
    fig.add_trace(go.Histogram(x=data[prediction_col], name='Churn Probability',
                              marker_color='red', opacity=0.7), row=1, col=1)
    
    # Churn by age group
    if 'Age' in data.columns:
        age_groups = pd.cut(data['Age'], bins=5)
        churn_by_age = data.groupby(age_groups)[prediction_col].mean()
        fig.add_trace(go.Bar(x=[str(x) for x in churn_by_age.index], 
                           y=churn_by_age.values, name='Avg Churn Prob',
                           marker_color='orange'), row=1, col=2)
    
    # Churn vs Purchase Frequency scatter
    if 'Purchase_Frequency' in data.columns:
        fig.add_trace(go.Scatter(x=data['Purchase_Frequency'], y=data[prediction_col],
                               mode='markers', name='Frequency vs Churn',
                               marker=dict(color='blue', opacity=0.6)), row=2, col=1)
    
    # Risk categories pie chart
    if 'churn_risk_level' in data.columns:
        risk_counts = data['churn_risk_level'].value_counts()
        fig.add_trace(go.Pie(labels=risk_counts.index, values=risk_counts.values,
                           name='Risk Categories'), row=2, col=2)
    
    fig.update_layout(height=800, title_text="Churn Analysis Dashboard")
    return fig

def create_clv_analysis_plot(data: pd.DataFrame, clv_col: str = 'predicted_clv') -> go.Figure:
    """
    Create Customer Lifetime Value analysis visualization
    
    Args:
        data: DataFrame with CLV predictions
        clv_col: Column with CLV values
        
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('CLV Distribution', 'CLV by Customer Segment',
                       'CLV vs Age', 'CLV Quartiles'),
        specs=[[{'type': 'histogram'}, {'type': 'box'}],
               [{'type': 'scatter'}, {'type': 'bar'}]]
    )
    
    # CLV distribution
    fig.add_trace(go.Histogram(x=data[clv_col], name='CLV Distribution',
                              marker_color='green', opacity=0.7), row=1, col=1)
    
    # CLV by segment (if available)
    if 'predicted_segment' in data.columns:
        fig.add_trace(go.Box(x=data['predicted_segment'], y=data[clv_col],
                           name='CLV by Segment'), row=1, col=2)
    
    # CLV vs Age scatter
    if 'Age' in data.columns:
        fig.add_trace(go.Scatter(x=data['Age'], y=data[clv_col],
                               mode='markers', name='Age vs CLV',
                               marker=dict(color='purple', opacity=0.6)), row=2, col=1)
    
    # CLV quartiles
    if 'clv_quartile' in data.columns:
        quartile_counts = data['clv_quartile'].value_counts()
        fig.add_trace(go.Bar(x=quartile_counts.index, y=quartile_counts.values,
                           name='CLV Quartiles', marker_color='gold'), row=2, col=2)
    
    fig.update_layout(height=800, title_text="Customer Lifetime Value Analysis")
    return fig

def create_recommendation_analysis_plot(recommendation_metrics: Dict[str, Any]) -> go.Figure:
    """
    Create recommendation system performance visualization
    
    Args:
        recommendation_metrics: Dictionary with recommendation metrics
        
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Method Performance', 'Precision vs Recall', 'Coverage Analysis'),
        specs=[[{'type': 'bar'}, {'type': 'scatter'}, {'type': 'indicator'}]]
    )
    
    # Method performance comparison
    methods = list(recommendation_metrics.keys())
    f1_scores = [metrics.get('f1_score', 0) for metrics in recommendation_metrics.values()]
    
    fig.add_trace(go.Bar(x=methods, y=f1_scores, name='F1 Score',
                        marker_color='lightblue'), row=1, col=1)
    
    # Precision vs Recall scatter
    precisions = [metrics.get('precision', 0) for metrics in recommendation_metrics.values()]
    recalls = [metrics.get('recall', 0) for metrics in recommendation_metrics.values()]
    
    fig.add_trace(go.Scatter(x=recalls, y=precisions, mode='markers+text',
                           text=methods, name='Precision vs Recall',
                           marker=dict(size=12, color='red')), row=1, col=2)
    
    # Coverage indicator
    avg_coverage = np.mean([metrics.get('coverage', 0) for metrics in recommendation_metrics.values()])
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=avg_coverage,
        title={'text': "Avg Coverage"},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "darkgreen"},
               'steps': [{'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "gray"}]},
    ), row=1, col=3)
    
    fig.update_layout(height=400, title_text="Recommendation System Analysis")
    return fig

def create_performance_dashboard(model_metrics: Dict[str, Any]) -> go.Figure:
    """
    Create a comprehensive model performance dashboard
    
    Args:
        model_metrics: Dictionary with model performance metrics
        
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Model Accuracy Comparison', 'Precision Scores', 'Recall Scores',
                       'F1 Scores', 'ROC AUC Scores', 'Overall Performance'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}, {'type': 'radar'}]]
    )
    
    models = list(model_metrics.keys())
    
    # Accuracy
    accuracies = [metrics.get('accuracy', 0) for metrics in model_metrics.values()]
    fig.add_trace(go.Bar(x=models, y=accuracies, name='Accuracy', 
                        marker_color='blue'), row=1, col=1)
    
    # Precision
    precisions = [metrics.get('precision', 0) for metrics in model_metrics.values()]
    fig.add_trace(go.Bar(x=models, y=precisions, name='Precision',
                        marker_color='green'), row=1, col=2)
    
    # Recall
    recalls = [metrics.get('recall', 0) for metrics in model_metrics.values()]
    fig.add_trace(go.Bar(x=models, y=recalls, name='Recall',
                        marker_color='orange'), row=1, col=3)
    
    # F1 Scores
    f1_scores = [metrics.get('f1_score', 0) for metrics in model_metrics.values()]
    fig.add_trace(go.Bar(x=models, y=f1_scores, name='F1 Score',
                        marker_color='red'), row=2, col=1)
    
    # ROC AUC
    aucs = [metrics.get('auc_roc', 0) for metrics in model_metrics.values()]
    fig.add_trace(go.Bar(x=models, y=aucs, name='ROC AUC',
                        marker_color='purple'), row=2, col=2)
    
    # Radar chart for overall performance
    if models:
        fig.add_trace(go.Scatterpolar(
            r=[accuracies[0], precisions[0], recalls[0], f1_scores[0], aucs[0]],
            theta=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'],
            fill='toself',
            name=models[0]
        ), row=2, col=3)
    
    fig.update_layout(height=800, title_text="Model Performance Dashboard")
    return fig

def create_sidebar(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create sidebar filters for Streamlit dashboard
    
    Args:
        df: DataFrame to create filters for
        
    Returns:
        Dictionary of filter options
    """
    import streamlit as st
    
    st.sidebar.header("🔧 Filters")
    
    filters = {}
    
    # Age filter
    if 'Age' in df.columns:
        age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
        filters['age_range'] = st.sidebar.slider(
            "Age Range", age_min, age_max, (age_min, age_max)
        )
    
    # Gender filter
    if 'Gender' in df.columns:
        filters['gender'] = st.sidebar.multiselect(
            "Gender", options=df['Gender'].unique(), 
            default=df['Gender'].unique()
        )
    
    # Category filter
    if 'Category' in df.columns:
        filters['category'] = st.sidebar.multiselect(
            "Product Category", options=df['Category'].unique(),
            default=df['Category'].unique()
        )
    
    # Location filter
    if 'Location' in df.columns:
        filters['location'] = st.sidebar.multiselect(
            "Location", options=df['Location'].unique(),
            default=df['Location'].unique()
        )
    
    # Purchase amount filter
    if 'Purchase Amount (USD)' in df.columns:
        amount_min, amount_max = float(df['Purchase Amount (USD)'].min()), float(df['Purchase Amount (USD)'].max())
        filters['amount_range'] = st.sidebar.slider(
            "Purchase Amount Range ($)", amount_min, amount_max, (amount_min, amount_max)
        )
    
    return filters

def save_plot(fig: Union[plt.Figure, go.Figure], filename: str, 
              output_dir: str = "reports/figures/", format: str = "png") -> None:
    """
    Save plot to file
    
    Args:
        fig: Figure object (matplotlib or plotly)
        filename: Name of the file (without extension)
        output_dir: Output directory
        format: File format ('png', 'html', 'pdf')
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_path = output_path / f"{filename}.{format}"
    
    try:
        if isinstance(fig, plt.Figure):
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            logger.info(f"Matplotlib plot saved: {file_path}")
        elif hasattr(fig, 'write_html') or hasattr(fig, 'write_image'):
            if format == 'html':
                fig.write_html(file_path)
            else:
                fig.write_image(file_path)
            logger.info(f"Plotly plot saved: {file_path}")
        else:
            logger.error(f"Unsupported figure type: {type(fig)}")
            
    except Exception as e:
        logger.error(f"Error saving plot: {e}")

def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply filters to DataFrame
    
    Args:
        df: Input DataFrame
        filters: Dictionary of filter values
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    try:
        # Age filter
        if 'age_range' in filters and 'Age' in filtered_df.columns:
            age_min, age_max = filters['age_range']
            filtered_df = filtered_df[filtered_df['Age'].between(age_min, age_max)]
        
        # Gender filter
        if 'gender' in filters and 'Gender' in filtered_df.columns:
            if filters['gender']:  # Check if not empty
                filtered_df = filtered_df[filtered_df['Gender'].isin(filters['gender'])]
        
        # Category filter
        if 'category' in filters and 'Category' in filtered_df.columns:
            if filters['category']:
                filtered_df = filtered_df[filtered_df['Category'].isin(filters['category'])]
        
        # Location filter
        if 'location' in filters and 'Location' in filtered_df.columns:
            if filters['location']:
                filtered_df = filtered_df[filtered_df['Location'].isin(filters['location'])]
        
        # Amount filter
        if 'amount_range' in filters and 'Purchase Amount (USD)' in filtered_df.columns:
            amount_min, amount_max = filters['amount_range']
            filtered_df = filtered_df[filtered_df['Purchase Amount (USD)'].between(amount_min, amount_max)]
        
        logger.info(f"Applied filters. Dataset reduced from {len(df)} to {len(filtered_df)} records")
        return filtered_df
        
    except Exception as e:
        logger.error(f"Error applying filters: {e}")
        return df