# streamlit_app/components/charts.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns

def create_kpi_chart(value: float, title: str, target: Optional[float] = None, 
                    format_type: str = "number", color: str = "#1f77b4") -> go.Figure:
    """
    Create a KPI indicator chart
    
    Args:
        value: Current value
        title: Chart title
        target: Target value (optional)
        format_type: Format type ('number', 'percentage', 'currency')
        color: Color for the indicator
        
    Returns:
        Plotly figure
    """
    # Format value based on type
    if format_type == "percentage":
        displayed_value = f"{value:.1%}"
        gauge_value = value * 100
        gauge_max = 100
    elif format_type == "currency":
        displayed_value = f"${value:,.2f}"
        gauge_value = value
        gauge_max = value * 1.5 if target is None else target * 1.2
    else:
        displayed_value = f"{value:,.0f}"
        gauge_value = value
        gauge_max = value * 1.5 if target is None else target * 1.2
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = gauge_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 16}},
        delta = {'reference': target if target else gauge_value * 0.9},
        gauge = {
            'axis': {'range': [None, gauge_max], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, gauge_max * 0.5], 'color': 'lightgray'},
                {'range': [gauge_max * 0.5, gauge_max * 0.8], 'color': 'gray'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': target if target else gauge_max * 0.9
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "darkblue", 'family': "Arial"},
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_trend_chart(df: pd.DataFrame, x_col: str, y_col: str, 
                      title: str = "Trend Analysis", 
                      color: str = "#1f77b4",
                      show_trend: bool = True) -> go.Figure:
    """
    Create a trend line chart with optional trend line
    
    Args:
        df: DataFrame with data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        title: Chart title
        color: Line color
        show_trend: Whether to show trend line
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Main trend line
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='lines+markers',
        name='Actual',
        line=dict(color=color, width=3),
        marker=dict(size=6)
    ))
    
    # Add trend line if requested
    if show_trend and len(df) > 1:
        z = np.polyfit(range(len(df)), df[y_col], 1)
        p = np.poly1d(z)
        trend_line = p(range(len(df)))
        
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=trend_line,
            mode='lines',
            name='Trend',
            line=dict(color='red', width=2, dash='dash'),
            opacity=0.7
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title(),
        height=400,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_distribution_chart(data: pd.Series, title: str = "Distribution", 
                            chart_type: str = "histogram", bins: int = 30) -> go.Figure:
    """
    Create distribution charts (histogram, box plot, violin plot)
    
    Args:
        data: Data series
        title: Chart title
        chart_type: Type of chart ('histogram', 'box', 'violin')
        bins: Number of bins for histogram
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    if chart_type == "histogram":
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=bins,
            name="Distribution",
            marker_color="#1f77b4",
            opacity=0.7
        ))
        
    elif chart_type == "box":
        fig.add_trace(go.Box(
            y=data,
            name="Distribution",
            marker_color="#1f77b4"
        ))
        
    elif chart_type == "violin":
        fig.add_trace(go.Violin(
            y=data,
            name="Distribution",
            marker_color="#1f77b4"
        ))
    
    fig.update_layout(
        title=title,
        height=400,
        showlegend=False
    )
    
    return fig

def create_correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    """
    Create correlation heatmap
    
    Args:
        df: DataFrame with numeric data
        title: Chart title
        
    Returns:
        Plotly figure
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        # Return empty figure if no numeric data
        fig = go.Figure()
        fig.add_annotation(
            text="No numeric data available for correlation analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(height=400, title=title)
        return fig
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        height=500,
        width=500
    )
    
    return fig

def create_segmentation_scatter(df: pd.DataFrame, x_col: str, y_col: str, 
                               color_col: str, title: str = "Customer Segmentation") -> go.Figure:
    """
    Create segmentation scatter plot
    
    Args:
        df: DataFrame with data
        x_col: X-axis column
        y_col: Y-axis column  
        color_col: Column for color coding
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        hover_data=df.columns.tolist(),
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        height=500,
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title()
    )
    
    return fig

def create_funnel_chart(data: Dict[str, float], title: str = "Conversion Funnel") -> go.Figure:
    """
    Create funnel chart for conversion analysis
    
    Args:
        data: Dictionary with stage names and values
        title: Chart title
        
    Returns:
        Plotly figure
    """
    stages = list(data.keys())
    values = list(data.values())
    
    fig = go.Figure(go.Funnel(
        y=stages,
        x=values,
        textinfo="value+percent initial",
        marker=dict(
            color=["deepskyblue", "lightsalmon", "tan", "teal", "silver"],
            line=dict(width=2, color="wheat")
        ),
        connector={"line": {"color": "royalblue"}}
    ))
    
    fig.update_layout(
        title=title,
        height=400
    )
    
    return fig

def create_waterfall_chart(categories: List[str], values: List[float], 
                          title: str = "Waterfall Analysis") -> go.Figure:
    """
    Create waterfall chart for incremental analysis
    
    Args:
        categories: List of category names
        values: List of values (positive/negative)
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(go.Waterfall(
        name="",
        orientation="v",
        measure=["absolute"] + ["relative"] * (len(categories) - 2) + ["total"],
        x=categories,
        texttemplate="%{text}",
        text=[f"{v:+.1f}" for v in values],
        y=values,
        connector={"mode": "between", "line": {"width": 4, "color": "rgb(0, 0, 0)", "dash": "solid"}}
    ))
    
    fig.update_layout(
        title=title,
        height=400
    )
    
    return fig

def create_gauge_chart(value: float, max_value: float, title: str, 
                      thresholds: Optional[Dict[str, float]] = None) -> go.Figure:
    """
    Create gauge chart for performance metrics
    
    Args:
        value: Current value
        max_value: Maximum value for gauge
        title: Chart title
        thresholds: Dictionary with threshold names and values
        
    Returns:
        Plotly figure
    """
    if thresholds is None:
        thresholds = {
            'Poor': max_value * 0.3,
            'Fair': max_value * 0.6,
            'Good': max_value * 0.8,
            'Excellent': max_value
        }
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, thresholds['Poor']], 'color': "lightgray"},
                {'range': [thresholds['Poor'], thresholds['Fair']], 'color': "gray"},
                {'range': [thresholds['Fair'], thresholds['Good']], 'color': "lightgreen"},
                {'range': [thresholds['Good'], max_value], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_comparison_chart(data: Dict[str, List[float]], categories: List[str], 
                          title: str = "Comparison Analysis", 
                          chart_type: str = "bar") -> go.Figure:
    """
    Create comparison chart (bar, radar, etc.)
    
    Args:
        data: Dictionary with series names and values
        categories: List of category names
        title: Chart title
        chart_type: Type of chart ('bar', 'radar')
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    if chart_type == "bar":
        for series_name, values in data.items():
            fig.add_trace(go.Bar(
                name=series_name,
                x=categories,
                y=values
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Categories",
            yaxis_title="Values",
            barmode='group',
            height=400
        )
        
    elif chart_type == "radar":
        for series_name, values in data.items():
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=series_name
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(max(values) for values in data.values())]
                )
            ),
            title=title,
            height=500
        )
    
    return fig

def create_time_series_chart(df: pd.DataFrame, date_col: str, value_cols: List[str], 
                           title: str = "Time Series Analysis") -> go.Figure:
    """
    Create time series chart with multiple series
    
    Args:
        df: DataFrame with time series data
        date_col: Date column name
        value_cols: List of value column names
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, col in enumerate(value_cols):
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df[col],
            mode='lines+markers',
            name=col.replace('_', ' ').title(),
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=4)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        height=400,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_donut_chart(data: Dict[str, float], title: str = "Distribution",
                      hole_size: float = 0.3) -> go.Figure:
    """
    Create donut chart for categorical data
    
    Args:
        data: Dictionary with categories and values
        title: Chart title
        hole_size: Size of the hole (0-1)
        
    Returns:
        Plotly figure
    """
    labels = list(data.keys())
    values = list(data.values())
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=hole_size,
        textinfo='label+percent',
        textposition='outside'
    )])
    
    fig.update_layout(
        title=title,
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def display_chart_with_download(fig: go.Figure, filename: str, 
                               chart_title: str = "Chart"):
    """
    Display chart with download option
    
    Args:
        fig: Plotly figure
        filename: Filename for download
        chart_title: Title for the chart section
    """
    st.subheader(chart_title)
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Download Options:**")
        
        # PNG download
        img_bytes = fig.to_image(format="png", width=800, height=600)
        st.download_button(
            label="📸 PNG",
            data=img_bytes,
            file_name=f"{filename}.png",
            mime="image/png"
        )
        
        # HTML download
        html_str = fig.to_html(include_plotlyjs='cdn')
        st.download_button(
            label="🌐 HTML",
            data=html_str,
            file_name=f"{filename}.html",
            mime="text/html"
        )

def create_advanced_scatter_matrix(df: pd.DataFrame, 
                                 dimensions: List[str] = None,
                                 color_col: str = None) -> go.Figure:
    """
    Create scatter plot matrix for multidimensional analysis
    
    Args:
        df: DataFrame with data
        dimensions: List of dimensions to include
        color_col: Column for color coding
        
    Returns:
        Plotly figure
    """
    if dimensions is None:
        dimensions = df.select_dtypes(include=[np.number]).columns.tolist()[:4]
    
    fig = go.Figure(data=go.Splom(
        dimensions=[dict(label=dim, values=df[dim]) for dim in dimensions],
        text=df[color_col] if color_col else None,
        marker=dict(
            color=df[color_col] if color_col else '#1f77b4',
            colorscale='Viridis',
            size=7,
            line=dict(width=0.5, color='rgb(230,230,230)')
        )
    ))
    
    fig.update_layout(
        title="Scatter Plot Matrix",
        height=600,
        width=800
    )
    
    return fig