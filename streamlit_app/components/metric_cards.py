# streamlit_app/components/metrics_cards.py

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import plotly.graph_objects as go
from datetime import datetime, timedelta

def create_metric_card(title: str, value: Union[float, int, str], 
                      delta: Optional[Union[float, int]] = None,
                      delta_color: str = "normal",
                      format_func: Optional[callable] = None,
                      help_text: Optional[str] = None,
                      icon: Optional[str] = None) -> None:
    """
    Create a single metric card with optional delta and formatting
    
    Args:
        title: Metric title
        value: Current value
        delta: Change from previous period
        delta_color: Color for delta ('normal', 'inverse')
        format_func: Function to format the value
        help_text: Help text for the metric
        icon: Icon emoji for the metric
    """
    # Format value if function provided
    if format_func:
        formatted_value = format_func(value)
    else:
        if isinstance(value, (int, float)):
            formatted_value = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
        else:
            formatted_value = str(value)
    
    # Add icon to title if provided
    display_title = f"{icon} {title}" if icon else title
    
    st.metric(
        label=display_title,
        value=formatted_value,
        delta=delta,
        delta_color=delta_color,
        help=help_text
    )

def create_kpi_grid(metrics: Dict[str, Dict[str, Any]], columns: int = 4) -> None:
    """
    Create a grid of KPI metric cards
    
    Args:
        metrics: Dictionary of metrics with their properties
        columns: Number of columns in the grid
    """
    # Create columns
    cols = st.columns(columns)
    
    for i, (metric_name, metric_data) in enumerate(metrics.items()):
        with cols[i % columns]:
            create_metric_card(
                title=metric_data.get('title', metric_name),
                value=metric_data.get('value', 0),
                delta=metric_data.get('delta'),
                delta_color=metric_data.get('delta_color', 'normal'),
                format_func=metric_data.get('format_func'),
                help_text=metric_data.get('help_text'),
                icon=metric_data.get('icon')
            )

def create_financial_metrics_card(revenue: float, profit: float, margin: float,
                                 previous_revenue: Optional[float] = None) -> None:
    """
    Create financial metrics card with revenue, profit, and margin
    
    Args:
        revenue: Current revenue
        profit: Current profit  
        margin: Profit margin percentage
        previous_revenue: Previous period revenue for delta calculation
    """
    st.markdown("### 💰 Financial Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        delta_revenue = None
        if previous_revenue is not None:
            delta_revenue = revenue - previous_revenue
        
        create_metric_card(
            title="Total Revenue",
            value=revenue,
            delta=delta_revenue,
            format_func=lambda x: f"${x:,.2f}",
            help_text="Total revenue for the selected period",
            icon="💵"
        )
    
    with col2:
        create_metric_card(
            title="Profit",
            value=profit,
            format_func=lambda x: f"${x:,.2f}",
            help_text="Total profit after costs",
            icon="📈"
        )
    
    with col3:
        create_metric_card(
            title="Profit Margin",
            value=margin,
            format_func=lambda x: f"{x:.1%}",
            help_text="Profit as percentage of revenue",
            icon="📊"
        )

def create_customer_metrics_card(total_customers: int, new_customers: int,
                                churn_rate: float, clv: float) -> None:
    """
    Create customer metrics card
    
    Args:
        total_customers: Total number of customers
        new_customers: New customers acquired
        churn_rate: Customer churn rate
        clv: Average customer lifetime value
    """
    st.markdown("### 👥 Customer Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card(
            title="Total Customers",
            value=total_customers,
            format_func=lambda x: f"{x:,}",
            help_text="Total active customers",
            icon="👥"
        )
    
    with col2:
        create_metric_card(
            title="New Customers",
            value=new_customers,
            format_func=lambda x: f"{x:,}",
            help_text="Customers acquired this period",
            icon="🆕"
        )
    
    with col3:
        create_metric_card(
            title="Churn Rate",
            value=churn_rate,
            format_func=lambda x: f"{x:.1%}",
            delta_color="inverse",
            help_text="Percentage of customers lost",
            icon="⚠️"
        )
    
    with col4:
        create_metric_card(
            title="Avg CLV",
            value=clv,
            format_func=lambda x: f"${x:,.2f}",
            help_text="Average customer lifetime value",
            icon="💎"
        )

def create_performance_scorecard(metrics: Dict[str, float], 
                               targets: Optional[Dict[str, float]] = None,
                               title: str = "Performance Scorecard") -> None:
    """
    Create a performance scorecard with traffic light indicators
    
    Args:
        metrics: Dictionary of metric names and values
        targets: Dictionary of target values
        title: Title for the scorecard
    """
    st.markdown(f"### {title}")
    
    # Create HTML for scorecard
    html_content = """
    <div style="border: 1px solid #ddd; border-radius: 10px; padding: 20px; margin: 10px 0;">
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr style="background-color: #f8f9fa;">
                    <th style="padding: 10px; text-align: left;">Metric</th>
                    <th style="padding: 10px; text-align: center;">Current</th>
                    <th style="padding: 10px; text-align: center;">Target</th>
                    <th style="padding: 10px; text-align: center;">Status</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for metric_name, current_value in metrics.items():
        target_value = targets.get(metric_name) if targets else None
        
        # Determine status color
        if target_value is not None:
            if current_value >= target_value:
                status_color = "🟢"
                status_text = "On Track"
            elif current_value >= target_value * 0.8:
                status_color = "🟡"
                status_text = "At Risk"
            else:
                status_color = "🔴"
                status_text = "Behind"
        else:
            status_color = "⚪"
            status_text = "N/A"
        
        # Format values
        formatted_current = f"{current_value:.2f}" if isinstance(current_value, float) else f"{current_value:,}"
        formatted_target = f"{target_value:.2f}" if target_value and isinstance(target_value, float) else f"{target_value:,}" if target_value else "N/A"
        
        html_content += f"""
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{metric_name}</td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: center; font-weight: bold;">{formatted_current}</td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: center;">{formatted_target}</td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: center;">{status_color} {status_text}</td>
                </tr>
        """
    
    html_content += """
            </tbody>
        </table>
    </div>
    """
    
    st.markdown(html_content, unsafe_allow_html=True)

def create_trend_indicators(data: Dict[str, List[float]], 
                          labels: List[str],
                          title: str = "Trend Indicators") -> None:
    """
    Create trend indicators showing direction of change
    
    Args:
        data: Dictionary with metric names and historical values
        labels: Labels for the time periods
        title: Title for the indicators
    """
    st.markdown(f"### {title}")
    
    cols = st.columns(len(data))
    
    for i, (metric_name, values) in enumerate(data.items()):
        with cols[i]:
            if len(values) >= 2:
                current_value = values[-1]
                previous_value = values[-2]
                change = current_value - previous_value
                change_pct = (change / previous_value * 100) if previous_value != 0 else 0
                
                # Determine trend direction
                if change > 0:
                    trend_icon = "📈"
                    trend_color = "🟢"
                elif change < 0:
                    trend_icon = "📉"
                    trend_color = "🔴"
                else:
                    trend_icon = "➡️"
                    trend_color = "🟡"
                
                # Create mini trend chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=values,
                    mode='lines+markers',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=4)
                ))
                fig.update_layout(
                    height=100,
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                    <h4>{metric_name}</h4>
                    <p style="font-size: 24px; margin: 5px 0;">{current_value:.2f}</p>
                    <p style="margin: 5px 0;">{trend_color} {change_pct:+.1f}%</p>
                    <p style="font-size: 20px;">{trend_icon}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def create_comparison_metrics(current_metrics: Dict[str, float],
                            comparison_metrics: Dict[str, float],
                            comparison_label: str = "Previous Period") -> None:
    """
    Create comparison metrics showing current vs previous period
    
    Args:
        current_metrics: Current period metrics
        comparison_metrics: Comparison period metrics
        comparison_label: Label for comparison period
    """
    st.markdown(f"### 📊 Current vs {comparison_label}")
    
    # Create comparison table
    comparison_data = []
    for metric_name in current_metrics.keys():
        current_value = current_metrics.get(metric_name, 0)
        previous_value = comparison_metrics.get(metric_name, 0)
        
        if previous_value != 0:
            change = current_value - previous_value
            change_pct = (change / previous_value) * 100
        else:
            change = current_value
            change_pct = 0
        
        comparison_data.append({
            'Metric': metric_name,
            'Current': current_value,
            comparison_label: previous_value,
            'Change': change,
            'Change %': f"{change_pct:+.1f}%"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Style the dataframe
    def highlight_change(row):
        if row['Change'] > 0:
            return [''] * 3 + ['background-color: #d4edda'] + ['']
        elif row['Change'] < 0:
            return [''] * 3 + ['background-color: #f8d7da'] + ['']
        else:
            return [''] * 5
    
    styled_df = comparison_df.style.apply(highlight_change, axis=1)
    st.dataframe(styled_df, use_container_width=True)

def create_gauge_metrics(metrics: Dict[str, Dict[str, float]], 
                        title: str = "Performance Gauges") -> None:
    """
    Create gauge charts for performance metrics
    
    Args:
        metrics: Dictionary with metric names and gauge properties
        title: Title for the gauges section
    """
    st.markdown(f"### {title}")
    
    cols = st.columns(len(metrics))
    
    for i, (metric_name, gauge_data) in enumerate(metrics.items()):
        with cols[i]:
            value = gauge_data.get('value', 0)
            max_value = gauge_data.get('max_value', 100)
            target = gauge_data.get('target', max_value * 0.8)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                title={'text': metric_name},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, max_value]},
                    'bar': {'color': "#1f77b4"},
                    'steps': [
                        {'range': [0, max_value * 0.5], 'color': "lightgray"},
                        {'range': [max_value * 0.5, max_value * 0.8], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': target
                    }
                }
            ))
            
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def create_summary_metrics_card(data: pd.DataFrame, 
                              title: str = "Data Summary") -> None:
    """
    Create summary metrics card for dataset overview
    
    Args:
        data: DataFrame to summarize
        title: Title for the summary card
    """
    st.markdown(f"### {title}")
    
    # Calculate summary metrics
    total_records = len(data)
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    missing_values = data.isnull().sum().sum()
    missing_percentage = (missing_values / (len(data) * len(data.columns))) * 100
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        create_metric_card(
            title="Total Records",
            value=total_records,
            format_func=lambda x: f"{x:,}",
            icon="📊"
        )
    
    with col2:
        create_metric_card(
            title="Columns",
            value=len(data.columns),
            icon="📋"
        )
    
    with col3:
        create_metric_card(
            title="Numeric Cols",
            value=len(numeric_columns),
            icon="🔢"
        )
    
    with col4:
        create_metric_card(
            title="Text Cols",
            value=len(categorical_columns),
            icon="📝"
        )
    
    with col5:
        create_metric_card(
            title="Missing Data",
            value=missing_percentage,
            format_func=lambda x: f"{x:.1f}%",
            delta_color="inverse",
            icon="❓"
        )

def format_currency(value: float) -> str:
    """Format value as currency"""
    return f"${value:,.2f}"

def format_percentage(value: float) -> str:
    """Format value as percentage"""
    return f"{value:.1%}"

def format_number(value: Union[int, float]) -> str:
    """Format number with appropriate commas"""
    if isinstance(value, float):
        return f"{value:,.2f}"
    return f"{value:,}"

def format_large_number(value: Union[int, float]) -> str:
    """Format large numbers with K, M, B suffixes"""
    if abs(value) >= 1_000_000_000:
        return f"{value/1_000_000_000:.1f}B"
    elif abs(value) >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"{value/1_000:.1f}K"
    else:
        return f"{value:.0f}"