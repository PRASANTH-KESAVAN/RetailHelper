# streamlit_app/__init__.py

"""
Streamlit web application for retail customer analytics.

This module contains the Streamlit dashboard components including:
- Main dashboard overview
- EDA insights and exploration
- Customer segmentation analysis
- Predictive modeling interface
- Recommendation engine interface
"""

__version__ = "1.0.0"
__author__ = "Data Science Team"

# Dashboard configuration
DASHBOARD_CONFIG = {
    'page_title': 'Retail Customer Analytics Dashboard',
    'page_icon': '📊',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'menu_items': {
        'Get Help': 'https://your-help-url.com',
        'Report a bug': 'https://your-bug-report-url.com',
        'About': 'Retail Customer Analytics Dashboard v1.0.0'
    }
}

# Color scheme
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'warning': '#d62728',
    'info': '#9467bd',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

# Navigation menu
NAVIGATION = {
    '📊 Dashboard': 'pages/01_📊_Dashboard.py',
    '🔍 EDA Insights': 'pages/02_🔍_EDA_Insights.py',
    '👥 Customer Segments': 'pages/03_👥_Customer_Segments.py',
    '🔮 Predictions': 'pages/04_🔮_Predictions.py',
    '💡 Recommendations': 'pages/05_💡_Recommendations.py'
}

__all__ = [
    'DASHBOARD_CONFIG',
    'COLORS',
    'NAVIGATION'
]