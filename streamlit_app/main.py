# streamlit_app/main.py

import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path

# Ensure project root is added
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import custom components
from src.utils.common import load_config
from src.utils.visualization import create_sidebar
from src.components.data_preprocessing import DataPreprocessor

from src.components.customer_segmentation import CustomerSegmentation

# Page configuration
st.set_page_config(
    page_title="Retail Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    with open("streamlit_app/assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Main application
def main():
    """Main Streamlit application"""
    
    # Load configuration
    config = load_config()
    
    # Load custom CSS
    load_css()
    
    # Sidebar
    st.sidebar.image("streamlit_app/assets/images/logo.png", width=200)
    st.sidebar.title("🛍️ Retail Analytics")
    st.sidebar.markdown("---")
    
    # Main content
    st.title("📊 Retail Customer Analytics Dashboard")
    st.markdown("""
    Welcome to the comprehensive retail customer analytics platform. This dashboard provides 
    insights into customer behavior, segmentation, and predictive analytics to enhance the 
    retail customer experience.
    """)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Customers",
            value="12,847",
            delta="8.2%"
        )
    
    with col2:
        st.metric(
            label="Average Order Value",
            value="$127.45",
            delta="5.1%"
        )
    
    with col3:
        st.metric(
            label="Customer Retention",
            value="73.2%",
            delta="2.8%"
        )
    
    with col4:
        st.metric(
            label="Customer Satisfaction",
            value="4.2/5.0",
            delta="0.3"
        )
    
    # Quick overview
    st.markdown("## 🎯 Quick Overview")
    
    overview_col1, overview_col2 = st.columns(2)
    
    with overview_col1:
        st.markdown("""
        ### 📈 Analytics Capabilities
        - **Customer Segmentation**: RFM analysis and behavioral clustering
        - **Predictive Modeling**: Churn prediction and purchase forecasting
        - **Recommendation Engine**: Personalized product suggestions
        - **Performance Tracking**: KPI monitoring and trend analysis
        """)
    
    with overview_col2:
        st.markdown("""
        ### 🔧 Key Features
        - **Real-time Dashboards**: Interactive visualizations
        - **Advanced EDA**: Comprehensive exploratory data analysis
        - **Model Performance**: Detailed evaluation metrics
        - **Business Insights**: Actionable recommendations
        """)
    
    # Navigation guide
    st.markdown("## 🧭 Navigation Guide")
    
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    
    with nav_col1:
        st.markdown("""
        ### 📊 Dashboard
        High-level overview of key metrics, trends, and performance indicators
        """)
    
    with nav_col2:
        st.markdown("""
        ### 🔍 EDA Insights
        Detailed exploratory data analysis with interactive visualizations
        """)
    
    with nav_col3:
        st.markdown("""
        ### 👥 Customer Segments
        Customer segmentation analysis and behavioral patterns
        """)
    
    # Recent activity
    st.markdown("## 📋 Recent Activity")
    
    activity_data = {
        "Timestamp": ["2025-09-01 21:15", "2025-09-01 21:00", "2025-09-01 20:45"],
        "Activity": [
            "Model retraining completed",
            "New customer segment identified",
            "Churn prediction updated"
        ],
        "Status": ["✅ Completed", "✅ Completed", "✅ Completed"]
    }
    
    st.dataframe(
        pd.DataFrame(activity_data),
        use_container_width=True,
        hide_index=True
    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Retail Customer Analytics Dashboard v1.0.0 | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()