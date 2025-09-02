"""
Working Streamlit Dashboard - All imports fixed
"""

import sys
from pathlib import Path

# Setup paths FIRST
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "streamlit_app"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Safe imports
try:
    from src.utils.common import load_sample_data
    from streamlit_app.components.metrics_cards import create_financial_metrics_card, create_kpi_grid
    imports_ok = True
    import_error = None
except Exception as e:
    imports_ok = False
    import_error = str(e)

def main():
    st.set_page_config(
        page_title="Retail Analytics - WORKING", 
        page_icon="✅",
        layout="wide"
    )
    
    st.title("Retail Customer Analytics - WORKING DASHBOARD")
    
    if imports_ok:
        st.success("All imports successful!")
        
        # Load sample data
        with st.spinner("Loading data..."):
            data = load_sample_data(1000)
        
        st.subheader("Key Performance Indicators")
        create_kpi_grid(data)
        
        st.subheader("Sample Charts")
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.histogram(data, x="Purchase Amount (USD)", title="Purchase Amount Distribution")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            category_counts = data["Category"].value_counts()
            fig2 = px.bar(x=category_counts.index, y=category_counts.values, title="Category Distribution")
            st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("Sample Data")
        st.dataframe(data.head(10))
        
    else:
        st.error(f"Imports failed: {import_error}")

if __name__ == "__main__":
    main()
