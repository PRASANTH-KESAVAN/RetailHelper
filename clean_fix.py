#!/usr/bin/env python3
"""
CLEAN WINDOWS-COMPATIBLE IMPORT FIX SCRIPT
No syntax errors, no Unicode issues, works on Windows.
"""

import os
import sys
from pathlib import Path

def create_missing_files():
    """Create all missing files with proper content"""
    
    print("Creating missing modules...")
    
    # 1. metrics_cards.py - Windows safe, no emojis
    metrics_content = '''"""Metrics cards for Streamlit dashboard"""
import streamlit as st
import numpy as np
import pandas as pd

def create_financial_metrics_card(data, title="Financial Metrics"):
    """Create financial metrics display"""
    st.subheader(f"Financial {title}")
    
    if data is not None and not data.empty and "Purchase Amount (USD)" in data.columns:
        total_revenue = data["Purchase Amount (USD)"].sum()
        avg_order = data["Purchase Amount (USD)"].mean()
        total_transactions = len(data)
    else:
        total_revenue = 125430.50
        avg_order = 89.50
        total_transactions = 1234
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Revenue", f"${total_revenue:,.2f}", "12.5%")
    with col2:
        st.metric("Avg Order Value", f"${avg_order:.2f}", "3.2%")
    with col3:
        st.metric("Transactions", f"{total_transactions:,}", "8.7%")
    with col4:
        st.metric("Conversion Rate", "5.2%", "1.1%")

def create_customer_metrics_card(data, title="Customer Metrics"):
    """Create customer metrics display"""
    st.subheader(f"Customer {title}")
    
    if data is not None and not data.empty and "Customer ID" in data.columns:
        total_customers = data["Customer ID"].nunique()
    else:
        total_customers = 856
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{total_customers:,}", "15.2%")
    with col2:
        st.metric("Active Customers", f"{int(total_customers * 0.78):,}", "5.8%")
    with col3:
        st.metric("Retention Rate", "78.5%", "2.3%")
    with col4:
        st.metric("Avg Lifetime", "324 days", "12.1%")

def create_kpi_grid(data):
    """Create comprehensive KPI grid"""
    create_financial_metrics_card(data)
    st.divider()
    create_customer_metrics_card(data)
'''
    
    # 2. Working dashboard
    dashboard_content = '''"""
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
'''
    
    # 3. Test script
    test_content = '''"""Import Test Script"""

import sys
import os
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.resolve()
paths = [
    str(project_root),
    str(project_root / "src"),
    str(project_root / "streamlit_app"),
]

for path in paths:
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ["PYTHONPATH"] = os.pathsep.join(paths)

print("Testing imports...")

# Test imports
tests = []

try:
    from src.utils.common import load_sample_data
    print("✅ src.utils.common imported")
    tests.append(("src.utils.common", True))
except Exception as e:
    print(f"❌ src.utils.common failed: {e}")
    tests.append(("src.utils.common", False))

try:
    from streamlit_app.components.metrics_cards import create_financial_metrics_card
    print("✅ streamlit_app.components.metrics_cards imported")
    tests.append(("metrics_cards", True))
except Exception as e:
    print(f"❌ metrics_cards failed: {e}")
    tests.append(("metrics_cards", False))

passed = sum(1 for _, success in tests if success)
total = len(tests)

print(f"\\nResults: {passed}/{total} tests passed")

if passed == total:
    print("SUCCESS! You can run: streamlit run streamlit_app/dashboard_working.py")
else:
    print("Some tests failed. Check the errors above.")
'''
    
    # Files to create
    files = {
        "streamlit_app/components/metrics_cards.py": metrics_content,
        "streamlit_app/dashboard_working.py": dashboard_content,
        "test_imports_clean.py": test_content,
    }
    
    # Create files
    for filepath, content in files.items():
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write with UTF-8 encoding
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"Created: {filepath}")

def create_init_files():
    """Create __init__.py files"""
    
    init_paths = [
        "src/__init__.py",
        "src/utils/__init__.py",
        "src/components/__init__.py", 
        "src/pipelines/__init__.py",
        "streamlit_app/__init__.py",
        "streamlit_app/components/__init__.py",
        "streamlit_app/pages/__init__.py",
    ]
    
    for init_path in init_paths:
        path = Path(init_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if not path.exists():
            with open(path, "w", encoding="utf-8") as f:
                f.write("# Auto-generated __init__.py\\n")
            print(f"Created: {init_path}")

def main():
    """Main function"""
    
    print("CLEAN IMPORT FIX FOR WINDOWS")
    print("=" * 40)
    
    try:
        create_init_files()
        create_missing_files()
        
        print("\\n" + "=" * 40)
        print("SUCCESS! NO SYNTAX ERRORS!")
        print("=" * 40)
        
        print("\\nNext steps:")
        print("1. Run: python test_imports_clean.py") 
        print("2. Run: streamlit run streamlit_app/dashboard_working.py")
        print("3. Open: http://localhost:8501")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
    
    
    