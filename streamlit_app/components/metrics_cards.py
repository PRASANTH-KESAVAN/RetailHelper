"""Metrics cards for Streamlit dashboard"""
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
