# streamlit_app/pages/01_📊_Dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from utils.common import load_sample_data
from components.metrics_cards import create_financial_metrics_card, create_customer_metrics_card, create_kpi_grid
from components.charts import create_kpi_chart, create_trend_chart, create_distribution_chart
from components.sidebar import create_main_sidebar, apply_filters_to_dataframe, show_filter_summary

# Page configuration
st.set_page_config(
    page_title="Dashboard - Retail Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dashboard
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 0.5rem 0;
    }
    .insight-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .kpi-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 2px solid #e9ecef;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the customer data"""
    try:
        data_path = Path("data/raw/customer_shopping_data.csv")
        if data_path.exists():
            df = pd.read_csv(data_path)
        else:
            df = load_sample_data(n_customers=2000)
        
        # Data preprocessing for dashboard
        if 'Purchase Date' in df.columns:
            df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
            df['Year'] = df['Purchase Date'].dt.year
            df['Month'] = df['Purchase Date'].dt.month
            df['Quarter'] = df['Purchase Date'].dt.quarter
            df['Weekday'] = df['Purchase Date'].dt.day_name()
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return load_sample_data(n_customers=2000)

def calculate_kpis(df):
    """Calculate key performance indicators"""
    kpis = {}
    
    # Basic metrics
    kpis['total_customers'] = df['Customer ID'].nunique() if 'Customer ID' in df.columns else len(df)
    kpis['total_transactions'] = len(df)
    kpis['total_revenue'] = df['Purchase Amount (USD)'].sum() if 'Purchase Amount (USD)' in df.columns else 0
    kpis['avg_order_value'] = df['Purchase Amount (USD)'].mean() if 'Purchase Amount (USD)' in df.columns else 0
    
    # Advanced metrics
    if 'Purchase Amount (USD)' in df.columns and 'Customer ID' in df.columns:
        customer_stats = df.groupby('Customer ID')['Purchase Amount (USD)'].agg(['sum', 'count']).reset_index()
        kpis['avg_customer_value'] = customer_stats['sum'].mean()
        kpis['avg_transactions_per_customer'] = customer_stats['count'].mean()
        
        # Calculate monthly recurring metrics
        if 'Purchase Date' in df.columns:
            monthly_stats = df.groupby([df['Purchase Date'].dt.to_period('M')])['Purchase Amount (USD)'].sum()
            kpis['monthly_growth_rate'] = ((monthly_stats.iloc[-1] - monthly_stats.iloc[0]) / monthly_stats.iloc[0] * 100) if len(monthly_stats) > 1 else 0
    
    # Customer satisfaction
    if 'Review Rating' in df.columns:
        kpis['avg_rating'] = df['Review Rating'].mean()
        kpis['satisfaction_rate'] = (df['Review Rating'] >= 4.0).mean() * 100
    
    # Category performance
    if 'Category' in df.columns:
        kpis['total_categories'] = df['Category'].nunique()
        top_category = df['Category'].value_counts().index[0]
        kpis['top_category'] = top_category
        kpis['top_category_share'] = (df['Category'] == top_category).mean() * 100
    
    return kpis

def create_revenue_trend_chart(df):
    """Create revenue trend over time"""
    if 'Purchase Date' not in df.columns or 'Purchase Amount (USD)' not in df.columns:
        return None
    
    # Daily revenue
    daily_revenue = df.groupby(df['Purchase Date'].dt.date)['Purchase Amount (USD)'].sum().reset_index()
    daily_revenue.columns = ['Date', 'Revenue']
    
    fig = px.line(daily_revenue, x='Date', y='Revenue', 
                  title='Daily Revenue Trend',
                  labels={'Revenue': 'Revenue ($)', 'Date': 'Date'})
    
    fig.update_traces(line=dict(width=3))
    fig.update_layout(
        height=400,
        hovermode='x unified',
        showlegend=False
    )
    
    return fig

def create_category_performance_chart(df):
    """Create category performance chart"""
    if 'Category' not in df.columns:
        return None
    
    if 'Purchase Amount (USD)' in df.columns:
        category_stats = df.groupby('Category').agg({
            'Purchase Amount (USD)': ['sum', 'count', 'mean'],
            'Customer ID': 'nunique'
        }).round(2)
        
        category_stats.columns = ['Total_Revenue', 'Transaction_Count', 'Avg_Transaction', 'Unique_Customers']
        category_stats = category_stats.sort_values('Total_Revenue', ascending=True)
        
        fig = px.bar(
            y=category_stats.index,
            x=category_stats['Total_Revenue'],
            orientation='h',
            title='Revenue by Category',
            labels={'x': 'Total Revenue ($)', 'y': 'Category'}
        )
        
        fig.update_layout(height=500)
        return fig
    else:
        category_counts = df['Category'].value_counts()
        fig = px.pie(values=category_counts.values, names=category_counts.index,
                     title='Transaction Distribution by Category')
        return fig

def create_customer_distribution_chart(df):
    """Create customer distribution chart"""
    charts = []
    
    # Age distribution
    if 'Age' in df.columns:
        fig1 = px.histogram(df, x='Age', nbins=20, title='Customer Age Distribution')
        fig1.update_layout(height=300)
        charts.append(('Age Distribution', fig1))
    
    # Gender distribution
    if 'Gender' in df.columns:
        gender_counts = df['Gender'].value_counts()
        fig2 = px.pie(values=gender_counts.values, names=gender_counts.index,
                     title='Customer Gender Distribution')
        fig2.update_layout(height=300)
        charts.append(('Gender Distribution', fig2))
    
    # Location distribution
    if 'Location' in df.columns:
        location_counts = df['Location'].value_counts().head(10)
        fig3 = px.bar(x=location_counts.values, y=location_counts.index,
                     orientation='h', title='Top 10 Locations')
        fig3.update_layout(height=300)
        charts.append(('Location Distribution', fig3))
    
    return charts

def display_executive_summary(kpis):
    """Display executive summary"""
    st.markdown("## 📈 Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="kpi-card">
            <h3 style="color: #1f77b4; margin: 0;">💰 Revenue</h3>
            <h2 style="margin: 0.5rem 0;">${:,.2f}</h2>
            <p style="margin: 0; color: #666;">Total Revenue</p>
        </div>
        """.format(kpis['total_revenue']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="kpi-card">
            <h3 style="color: #ff7f0e; margin: 0;">👥 Customers</h3>
            <h2 style="margin: 0.5rem 0;">{:,}</h2>
            <p style="margin: 0; color: #666;">Total Customers</p>
        </div>
        """.format(kpis['total_customers']), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="kpi-card">
            <h3 style="color: #2ca02c; margin: 0;">🛒 AOV</h3>
            <h2 style="margin: 0.5rem 0;">${:.2f}</h2>
            <p style="margin: 0; color: #666;">Avg Order Value</p>
        </div>
        """.format(kpis['avg_order_value']), unsafe_allow_html=True)
    
    with col4:
        rating = kpis.get('avg_rating', 0)
        st.markdown("""
        <div class="kpi-card">
            <h3 style="color: #d62728; margin: 0;">⭐ Rating</h3>
            <h2 style="margin: 0.5rem 0;">{:.1f}/5.0</h2>
            <p style="margin: 0; color: #666;">Avg Rating</p>
        </div>
        """.format(rating), unsafe_allow_html=True)

def display_business_insights(df, kpis):
    """Display key business insights"""
    st.markdown("## 💡 Key Business Insights")
    
    insights = []
    
    # Revenue insights
    if kpis['total_revenue'] > 0:
        insights.append({
            'icon': '💰',
            'title': 'Revenue Performance',
            'content': f"Total revenue of ${kpis['total_revenue']:,.2f} with an average order value of ${kpis['avg_order_value']:.2f}."
        })
    
    # Customer insights
    if kpis['total_customers'] > 0:
        avg_transactions = kpis.get('avg_transactions_per_customer', 0)
        insights.append({
            'icon': '👥',
            'title': 'Customer Engagement',
            'content': f"{kpis['total_customers']:,} unique customers with an average of {avg_transactions:.1f} transactions per customer."
        })
    
    # Category insights
    if 'top_category' in kpis:
        insights.append({
            'icon': '🏷️',
            'title': 'Category Performance',
            'content': f"{kpis['top_category']} is the leading category, representing {kpis['top_category_share']:.1f}% of all transactions."
        })
    
    # Satisfaction insights
    if 'satisfaction_rate' in kpis:
        insights.append({
            'icon': '⭐',
            'title': 'Customer Satisfaction',
            'content': f"{kpis['satisfaction_rate']:.1f}% of customers rate their experience 4 stars or higher (avg: {kpis['avg_rating']:.1f})."
        })
    
    # Growth insights
    if 'monthly_growth_rate' in kpis and abs(kpis['monthly_growth_rate']) > 1:
        growth_direction = "growth" if kpis['monthly_growth_rate'] > 0 else "decline"
        insights.append({
            'icon': '📈' if kpis['monthly_growth_rate'] > 0 else '📉',
            'title': 'Growth Trend',
            'content': f"Monthly revenue shows {abs(kpis['monthly_growth_rate']):.1f}% {growth_direction} trend."
        })
    
    # Display insights
    for insight in insights:
        st.markdown(f"""
        <div class="insight-card">
            <h4>{insight['icon']} {insight['title']}</h4>
            <p style="margin: 0; font-size: 1.1rem;">{insight['content']}</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main dashboard function"""
    st.markdown('<h1 class="main-header">📊 Retail Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df is None or df.empty:
        st.error("No data available for analysis.")
        return
    
    # Sidebar filters
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        filters = create_main_sidebar(df)
    
    # Apply filters
    filtered_df = apply_filters_to_dataframe(df, filters)
    
    # Show filter summary
    with st.sidebar:
        show_filter_summary(df, filtered_df)
    
    # Calculate KPIs
    kpis = calculate_kpis(filtered_df)
    
    # Executive Summary
    display_executive_summary(kpis)
    
    # Main dashboard content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Revenue trend chart
        revenue_chart = create_revenue_trend_chart(filtered_df)
        if revenue_chart:
            st.plotly_chart(revenue_chart, use_container_width=True)
        
        # Category performance
        category_chart = create_category_performance_chart(filtered_df)
        if category_chart:
            st.plotly_chart(category_chart, use_container_width=True)
    
    with col2:
        # Customer distribution charts
        customer_charts = create_customer_distribution_chart(filtered_df)
        for title, chart in customer_charts:
            st.plotly_chart(chart, use_container_width=True)
    
    # Business insights
    display_business_insights(filtered_df, kpis)
    
    # Additional metrics section
    st.markdown("## 📊 Additional Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'Purchase Date' in filtered_df.columns:
            date_range = filtered_df['Purchase Date'].max() - filtered_df['Purchase Date'].min()
            st.metric("Analysis Period", f"{date_range.days} days")
        
        if 'Category' in filtered_df.columns:
            st.metric("Product Categories", f"{filtered_df['Category'].nunique()}")
    
    with col2:
        if 'Location' in filtered_df.columns:
            st.metric("Markets Served", f"{filtered_df['Location'].nunique()}")
        
        repeat_customers = 0
        if 'Customer ID' in filtered_df.columns:
            customer_purchases = filtered_df['Customer ID'].value_counts()
            repeat_customers = (customer_purchases > 1).sum()
            st.metric("Repeat Customers", f"{repeat_customers:,}")
    
    with col3:
        if 'Purchase Amount (USD)' in filtered_df.columns:
            median_order = filtered_df['Purchase Amount (USD)'].median()
            st.metric("Median Order Value", f"${median_order:.2f}")
        
        if 'Review Rating' in filtered_df.columns:
            high_ratings = (filtered_df['Review Rating'] >= 4.0).sum()
            st.metric("High Ratings (4+)", f"{high_ratings:,}")
    
    # Data quality indicators
    with st.expander("📋 Data Quality Summary"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            completeness = (1 - filtered_df.isnull().sum().sum() / (len(filtered_df) * len(filtered_df.columns))) * 100
            st.metric("Data Completeness", f"{completeness:.1f}%")
        
        with col2:
            duplicates = filtered_df.duplicated().sum()
            st.metric("Duplicate Records", f"{duplicates:,}")
        
        with col3:
            data_points = len(filtered_df) * len(filtered_df.columns)
            st.metric("Total Data Points", f"{data_points:,}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        📊 Retail Customer Analytics Dashboard | 
        Last Updated: {} | 
        📧 Contact: analytics@company.com
    </div>
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M')), unsafe_allow_html=True)

if __name__ == "__main__":
    main()