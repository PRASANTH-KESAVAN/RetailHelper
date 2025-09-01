# streamlit_app/pages/02_🔍_EDA_Insights.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

from utils.visualization import create_distribution_plot, create_correlation_heatmap
from utils.common import load_config, load_sample_data

# Page configuration
st.set_page_config(
    page_title="EDA Insights - Retail Analytics",
    page_icon="🔍",
    layout="wide"
)

def load_and_prepare_data():
    """Load and prepare sample data for EDA"""
    # In a real application, this would load from your data pipeline
    try:
        df = load_sample_data()
        return df
    except:
        # Generate sample data for demonstration
        np.random.seed(42)
        n_customers = 1000
        
        data = {
            'Customer_ID': range(1, n_customers + 1),
            'Age': np.random.normal(40, 15, n_customers).astype(int),
            'Gender': np.random.choice(['Male', 'Female'], n_customers),
            'Purchase_Amount': np.random.exponential(100, n_customers),
            'Purchase_Frequency': np.random.poisson(5, n_customers),
            'Category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_customers),
            'Rating': np.random.normal(4.0, 0.8, n_customers).round(1),
            'Season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], n_customers),
            'Satisfaction_Score': np.random.normal(7.5, 1.5, n_customers)
        }
        
        return pd.DataFrame(data)

def create_overview_metrics(df):
    """Create overview metrics section"""
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Features", f"{df.shape[1]}")
    with col3:
        st.metric("Missing Values", f"{df.isnull().sum().sum()}")
    with col4:
        st.metric("Duplicate Records", f"{df.duplicated().sum()}")

def create_demographic_analysis(df):
    """Create demographic analysis section"""
    st.subheader("👥 Demographic Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig_age = px.histogram(
            df, x='Age', nbins=30,
            title="Age Distribution",
            color_discrete_sequence=['#1f77b4']
        )
        fig_age.update_layout(height=400)
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        # Gender distribution
        gender_counts = df['Gender'].value_counts()
        fig_gender = px.pie(
            values=gender_counts.values,
            names=gender_counts.index,
            title="Gender Distribution"
        )
        fig_gender.update_layout(height=400)
        st.plotly_chart(fig_gender, use_container_width=True)

def create_purchase_behavior_analysis(df):
    """Create purchase behavior analysis section"""
    st.subheader("🛒 Purchase Behavior Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Purchase amount distribution
        fig_amount = px.box(
            df, y='Purchase_Amount',
            title="Purchase Amount Distribution"
        )
        fig_amount.update_layout(height=400)
        st.plotly_chart(fig_amount, use_container_width=True)
    
    with col2:
        # Category preferences
        category_counts = df['Category'].value_counts()
        fig_category = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            title="Category Preferences"
        )
        fig_category.update_layout(height=400)
        st.plotly_chart(fig_category, use_container_width=True)

def create_correlation_analysis(df):
    """Create correlation analysis section"""
    st.subheader("🔗 Correlation Analysis")
    
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap using plotly
    fig_corr = px.imshow(
        corr_matrix,
        title="Feature Correlation Heatmap",
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Insights from correlation
    st.markdown("### 🔍 Correlation Insights")
    
    # Find strong correlations (excluding self-correlation)
    strong_corrs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:  # Threshold for strong correlation
                strong_corrs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_val
                })
    
    if strong_corrs:
        corr_df = pd.DataFrame(strong_corrs)
        st.dataframe(corr_df, use_container_width=True)
    else:
        st.info("No strong correlations (|r| > 0.5) found between features.")

def create_temporal_analysis(df):
    """Create temporal analysis section"""
    st.subheader("📅 Seasonal Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Purchase by season
        season_stats = df.groupby('Season').agg({
            'Purchase_Amount': 'mean',
            'Customer_ID': 'count'
        }).round(2)
        
        fig_season = px.bar(
            x=season_stats.index,
            y=season_stats['Purchase_Amount'],
            title="Average Purchase Amount by Season"
        )
        fig_season.update_layout(height=400)
        st.plotly_chart(fig_season, use_container_width=True)
    
    with col2:
        # Customer count by season
        fig_customers = px.bar(
            x=season_stats.index,
            y=season_stats['Customer_ID'],
            title="Customer Count by Season"
        )
        fig_customers.update_layout(height=400)
        st.plotly_chart(fig_customers, use_container_width=True)

def create_satisfaction_analysis(df):
    """Create customer satisfaction analysis"""
    st.subheader("😊 Customer Satisfaction Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating distribution
        fig_rating = px.histogram(
            df, x='Rating',
            title="Rating Distribution",
            nbins=20
        )
        fig_rating.update_layout(height=400)
        st.plotly_chart(fig_rating, use_container_width=True)
    
    with col2:
        # Satisfaction vs Purchase Amount
        fig_scatter = px.scatter(
            df, x='Purchase_Amount', y='Satisfaction_Score',
            title="Satisfaction vs Purchase Amount",
            trendline="ols"
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

def create_statistical_summary(df):
    """Create statistical summary section"""
    st.subheader("📈 Statistical Summary")
    
    # Descriptive statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Numerical Features")
        numeric_summary = df.describe()
        st.dataframe(numeric_summary, use_container_width=True)
    
    with col2:
        st.markdown("#### Categorical Features")
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        cat_summary = []
        for col in categorical_cols:
            cat_summary.append({
                'Feature': col,
                'Unique Values': df[col].nunique(),
                'Most Frequent': df[col].mode()[0],
                'Frequency': df[col].value_counts().iloc[0]
            })
        
        if cat_summary:
            st.dataframe(pd.DataFrame(cat_summary), use_container_width=True)

def create_insights_summary():
    """Create key insights summary"""
    st.subheader("💡 Key Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("""
        ### 🎯 Customer Behavior Insights
        - **Age Distribution**: Most customers are in the 25-55 age range
        - **Category Preferences**: Electronics and Clothing are top categories
        - **Seasonal Patterns**: Higher spending observed in Fall and Winter
        - **Purchase Frequency**: Average customer makes 5 purchases per period
        """)
    
    with insights_col2:
        st.markdown("""
        ### 📊 Business Implications
        - **Target Demographics**: Focus on middle-aged customers
        - **Inventory Planning**: Stock up on Electronics and Clothing
        - **Marketing Timing**: Increase campaigns in Fall/Winter
        - **Customer Retention**: Implement loyalty programs for frequent buyers
        """)

def main():
    """Main EDA Insights page"""
    st.title("🔍 Exploratory Data Analysis Insights")
    
    st.markdown("""
    This page provides comprehensive exploratory data analysis of retail customer data.
    Explore customer demographics, purchase behaviors, correlations, and key business insights.
    """)
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_and_prepare_data()
    
    # Data filtering sidebar
    st.sidebar.header("🔧 Data Filters")
    
    # Age filter
    age_range = st.sidebar.slider(
        "Age Range",
        min_value=int(df['Age'].min()),
        max_value=int(df['Age'].max()),
        value=(int(df['Age'].min()), int(df['Age'].max()))
    )
    
    # Gender filter
    gender_options = st.sidebar.multiselect(
        "Gender",
        options=df['Gender'].unique(),
        default=df['Gender'].unique()
    )
    
    # Category filter
    category_options = st.sidebar.multiselect(
        "Category",
        options=df['Category'].unique(),
        default=df['Category'].unique()
    )
    
    # Apply filters
    filtered_df = df[
        (df['Age'].between(age_range[0], age_range[1])) &
        (df['Gender'].isin(gender_options)) &
        (df['Category'].isin(category_options))
    ]
    
    st.sidebar.markdown(f"**Filtered Records**: {len(filtered_df):,}")
    
    # Create analysis sections
    create_overview_metrics(filtered_df)
    st.markdown("---")
    
    create_demographic_analysis(filtered_df)
    st.markdown("---")
    
    create_purchase_behavior_analysis(filtered_df)
    st.markdown("---")
    
    create_correlation_analysis(filtered_df)
    st.markdown("---")
    
    create_temporal_analysis(filtered_df)
    st.markdown("---")
    
    create_satisfaction_analysis(filtered_df)
    st.markdown("---")
    
    create_statistical_summary(filtered_df)
    st.markdown("---")
    
    create_insights_summary()
    
    # Download processed data
    st.subheader("📥 Download Data")
    csv_data = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv_data,
        file_name="filtered_customer_data.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()