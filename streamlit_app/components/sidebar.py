# streamlit_app/components/sidebar.py

import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path

def create_main_sidebar(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create the main sidebar with filters and navigation
    
    Args:
        df: Input dataframe for creating filters
        
    Returns:
        Dictionary of filter values
    """
    st.sidebar.header("🎛️ Dashboard Controls")
    
    filters = {}
    
    # Data range info
    st.sidebar.markdown("### 📊 Data Overview")
    st.sidebar.info(f"**Total Records**: {len(df):,}")
    
    if 'Purchase Date' in df.columns:
        min_date = pd.to_datetime(df['Purchase Date']).min()
        max_date = pd.to_datetime(df['Purchase Date']).max()
        st.sidebar.info(f"**Date Range**: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    
    # Filters section
    st.sidebar.markdown("### 🔍 Filters")
    
    # Age filter
    if 'Age' in df.columns:
        age_range = st.sidebar.slider(
            "Age Range",
            min_value=int(df['Age'].min()),
            max_value=int(df['Age'].max()),
            value=(int(df['Age'].min()), int(df['Age'].max())),
            key="age_filter"
        )
        filters['age_range'] = age_range
    
    # Gender filter
    if 'Gender' in df.columns:
        gender_options = df['Gender'].unique().tolist()
        selected_genders = st.sidebar.multiselect(
            "Gender",
            options=gender_options,
            default=gender_options,
            key="gender_filter"
        )
        filters['gender'] = selected_genders
    
    # Location filter
    if 'Location' in df.columns:
        location_options = df['Location'].unique().tolist()
        selected_locations = st.sidebar.multiselect(
            "Location",
            options=location_options,
            default=location_options,
            key="location_filter"
        )
        filters['location'] = selected_locations
    
    # Category filter
    if 'Category' in df.columns:
        category_options = df['Category'].unique().tolist()
        selected_categories = st.sidebar.multiselect(
            "Product Category",
            options=category_options,
            default=category_options,
            key="category_filter"
        )
        filters['category'] = selected_categories
    
    # Purchase amount filter
    if 'Purchase Amount (USD)' in df.columns:
        amount_range = st.sidebar.slider(
            "Purchase Amount Range ($)",
            min_value=float(df['Purchase Amount (USD)'].min()),
            max_value=float(df['Purchase Amount (USD)'].max()),
            value=(float(df['Purchase Amount (USD)'].min()), float(df['Purchase Amount (USD)'].max())),
            key="amount_filter"
        )
        filters['amount_range'] = amount_range
    
    # Review rating filter
    if 'Review Rating' in df.columns:
        rating_range = st.sidebar.slider(
            "Review Rating Range",
            min_value=float(df['Review Rating'].min()),
            max_value=float(df['Review Rating'].max()),
            value=(float(df['Review Rating'].min()), float(df['Review Rating'].max())),
            step=0.1,
            key="rating_filter"
        )
        filters['rating_range'] = rating_range
    
    return filters

def create_analytics_sidebar():
    """Create sidebar for analytics configuration"""
    st.sidebar.header("⚙️ Analytics Settings")
    
    settings = {}
    
    # Model settings
    st.sidebar.markdown("### 🤖 Model Configuration")
    settings['random_state'] = st.sidebar.number_input("Random State", value=42, min_value=0)
    settings['test_size'] = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    settings['cv_folds'] = st.sidebar.slider("Cross Validation Folds", 3, 10, 5)
    
    # Clustering settings
    st.sidebar.markdown("### 🎯 Clustering Settings")
    settings['max_clusters'] = st.sidebar.slider("Max Clusters", 2, 15, 10)
    settings['clustering_method'] = st.sidebar.selectbox(
        "Clustering Method",
        options=['K-Means', 'Hierarchical', 'DBSCAN'],
        index=0
    )
    
    # Recommendation settings
    st.sidebar.markdown("### 💡 Recommendation Settings")
    settings['n_recommendations'] = st.sidebar.slider("Number of Recommendations", 1, 20, 5)
    settings['rec_method'] = st.sidebar.selectbox(
        "Recommendation Method",
        options=['Collaborative Filtering', 'Content-Based', 'Matrix Factorization', 'Hybrid'],
        index=0
    )
    
    return settings

def create_data_info_sidebar(df: pd.DataFrame):
    """Create sidebar with data information"""
    st.sidebar.header("📈 Data Statistics")
    
    # Basic statistics
    st.sidebar.markdown("### 📊 Dataset Overview")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.metric("Rows", f"{len(df):,}")
        st.metric("Columns", len(df.columns))
    
    with col2:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Missing %", f"{missing_pct:.1f}%")
        
        duplicates = df.duplicated().sum()
        st.metric("Duplicates", duplicates)
    
    # Column information
    st.sidebar.markdown("### 📋 Column Types")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    
    st.sidebar.write(f"**Numeric**: {len(numeric_cols)}")
    st.sidebar.write(f"**Categorical**: {len(categorical_cols)}")
    st.sidebar.write(f"**DateTime**: {len(datetime_cols)}")
    
    # Key metrics
    if 'Purchase Amount (USD)' in df.columns:
        st.sidebar.markdown("### 💰 Financial Metrics")
        total_revenue = df['Purchase Amount (USD)'].sum()
        avg_order_value = df['Purchase Amount (USD)'].mean()
        
        st.sidebar.metric("Total Revenue", f"${total_revenue:,.2f}")
        st.sidebar.metric("Avg Order Value", f"${avg_order_value:.2f}")
    
    if 'Customer ID' in df.columns:
        st.sidebar.markdown("### 👥 Customer Metrics")
        unique_customers = df['Customer ID'].nunique()
        avg_orders_per_customer = len(df) / unique_customers if unique_customers > 0 else 0
        
        st.sidebar.metric("Unique Customers", f"{unique_customers:,}")
        st.sidebar.metric("Avg Orders/Customer", f"{avg_orders_per_customer:.1f}")

def create_download_sidebar(df: pd.DataFrame, filtered_df: pd.DataFrame = None):
    """Create sidebar with download options"""
    st.sidebar.header("📥 Download Data")
    
    # Determine which dataset to offer for download
    download_df = filtered_df if filtered_df is not None else df
    
    st.sidebar.markdown(f"**Records to Download**: {len(download_df):,}")
    
    # CSV download
    csv_data = download_df.to_csv(index=False)
    st.sidebar.download_button(
        label="📄 Download as CSV",
        data=csv_data,
        file_name=f"retail_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # JSON download
    json_data = download_df.to_json(orient='records', date_format='iso')
    st.sidebar.download_button(
        label="🔗 Download as JSON",
        data=json_data,
        file_name=f"retail_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def create_navigation_sidebar():
    """Create navigation sidebar"""
    st.sidebar.header("🧭 Navigation")
    
    pages_info = {
        "📊 Dashboard": "Overview of key metrics and KPIs",
        "🔍 EDA Insights": "Detailed exploratory data analysis",
        "👥 Customer Segments": "Customer segmentation and profiling",
        "🔮 Predictions": "Predictive modeling and forecasting",
        "💡 Recommendations": "Product recommendation system"
    }
    
    for page, description in pages_info.items():
        st.sidebar.markdown(f"**{page}**")
        st.sidebar.caption(description)
        st.sidebar.markdown("---")

def create_help_sidebar():
    """Create help and information sidebar"""
    st.sidebar.header("❓ Help & Information")
    
    with st.sidebar.expander("📖 How to Use"):
        st.write("""
        **Dashboard Navigation:**
        - Use the sidebar filters to refine your data view
        - Switch between pages using the navigation menu
        - Download data using the download buttons
        
        **Key Features:**
        - Interactive charts and visualizations
        - Real-time filtering and analysis
        - Advanced analytics and predictions
        - Customer segmentation insights
        """)
    
    with st.sidebar.expander("🔧 Troubleshooting"):
        st.write("""
        **Common Issues:**
        
        1. **No data showing**: Check if filters are too restrictive
        2. **Charts not loading**: Try refreshing the page
        3. **Predictions not available**: Ensure sufficient data is present
        4. **Downloads failing**: Check your browser's download settings
        
        **Performance Tips:**
        - Use filters to reduce data size for better performance
        - Clear browser cache if experiencing issues
        """)
    
    with st.sidebar.expander("📞 Support"):
        st.write("""
        **Need Help?**
        
        - 📧 Email: support@retailanalytics.com
        - 📱 Phone: +1-555-RETAIL
        - 🌐 Documentation: docs.retailanalytics.com
        - 💬 Live Chat: Available 9 AM - 5 PM EST
        """)

def apply_filters_to_dataframe(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply filters to dataframe based on sidebar selections
    
    Args:
        df: Input dataframe
        filters: Dictionary of filter values
        
    Returns:
        Filtered dataframe
    """
    filtered_df = df.copy()
    
    try:
        # Apply age filter
        if 'age_range' in filters and 'Age' in filtered_df.columns:
            age_min, age_max = filters['age_range']
            filtered_df = filtered_df[filtered_df['Age'].between(age_min, age_max)]
        
        # Apply gender filter
        if 'gender' in filters and 'Gender' in filtered_df.columns:
            if filters['gender']:  # Check if not empty
                filtered_df = filtered_df[filtered_df['Gender'].isin(filters['gender'])]
        
        # Apply location filter
        if 'location' in filters and 'Location' in filtered_df.columns:
            if filters['location']:
                filtered_df = filtered_df[filtered_df['Location'].isin(filters['location'])]
        
        # Apply category filter
        if 'category' in filters and 'Category' in filtered_df.columns:
            if filters['category']:
                filtered_df = filtered_df[filtered_df['Category'].isin(filters['category'])]
        
        # Apply amount filter
        if 'amount_range' in filters and 'Purchase Amount (USD)' in filtered_df.columns:
            amount_min, amount_max = filters['amount_range']
            filtered_df = filtered_df[filtered_df['Purchase Amount (USD)'].between(amount_min, amount_max)]
        
        # Apply rating filter
        if 'rating_range' in filters and 'Review Rating' in filtered_df.columns:
            rating_min, rating_max = filters['rating_range']
            filtered_df = filtered_df[filtered_df['Review Rating'].between(rating_min, rating_max)]
        
        return filtered_df
        
    except Exception as e:
        st.error(f"Error applying filters: {e}")
        return df

def show_filter_summary(original_df: pd.DataFrame, filtered_df: pd.DataFrame):
    """Show summary of applied filters"""
    if len(filtered_df) != len(original_df):
        reduction_pct = (1 - len(filtered_df) / len(original_df)) * 100
        
        st.sidebar.markdown("### 🎯 Filter Impact")
        st.sidebar.success(f"""
        **Original Records**: {len(original_df):,}  
        **Filtered Records**: {len(filtered_df):,}  
        **Reduction**: {reduction_pct:.1f}%
        """)
        
        if len(filtered_df) == 0:
            st.sidebar.warning("⚠️ No data matches current filters. Try adjusting your selections.")
        elif len(filtered_df) < 10:
            st.sidebar.warning("⚠️ Very few records match filters. Consider broadening your selection.")

def create_advanced_filters_sidebar(df: pd.DataFrame) -> Dict[str, Any]:
    """Create advanced filtering options"""
    st.sidebar.header("🔬 Advanced Filters")
    
    advanced_filters = {}
    
    # Date range filter
    if 'Purchase Date' in df.columns:
        df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
        min_date = df['Purchase Date'].min().date()
        max_date = df['Purchase Date'].max().date()
        
        date_range = st.sidebar.date_input(
            "Purchase Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="date_filter"
        )
        
        if len(date_range) == 2:
            advanced_filters['date_range'] = date_range
    
    # Statistical filters
    st.sidebar.markdown("### 📊 Statistical Filters")
    
    # Outlier removal
    remove_outliers = st.sidebar.checkbox(
        "Remove Outliers",
        help="Remove statistical outliers based on IQR method"
    )
    advanced_filters['remove_outliers'] = remove_outliers
    
    # Minimum transaction filter
    if 'Purchase Amount (USD)' in df.columns:
        min_transaction = st.sidebar.number_input(
            "Minimum Transaction Amount ($)",
            min_value=0.0,
            value=0.0,
            step=5.0
        )
        advanced_filters['min_transaction'] = min_transaction
    
    # Customer frequency filter
    if 'Customer ID' in df.columns:
        min_purchases = st.sidebar.number_input(
            "Minimum Customer Purchases",
            min_value=1,
            value=1,
            help="Filter customers with at least N purchases"
        )
        advanced_filters['min_purchases'] = min_purchases
    
    return advanced_filters

def display_sidebar_metrics(df: pd.DataFrame):
    """Display key metrics in sidebar"""
    st.sidebar.header("📊 Key Metrics")
    
    # Customer metrics
    if 'Customer ID' in df.columns:
        unique_customers = df['Customer ID'].nunique()
        st.sidebar.metric(
            "Unique Customers",
            f"{unique_customers:,}",
            delta=f"{len(df) / unique_customers:.1f} orders/customer"
        )
    
    # Revenue metrics
    if 'Purchase Amount (USD)' in df.columns:
        total_revenue = df['Purchase Amount (USD)'].sum()
        avg_order = df['Purchase Amount (USD)'].mean()
        
        st.sidebar.metric("Total Revenue", f"${total_revenue:,.2f}")
        st.sidebar.metric("Average Order", f"${avg_order:.2f}")
    
    # Rating metrics
    if 'Review Rating' in df.columns:
        avg_rating = df['Review Rating'].mean()
        rating_std = df['Review Rating'].std()
        
        st.sidebar.metric(
            "Average Rating",
            f"{avg_rating:.2f}",
            delta=f"±{rating_std:.2f}"
        )
    
    # Category diversity
    if 'Category' in df.columns:
        unique_categories = df['Category'].nunique()
        most_popular = df['Category'].value_counts().index[0]
        
        st.sidebar.metric("Product Categories", unique_categories)
        st.sidebar.info(f"Most Popular: **{most_popular}**")

# Utility functions for sidebar components
def reset_filters():
    """Reset all filters to default values"""
    for key in st.session_state.keys():
        if key.endswith('_filter'):
            del st.session_state[key]

def save_filter_preset(filters: Dict[str, Any], preset_name: str):
    """Save current filter configuration as preset"""
    if 'filter_presets' not in st.session_state:
        st.session_state.filter_presets = {}
    
    st.session_state.filter_presets[preset_name] = filters
    st.sidebar.success(f"Preset '{preset_name}' saved!")

def load_filter_preset(preset_name: str):
    """Load a saved filter preset"""
    if 'filter_presets' in st.session_state and preset_name in st.session_state.filter_presets:
        preset_filters = st.session_state.filter_presets[preset_name]
        
        # Update session state with preset values
        for key, value in preset_filters.items():
            st.session_state[f"{key}_filter"] = value
        
        st.sidebar.success(f"Preset '{preset_name}' loaded!")
        st.experimental_rerun()