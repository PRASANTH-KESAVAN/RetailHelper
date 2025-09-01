# streamlit_app/pages/05_💡_Recommendations.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from utils.common import load_sample_data
from components.recommendation_engine import RecommendationEngine
from components.customer_segmentation import CustomerSegmentation

# Page configuration
st.set_page_config(
    page_title="Recommendations - Retail Analytics",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
    }
    .recommendation-item {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        background-color: #28a745;
        color: white;
        border-radius: 1rem;
        font-size: 0.8rem;
        margin-left: 1rem;
    }
    .method-comparison {
        background-color: #e3f2fd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the sample data"""
    try:
        data_path = Path("data/raw/customer_shopping_data.csv")
        if data_path.exists():
            df = pd.read_csv(data_path)
        else:
            df = load_sample_data(n_customers=1000)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return load_sample_data(n_customers=1000)

@st.cache_data
def setup_recommendation_system(df):
    """Setup and prepare recommendation system"""
    try:
        # Initialize recommendation engine
        rec_engine = RecommendationEngine({})
        
        # Prepare user-item matrix
        user_item_matrix = rec_engine.prepare_user_item_matrix(
            df, 
            user_col='Customer ID',
            item_col='Category', 
            rating_col='Review Rating'
        )
        
        # Get sample users for demonstration
        sample_users = list(rec_engine.user_mapping.keys())[:50] if rec_engine.user_mapping else []
        
        return {
            'rec_engine': rec_engine,
            'user_item_matrix': user_item_matrix,
            'sample_users': sample_users,
            'setup_success': True
        }
        
    except Exception as e:
        st.error(f"Error setting up recommendation system: {e}")
        return {'setup_success': False}

def display_recommendation_overview(rec_system, df):
    """Display recommendation system overview"""
    if not rec_system['setup_success']:
        return
    
    rec_engine = rec_system['rec_engine']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        n_users = len(rec_engine.user_mapping) if rec_engine.user_mapping else 0
        st.markdown(f"""
        <div class="recommendation-card">
            <h3>Total Users</h3>
            <h2>{n_users:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        n_items = len(rec_engine.item_mapping) if rec_engine.item_mapping else 0
        st.markdown(f"""
        <div class="recommendation-card">
            <h3>Total Items</h3>
            <h2>{n_items}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        matrix_shape = rec_engine.user_item_matrix.shape if rec_engine.user_item_matrix is not None else (0, 0)
        density = (rec_engine.user_item_matrix.nnz / (matrix_shape[0] * matrix_shape[1]) * 100) if matrix_shape[0] > 0 and matrix_shape[1] > 0 else 0
        st.markdown(f"""
        <div class="recommendation-card">
            <h3>Matrix Density</h3>
            <h2>{density:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_rating = df['Review Rating'].mean() if 'Review Rating' in df.columns else 0
        st.markdown(f"""
        <div class="recommendation-card">
            <h3>Avg Rating</h3>
            <h2>{avg_rating:.1f}/5</h2>
        </div>
        """, unsafe_allow_html=True)

def generate_customer_recommendations(rec_system, customer_id, n_recommendations=5):
    """Generate recommendations for a specific customer"""
    if not rec_system['setup_success']:
        return None
    
    rec_engine = rec_system['rec_engine']
    
    try:
        # Try different recommendation methods
        recommendations = {}
        
        # Collaborative Filtering
        try:
            collab_recs = rec_engine.collaborative_filtering_item_based(
                customer_id, n_recommendations
            )
            recommendations['Collaborative Filtering'] = collab_recs
        except Exception as e:
            recommendations['Collaborative Filtering'] = []
        
        # Matrix Factorization
        try:
            matrix_recs = rec_engine.matrix_factorization_recommendations(
                customer_id, n_recommendations, n_components=25
            )
            recommendations['Matrix Factorization'] = matrix_recs
        except Exception as e:
            recommendations['Matrix Factorization'] = []
        
        return recommendations
        
    except Exception as e:
        st.error(f"Error generating recommendations for {customer_id}: {e}")
        return None

def display_recommendations_for_customer(recommendations, customer_id):
    """Display recommendations for a specific customer"""
    st.subheader(f"🎯 Recommendations for Customer: {customer_id}")
    
    if not recommendations:
        st.warning("No recommendations available for this customer.")
        return
    
    # Create tabs for different methods
    methods = list(recommendations.keys())
    if len(methods) == 1:
        method = methods[0]
        recs = recommendations[method]
        display_recommendation_list(recs, method)
    else:
        tabs = st.tabs(methods)
        for i, method in enumerate(methods):
            with tabs[i]:
                recs = recommendations[method]
                display_recommendation_list(recs, method)

def display_recommendation_list(recommendations, method):
    """Display a list of recommendations"""
    if not recommendations:
        st.info(f"No recommendations available using {method}")
        return
    
    st.markdown(f"### {method} Recommendations")
    
    for i, (item, score) in enumerate(recommendations, 1):
        st.markdown(f"""
        <div class="recommendation-item">
            <strong>#{i}. {item}</strong>
            <span class="score-badge">Score: {score:.3f}</span>
            <br>
            <small>Recommended based on {method.lower()} analysis</small>
        </div>
        """, unsafe_allow_html=True)

def create_popularity_analysis(df):
    """Create item popularity analysis"""
    if 'Category' not in df.columns:
        return None
    
    # Calculate item popularity metrics
    item_stats = df.groupby('Category').agg({
        'Customer ID': 'nunique',
        'Review Rating': 'mean',
        'Purchase Amount (USD)': ['sum', 'mean', 'count']
    }).round(2)
    
    item_stats.columns = ['Unique_Customers', 'Avg_Rating', 'Total_Revenue', 'Avg_Purchase', 'Total_Orders']
    item_stats = item_stats.reset_index()
    item_stats['Popularity_Score'] = (
        (item_stats['Unique_Customers'] / item_stats['Unique_Customers'].max()) * 0.4 +
        (item_stats['Avg_Rating'] / 5.0) * 0.3 +
        (item_stats['Total_Orders'] / item_stats['Total_Orders'].max()) * 0.3
    )
    
    item_stats = item_stats.sort_values('Popularity_Score', ascending=False)
    
    return item_stats

def display_item_popularity(item_stats):
    """Display item popularity analysis"""
    if item_stats is None:
        st.warning("Item popularity data not available.")
        return
    
    st.subheader("📊 Item Popularity Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top items by popularity score
        top_items = item_stats.head(10)
        fig = px.bar(
            top_items,
            x='Popularity_Score',
            y='Category',
            orientation='h',
            title='Top 10 Most Popular Items',
            labels={'Popularity_Score': 'Popularity Score', 'Category': 'Product Category'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Revenue vs Rating scatter
        fig = px.scatter(
            item_stats,
            x='Avg_Rating',
            y='Total_Revenue',
            size='Total_Orders',
            hover_data=['Category'],
            title='Revenue vs Rating Analysis',
            labels={
                'Avg_Rating': 'Average Rating',
                'Total_Revenue': 'Total Revenue ($)',
                'Total_Orders': 'Total Orders'
            }
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    with st.expander("📋 Detailed Item Statistics"):
        st.dataframe(
            item_stats.style.format({
                'Avg_Rating': '{:.2f}',
                'Total_Revenue': '${:,.2f}',
                'Avg_Purchase': '${:.2f}',
                'Total_Orders': '{:,}',
                'Popularity_Score': '{:.3f}'
            })
        )

def create_recommendation_performance_analysis(df, rec_system):
    """Create recommendation system performance analysis"""
    if not rec_system['setup_success']:
        return None
    
    st.subheader("📈 Recommendation System Performance")
    
    # Mock performance metrics (in real implementation, use actual evaluation)
    methods = ['Collaborative Filtering', 'Matrix Factorization', 'Content-Based', 'Hybrid']
    
    # Generate mock performance data
    np.random.seed(42)
    performance_data = {
        'Method': methods,
        'Precision': np.random.uniform(0.15, 0.35, len(methods)),
        'Recall': np.random.uniform(0.10, 0.30, len(methods)),
        'F1_Score': np.random.uniform(0.12, 0.32, len(methods)),
        'Coverage': np.random.uniform(0.60, 0.85, len(methods))
    }
    
    performance_df = pd.DataFrame(performance_data)
    
    # Performance comparison chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Precision', 'Recall', 'F1-Score', 'Coverage'],
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    fig.add_trace(
        go.Bar(x=performance_df['Method'], y=performance_df['Precision'],
               marker_color=colors[0], name='Precision'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=performance_df['Method'], y=performance_df['Recall'],
               marker_color=colors[1], name='Recall'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=performance_df['Method'], y=performance_df['F1_Score'],
               marker_color=colors[2], name='F1-Score'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(x=performance_df['Method'], y=performance_df['Coverage'],
               marker_color=colors[3], name='Coverage'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance comparison table
    with st.expander("📊 Detailed Performance Metrics"):
        st.dataframe(
            performance_df.style.format({
                'Precision': '{:.3f}',
                'Recall': '{:.3f}',
                'F1_Score': '{:.3f}',
                'Coverage': '{:.3f}'
            })
        )

def segment_based_recommendations(df):
    """Generate segment-based recommendations"""
    st.subheader("👥 Segment-Based Recommendations")
    
    # Mock segment recommendations (in real implementation, use actual segments)
    segments = ['High Value', 'Frequent Buyers', 'At-Risk', 'New Customers', 'Price Sensitive']
    
    segment_recs = {
        'High Value': ['Premium Electronics', 'Luxury Clothing', 'Home Automation'],
        'Frequent Buyers': ['Books', 'Accessories', 'Personal Care'],
        'At-Risk': ['Discount Items', 'Popular Categories', 'Bundle Offers'],
        'New Customers': ['Best Sellers', 'Starter Kits', 'Welcome Bundles'],
        'Price Sensitive': ['Sale Items', 'Budget Options', 'Value Packs']
    }
    
    for segment, recommendations in segment_recs.items():
        with st.expander(f"🎯 {segment} Customers"):
            st.markdown(f"**Recommended Strategy for {segment}:**")
            
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"""
                <div class="recommendation-item">
                    <strong>{i}. {rec}</strong>
                    <br>
                    <small>Tailored for {segment.lower()} customer behavior</small>
                </div>
                """, unsafe_allow_html=True)

def main():
    """Main function for recommendations page"""
    st.title("💡 Product Recommendation System")
    
    st.markdown("""
    Discover personalized product recommendations powered by advanced machine learning algorithms.
    Our recommendation system analyzes customer behavior patterns to suggest the most relevant products.
    """)
    
    # Load data
    with st.spinner("Loading customer data..."):
        df = load_data()
    
    if df is None or df.empty:
        st.error("No data available for recommendations.")
        return
    
    # Setup recommendation system
    with st.spinner("Setting up recommendation system..."):
        rec_system = setup_recommendation_system(df)
    
    if not rec_system['setup_success']:
        st.error("Failed to setup recommendation system.")
        return
    
    # Display overview
    st.subheader("🎯 Recommendation System Overview")
    display_recommendation_overview(rec_system, df)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "👤 Individual Recommendations", 
        "📊 Item Popularity", 
        "📈 System Performance",
        "👥 Segment Recommendations"
    ])
    
    with tab1:
        st.subheader("🔍 Get Personalized Recommendations")
        
        # Customer selection
        sample_users = rec_system.get('sample_users', [])
        if sample_users:
            selected_customer = st.selectbox(
                "Select a customer to get recommendations:",
                options=sample_users[:20],  # Limit for performance
                format_func=lambda x: f"Customer {x}"
            )
            
            n_recommendations = st.slider("Number of recommendations:", 1, 10, 5)
            
            if st.button("🎯 Generate Recommendations"):
                with st.spinner("Generating personalized recommendations..."):
                    recommendations = generate_customer_recommendations(
                        rec_system, selected_customer, n_recommendations
                    )
                    
                    if recommendations:
                        display_recommendations_for_customer(recommendations, selected_customer)
                    else:
                        st.warning(f"Could not generate recommendations for customer {selected_customer}")
        else:
            st.warning("No customers available for recommendations.")
    
    with tab2:
        item_stats = create_popularity_analysis(df)
        display_item_popularity(item_stats)
    
    with tab3:
        create_recommendation_performance_analysis(df, rec_system)
    
    with tab4:
        segment_based_recommendations(df)
    
    # Sidebar information
    with st.sidebar:
        st.subheader("🛠️ System Configuration")
        
        if rec_system['setup_success']:
            rec_engine = rec_system['rec_engine']
            
            st.success("✅ Recommendation System Active")
            
            if rec_engine.user_item_matrix is not None:
                matrix_shape = rec_engine.user_item_matrix.shape
                st.write(f"**Matrix Size**: {matrix_shape[0]} × {matrix_shape[1]}")
                
                sparsity = 1 - (rec_engine.user_item_matrix.nnz / (matrix_shape[0] * matrix_shape[1]))
                st.write(f"**Sparsity**: {sparsity:.2%}")
            
            st.write(f"**Available Users**: {len(rec_system.get('sample_users', []))}")
            
        else:
            st.error("❌ System Not Available")
        
        st.subheader("📋 Recommendation Methods")
        methods_info = {
            "Collaborative Filtering": "Based on user-item interactions",
            "Matrix Factorization": "Advanced latent factor analysis", 
            "Content-Based": "Based on item characteristics",
            "Hybrid": "Combination of multiple methods"
        }
        
        for method, description in methods_info.items():
            st.write(f"**{method}**: {description}")
        
        st.markdown("---")
        st.markdown("💡 **Tip**: Try different customers to see how recommendations vary based on their purchase history!")

if __name__ == "__main__":
    main()