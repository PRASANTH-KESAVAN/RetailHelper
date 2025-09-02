# streamlit_app/pages/03_👥_Customer_Segments.py

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

from src.utils.common import load_sample_data
from src.components.customer_segmentation import CustomerSegmentation
from streamlit_app.components.charts import create_segmentation_scatter, create_donut_chart
from streamlit_app.components.metric_cards import create_kpi_grid

# Page configuration
st.set_page_config(
    page_title="Customer Segments - Retail Analytics",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .segment-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 2px solid #e9ecef;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    .segment-card:hover {
        transform: translateY(-2px);
        border-color: #1f77b4;
    }
    .segment-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .segment-metric {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 0.5rem 0;
        padding: 0.75rem;
        background: white;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .rfm-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 1rem;
        color: white;
        font-weight: bold;
        margin: 0.25rem;
    }
    .champions { background-color: #28a745; }
    .loyal { background-color: #007bff; }
    .potential { background-color: #ffc107; color: #000; }
    .at-risk { background-color: #fd7e14; }
    .lost { background-color: #dc3545; }
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
def perform_customer_segmentation(df, method='kmeans', n_clusters=5):
    """Perform customer segmentation"""
    try:
        config = {
            'segmentation': {
                'random_state': 42,
                'n_clusters_range': [2, 10],
                'max_iter': 300
            }
        }
        
        segmentation = CustomerSegmentation(config)
        
        # Select features for clustering
        features = segmentation.select_features_for_clustering(df)
        
        if features.empty:
            st.warning("No suitable features found for clustering")
            return None, None, None
        
        # Perform clustering
        if method == 'kmeans':
            cluster_labels = segmentation.perform_kmeans_clustering(features, n_clusters)
        elif method == 'hierarchical':
            cluster_labels = segmentation.perform_hierarchical_clustering(features, n_clusters)
        else:
            cluster_labels = segmentation.perform_kmeans_clustering(features, n_clusters)
        
        # Analyze segments
        segment_analysis = segmentation.analyze_segments(df, cluster_labels)
        segment_profiles = segmentation.create_segment_profiles(segment_analysis)
        
        return cluster_labels, segment_analysis, segment_profiles
        
    except Exception as e:
        st.error(f"Error performing segmentation: {e}")
        return None, None, None

def create_rfm_analysis(df):
    """Create RFM analysis"""
    if not all(col in df.columns for col in ['Customer ID', 'Purchase Amount (USD)']):
        # Create mock RFM data
        unique_customers = df['Customer ID'].nunique() if 'Customer ID' in df.columns else 500
        
        rfm_data = {
            'Customer_ID': [f'CUST_{i:05d}' for i in range(unique_customers)],
            'Recency': np.random.exponential(30, unique_customers),
            'Frequency': np.random.poisson(5, unique_customers),
            'Monetary': np.random.exponential(200, unique_customers)
        }
        rfm_df = pd.DataFrame(rfm_data)
    else:
        # Calculate actual RFM if we have the data
        try:
            current_date = pd.to_datetime('2024-01-01')  # Use a reference date
            
            if 'Purchase Date' in df.columns:
                df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
                
                rfm_df = df.groupby('Customer ID').agg({
                    'Purchase Date': lambda x: (current_date - x.max()).days,
                    'Customer ID': 'count',
                    'Purchase Amount (USD)': 'sum'
                }).round(2)
                
                rfm_df.columns = ['Recency', 'Frequency', 'Monetary']
                rfm_df = rfm_df.reset_index()
            else:
                # Fallback to mock data
                unique_customers = df['Customer ID'].nunique()
                rfm_data = {
                    'Customer_ID': df['Customer ID'].unique(),
                    'Recency': np.random.exponential(30, unique_customers),
                    'Frequency': np.random.poisson(5, unique_customers),
                    'Monetary': np.random.exponential(200, unique_customers)
                }
                rfm_df = pd.DataFrame(rfm_data)
                
        except Exception as e:
            st.error(f"Error calculating RFM: {e}")
            unique_customers = 500
            rfm_data = {
                'Customer_ID': [f'CUST_{i:05d}' for i in range(unique_customers)],
                'Recency': np.random.exponential(30, unique_customers),
                'Frequency': np.random.poisson(5, unique_customers),
                'Monetary': np.random.exponential(200, unique_customers)
            }
            rfm_df = pd.DataFrame(rfm_data)
    
    # Create RFM segments
    rfm_df = create_rfm_segments(rfm_df)
    
    return rfm_df

def create_rfm_segments(df):
    """Create RFM segments based on quintiles"""
    # Create quintiles for each RFM metric
    df['R_Score'] = pd.qcut(df['Recency'].rank(method='first'), 5, labels=[5,4,3,2,1])
    df['F_Score'] = pd.qcut(df['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    df['M_Score'] = pd.qcut(df['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5])
    
    # Convert to integers
    df['R_Score'] = df['R_Score'].astype(int)
    df['F_Score'] = df['F_Score'].astype(int)
    df['M_Score'] = df['M_Score'].astype(int)
    
    # Create RFM segments
    def assign_rfm_segment(row):
        if row['R_Score'] >= 4 and row['F_Score'] >= 4 and row['M_Score'] >= 4:
            return 'Champions'
        elif row['R_Score'] >= 3 and row['F_Score'] >= 3:
            return 'Loyal Customers'
        elif row['R_Score'] >= 3 and row['F_Score'] <= 2:
            return 'Potential Loyalists'
        elif row['R_Score'] <= 2 and row['F_Score'] >= 2:
            return 'At Risk'
        elif row['R_Score'] <= 2:
            return 'Lost Customers'
        else:
            return 'Need Attention'
    
    df['RFM_Segment'] = df.apply(assign_rfm_segment, axis=1)
    
    return df

def display_segment_overview(segment_analysis, segment_profiles):
    """Display segment overview"""
    st.subheader("🎯 Customer Segments Overview")
    
    if not segment_analysis:
        st.warning("No segmentation data available")
        return
    
    # Calculate total customers
    total_customers = sum([stats.get('count', 0) for stats in segment_analysis.values()])
    
    # Create segment summary
    cols = st.columns(len(segment_analysis))
    
    for i, (segment_id, stats) in enumerate(segment_analysis.items()):
        with cols[i]:
            segment_name = segment_profiles.get(segment_id, f"Segment {segment_id}")
            count = stats.get('count', 0)
            percentage = (count / total_customers * 100) if total_customers > 0 else 0
            
            st.markdown(f"""
            <div class="segment-card">
                <div class="segment-header">{segment_name}</div>
                <div class="segment-metric">
                    <span><strong>Customers:</strong></span>
                    <span>{count:,} ({percentage:.1f}%)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

def display_rfm_analysis(rfm_df):
    """Display RFM analysis"""
    st.subheader("💎 RFM Analysis")
    
    # RFM segment distribution
    segment_counts = rfm_df['RFM_Segment'].value_counts()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # RFM segment pie chart
        colors = ['#28a745', '#007bff', '#ffc107', '#fd7e14', '#dc3545', '#6c757d']
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title='RFM Customer Segments',
            color_discrete_sequence=colors
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # RFM segment table
        st.markdown("**Segment Summary:**")
        segment_summary = []
        
        for segment, count in segment_counts.items():
            percentage = (count / len(rfm_df)) * 100
            avg_monetary = rfm_df[rfm_df['RFM_Segment'] == segment]['Monetary'].mean()
            avg_frequency = rfm_df[rfm_df['RFM_Segment'] == segment]['Frequency'].mean()
            
            segment_summary.append({
                'Segment': segment,
                'Count': count,
                'Percentage': f"{percentage:.1f}%",
                'Avg Monetary': f"${avg_monetary:.2f}",
                'Avg Frequency': f"{avg_frequency:.1f}"
            })
        
        summary_df = pd.DataFrame(segment_summary)
        st.dataframe(summary_df, hide_index=True)
    
    # RFM distribution plots
    st.markdown("### 📊 RFM Metrics Distribution")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.histogram(rfm_df, x='Recency', title='Recency Distribution (Days)')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(rfm_df, x='Frequency', title='Frequency Distribution')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = px.histogram(rfm_df, x='Monetary', title='Monetary Distribution ($)')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def display_segment_characteristics(df, cluster_labels, segment_analysis):
    """Display detailed segment characteristics"""
    st.subheader("📋 Segment Characteristics")
    
    if cluster_labels is None or segment_analysis is None:
        st.warning("No segmentation results to display")
        return
    
    # Add cluster labels to dataframe for analysis
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = cluster_labels
    
    # Create detailed analysis for each segment
    for cluster_id, stats in segment_analysis.items():
        with st.expander(f"📊 Segment {cluster_id} Details"):
            cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Key Metrics:**")
                st.write(f"• Customer Count: {stats.get('count', 0):,}")
                
                if 'Purchase Amount (USD)' in cluster_data.columns:
                    avg_spend = cluster_data['Purchase Amount (USD)'].mean()
                    total_spend = cluster_data['Purchase Amount (USD)'].sum()
                    st.write(f"• Average Spend: ${avg_spend:.2f}")
                    st.write(f"• Total Spend: ${total_spend:,.2f}")
                
                if 'Age' in cluster_data.columns:
                    avg_age = cluster_data['Age'].mean()
                    st.write(f"• Average Age: {avg_age:.1f}")
                
                if 'Review Rating' in cluster_data.columns:
                    avg_rating = cluster_data['Review Rating'].mean()
                    st.write(f"• Average Rating: {avg_rating:.1f}/5")
            
            with col2:
                # Category preferences
                if 'Category' in cluster_data.columns:
                    st.markdown("**Top Categories:**")
                    top_categories = cluster_data['Category'].value_counts().head(3)
                    for category, count in top_categories.items():
                        percentage = (count / len(cluster_data)) * 100
                        st.write(f"• {category}: {percentage:.1f}%")
                
                # Demographics
                if 'Gender' in cluster_data.columns:
                    st.markdown("**Demographics:**")
                    gender_dist = cluster_data['Gender'].value_counts(normalize=True) * 100
                    for gender, pct in gender_dist.items():
                        st.write(f"• {gender}: {pct:.1f}%")

def create_segment_comparison_chart(df, cluster_labels):
    """Create segment comparison visualizations"""
    if cluster_labels is None:
        return
    
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = cluster_labels
    
    st.subheader("📈 Segment Comparison")
    
    # Numeric columns for comparison
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            x_axis = st.selectbox("X-Axis", numeric_cols, index=0, key="seg_x")
        
        with col2:
            y_axis = st.selectbox("Y-Axis", numeric_cols, 
                                index=1 if len(numeric_cols) > 1 else 0, key="seg_y")
        
        if x_axis and y_axis and x_axis != y_axis:
            fig = px.scatter(
                df_with_clusters,
                x=x_axis,
                y=y_axis,
                color='Cluster',
                title=f'{y_axis} vs {x_axis} by Segment',
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

def display_segment_recommendations(rfm_df):
    """Display actionable recommendations for each segment"""
    st.subheader("💡 Segment-Based Recommendations")
    
    segment_strategies = {
        'Champions': {
            'emoji': '🏆',
            'strategy': 'Retention & Advocacy',
            'actions': [
                'Invite to exclusive VIP programs',
                'Early access to new products',
                'Request reviews and referrals',
                'Offer loyalty rewards and perks'
            ]
        },
        'Loyal Customers': {
            'emoji': '💙',
            'strategy': 'Relationship Building',
            'actions': [
                'Personalized product recommendations',
                'Birthday and anniversary offers',
                'Cross-sell complementary products',
                'Maintain regular engagement'
            ]
        },
        'Potential Loyalists': {
            'emoji': '🌱',
            'strategy': 'Development & Growth',
            'actions': [
                'Targeted marketing campaigns',
                'Incentives for repeat purchases',
                'Educational content about products',
                'Gradual loyalty program introduction'
            ]
        },
        'At Risk': {
            'emoji': '⚠️',
            'strategy': 'Win-Back Campaigns',
            'actions': [
                'Special discount offers',
                'Personalized re-engagement emails',
                'Customer satisfaction surveys',
                'Address potential pain points'
            ]
        },
        'Lost Customers': {
            'emoji': '😞',
            'strategy': 'Reactivation',
            'actions': [
                'Win-back promotions with deep discounts',
                'Survey to understand departure reasons',
                'Showcase new products and improvements',
                'Consider customer service outreach'
            ]
        }
    }
    
    # Get segments present in data
    present_segments = rfm_df['RFM_Segment'].unique()
    
    for segment in present_segments:
        if segment in segment_strategies:
            strategy = segment_strategies[segment]
            count = (rfm_df['RFM_Segment'] == segment).sum()
            percentage = (count / len(rfm_df)) * 100
            
            with st.expander(f"{strategy['emoji']} {segment} ({count:,} customers - {percentage:.1f}%)"):
                st.markdown(f"**Strategy:** {strategy['strategy']}")
                st.markdown("**Recommended Actions:**")
                
                for action in strategy['actions']:
                    st.markdown(f"• {action}")
                
                # Segment metrics
                segment_data = rfm_df[rfm_df['RFM_Segment'] == segment]
                avg_monetary = segment_data['Monetary'].mean()
                avg_frequency = segment_data['Frequency'].mean()
                avg_recency = segment_data['Recency'].mean()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Monetary", f"${avg_monetary:.2f}")
                with col2:
                    st.metric("Avg Frequency", f"{avg_frequency:.1f}")
                with col3:
                    st.metric("Avg Recency", f"{avg_recency:.0f} days")

def main():
    """Main function for customer segmentation page"""
    st.title("👥 Customer Segmentation Analysis")
    
    st.markdown("""
    Understand your customers better through advanced segmentation techniques. 
    Identify high-value customers, at-risk segments, and opportunities for growth.
    """)
    
    # Load data
    with st.spinner("Loading customer data..."):
        df = load_data()
    
    if df is None or df.empty:
        st.error("No data available for segmentation analysis.")
        return
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Segmentation Settings")
        
        segmentation_method = st.selectbox(
            "Segmentation Method",
            options=['RFM Analysis', 'K-Means Clustering', 'Hierarchical Clustering'],
            help="Choose the method for customer segmentation"
        )
        
        if segmentation_method in ['K-Means Clustering', 'Hierarchical Clustering']:
            n_clusters = st.slider("Number of Clusters", 2, 10, 5)
        
        st.markdown("---")
        st.markdown("### 📊 Data Overview")
        st.info(f"**Total Records**: {len(df):,}")
        if 'Customer ID' in df.columns:
            st.info(f"**Unique Customers**: {df['Customer ID'].nunique():,}")
    
    # Main analysis
    if segmentation_method == 'RFM Analysis':
        # RFM Analysis
        rfm_df = create_rfm_analysis(df)
        display_rfm_analysis(rfm_df)
        display_segment_recommendations(rfm_df)
        
    else:
        # K-Means or Hierarchical Clustering
        method = 'kmeans' if segmentation_method == 'K-Means Clustering' else 'hierarchical'
        
        with st.spinner(f"Performing {segmentation_method.lower()}..."):
            cluster_labels, segment_analysis, segment_profiles = perform_customer_segmentation(
                df, method, n_clusters
            )
        
        if cluster_labels is not None:
            # Display results
            display_segment_overview(segment_analysis, segment_profiles)
            display_segment_characteristics(df, cluster_labels, segment_analysis)
            create_segment_comparison_chart(df, cluster_labels)
            
            # Also show RFM for comparison
            with st.expander("🔍 RFM Analysis Comparison"):
                rfm_df = create_rfm_analysis(df)
                display_rfm_analysis(rfm_df)
        
        else:
            st.error("Segmentation analysis failed. Please check your data and try again.")
    
    # Additional insights
    st.markdown("---")
    st.subheader("🔍 Additional Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Key Takeaways:**
        - Customer segmentation helps identify distinct behavioral patterns
        - Different segments require tailored marketing strategies
        - RFM analysis provides actionable business insights
        - Regular re-segmentation ensures strategies remain relevant
        """)
    
    with col2:
        st.markdown("""
        **Next Steps:**
        - Implement segment-specific marketing campaigns
        - Monitor segment migration over time
        - A/B test different approaches by segment
        - Integrate insights into customer journey mapping
        """)

if __name__ == "__main__":
    main()