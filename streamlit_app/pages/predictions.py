# streamlit_app/pages/04_🔮_Predictions.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from utils.common import load_sample_data
from components.charts import create_gauge_chart, create_waterfall_chart

# Page configuration
st.set_page_config(
    page_title="Predictions - Retail Analytics",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    .prediction-card:hover {
        transform: scale(1.02);
    }
    .prediction-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .risk-high { 
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); 
    }
    .risk-medium { 
        background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%); 
    }
    .risk-low { 
        background: linear-gradient(135deg, #48cae4 0%, #0077b6 100%); 
    }
    .feature-importance {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .model-metrics {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
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
            df = load_sample_data(n_customers=1000)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return load_sample_data(n_customers=1000)

def prepare_churn_data(df):
    """Prepare data for churn prediction"""
    # Create a synthetic churn label based on business logic
    churn_features = df.copy()
    
    # Feature engineering for churn prediction
    if 'Purchase Date' in df.columns:
        df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
        reference_date = df['Purchase Date'].max()
        churn_features['Days_Since_Last_Purchase'] = (reference_date - df['Purchase Date']).dt.days
    else:
        churn_features['Days_Since_Last_Purchase'] = np.random.exponential(30, len(df))
    
    # Create synthetic churn labels based on recency and other factors
    churn_probability = np.minimum(
        churn_features['Days_Since_Last_Purchase'] / 365,  # Higher days since last purchase
        0.8
    )
    
    if 'Review Rating' in df.columns:
        # Lower ratings increase churn probability
        rating_factor = (5 - df['Review Rating']) / 5
        churn_probability += rating_factor * 0.3
    
    # Generate churn labels
    np.random.seed(42)
    churn_features['Churn'] = np.random.binomial(1, churn_probability)
    
    return churn_features

def prepare_clv_data(df):
    """Prepare data for Customer Lifetime Value prediction"""
    clv_features = df.copy()
    
    # Calculate actual CLV or create synthetic CLV
    if 'Customer ID' in df.columns and 'Purchase Amount (USD)' in df.columns:
        customer_stats = df.groupby('Customer ID').agg({
            'Purchase Amount (USD)': ['sum', 'mean', 'count'],
        })
        customer_stats.columns = ['Total_Spent', 'Avg_Order_Value', 'Purchase_Count']
        
        # Estimate CLV (simplified: AOV * Frequency * Lifespan)
        customer_stats['CLV'] = (
            customer_stats['Avg_Order_Value'] * 
            customer_stats['Purchase_Count'] * 2  # Assume 2-year relationship
        )
        
        # Merge back to main dataframe
        clv_features = df.merge(
            customer_stats[['CLV']], 
            left_on='Customer ID', 
            right_index=True, 
            how='left'
        )
    else:
        # Create synthetic CLV
        base_clv = np.random.exponential(500, len(df))
        clv_features['CLV'] = base_clv
    
    return clv_features

def build_churn_model(df):
    """Build and train churn prediction model"""
    try:
        # Prepare features
        features = []
        
        # Numeric features
        numeric_features = ['Age', 'Purchase Amount (USD)', 'Days_Since_Last_Purchase', 'Review Rating']
        for feature in numeric_features:
            if feature in df.columns:
                features.append(feature)
        
        # Categorical features (encoded)
        categorical_features = ['Gender', 'Category', 'Location']
        encoders = {}
        
        for feature in categorical_features:
            if feature in df.columns:
                encoder = LabelEncoder()
                df[f'{feature}_encoded'] = encoder.fit_transform(df[feature].astype(str))
                features.append(f'{feature}_encoded')
                encoders[feature] = encoder
        
        if len(features) == 0:
            st.error("No suitable features found for churn prediction")
            return None, None, None, None, None
        
        # Prepare data
        X = df[features]
        y = df['Churn']
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        return model, scaler, encoders, (X_test, y_test, y_pred, y_pred_proba), features
        
    except Exception as e:
        st.error(f"Error building churn model: {e}")
        return None, None, None, None, None

def build_clv_model(df):
    """Build and train CLV prediction model"""
    try:
        # Prepare features
        features = []
        
        # Numeric features
        numeric_features = ['Age', 'Purchase Amount (USD)', 'Review Rating']
        for feature in numeric_features:
            if feature in df.columns:
                features.append(feature)
        
        # Categorical features (encoded)
        categorical_features = ['Gender', 'Category', 'Location']
        encoders = {}
        
        for feature in categorical_features:
            if feature in df.columns:
                encoder = LabelEncoder()
                df[f'{feature}_encoded'] = encoder.fit_transform(df[feature].astype(str))
                features.append(f'{feature}_encoded')
                encoders[feature] = encoder
        
        if len(features) == 0:
            st.error("No suitable features found for CLV prediction")
            return None, None, None, None, None
        
        # Prepare data
        X = df[features]
        y = df['CLV']
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        return model, scaler, encoders, (X_test, y_test, y_pred), features
        
    except Exception as e:
        st.error(f"Error building CLV model: {e}")
        return None, None, None, None, None

def display_churn_predictions(model_results, features):
    """Display churn prediction results"""
    if model_results is None:
        st.warning("No churn model results to display")
        return
    
    X_test, y_test, y_pred, y_pred_proba = model_results
    
    st.subheader("🚨 Churn Prediction Results")
    
    # Model performance metrics
    accuracy = (y_pred == y_test).mean()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="model-metrics">
            <h3>Model Accuracy</h3>
            <h2>{accuracy:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        churn_rate = y_test.mean()
        st.markdown(f"""
        <div class="model-metrics">
            <h3>Churn Rate</h3>
            <h2>{churn_rate:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        high_risk_customers = (y_pred_proba > 0.7).sum()
        st.markdown(f"""
        <div class="model-metrics">
            <h3>High Risk Customers</h3>
            <h2>{high_risk_customers}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk distribution
    st.markdown("### 📊 Churn Risk Distribution")
    
    # Create risk categories
    risk_categories = []
    risk_colors = []
    
    for prob in y_pred_proba:
        if prob > 0.7:
            risk_categories.append("High Risk")
            risk_colors.append("#ff6b6b")
        elif prob > 0.4:
            risk_categories.append("Medium Risk")
            risk_colors.append("#feca57")
        else:
            risk_categories.append("Low Risk")
            risk_colors.append("#48cae4")
    
    risk_counts = pd.Series(risk_categories).value_counts()
    
    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title='Customer Churn Risk Distribution',
        color_discrete_sequence=['#48cae4', '#feca57', '#ff6b6b']
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.markdown("### 🎯 Feature Importance")
    
    if hasattr(model_results, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': model_results.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance in Churn Prediction'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def display_clv_predictions(model_results, features):
    """Display CLV prediction results"""
    if model_results is None:
        st.warning("No CLV model results to display")
        return
    
    X_test, y_test, y_pred = model_results
    
    st.subheader("💎 Customer Lifetime Value Predictions")
    
    # Model performance metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="model-metrics">
            <h3>R² Score</h3>
            <h2>{r2:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_clv = y_test.mean()
        st.markdown(f"""
        <div class="model-metrics">
            <h3>Average CLV</h3>
            <h2>${avg_clv:.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="model-metrics">
            <h3>RMSE</h3>
            <h2>${rmse:.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # CLV distribution
    st.markdown("### 📊 CLV Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            x=y_pred,
            title='Predicted CLV Distribution',
            labels={'x': 'Predicted CLV ($)', 'y': 'Frequency'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            x=y_test,
            y=y_pred,
            title='Actual vs Predicted CLV',
            labels={'x': 'Actual CLV ($)', 'y': 'Predicted CLV ($)'}
        )
        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_shape(
            type="line",
            x0=min_val, y0=min_val, x1=max_val, y1=max_val,
            line=dict(color="red", width=2, dash="dash")
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def create_customer_predictor():
    """Create interactive customer predictor"""
    st.subheader("🎯 Individual Customer Prediction")
    
    st.markdown("Enter customer details to get churn probability and CLV estimate:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 80, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        location = st.selectbox("Location", ["New York", "California", "Texas", "Florida", "Illinois"])
    
    with col2:
        purchase_amount = st.number_input("Last Purchase Amount ($)", min_value=0.0, value=100.0)
        category = st.selectbox("Preferred Category", ["Clothing", "Electronics", "Books", "Home & Garden", "Sports"])
        days_since_purchase = st.slider("Days Since Last Purchase", 1, 365, 30)
    
    if st.button("🔮 Make Prediction", type="primary"):
        # Create sample prediction (in a real app, you'd use your trained model)
        
        # Simulate churn probability based on inputs
        churn_prob = min(0.8, days_since_purchase / 365 + np.random.random() * 0.3)
        
        # Simulate CLV based on inputs
        clv_estimate = purchase_amount * (100 / days_since_purchase) * (age / 40) * np.random.uniform(0.8, 1.2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn risk card
            risk_class = "risk-high" if churn_prob > 0.6 else "risk-medium" if churn_prob > 0.3 else "risk-low"
            risk_text = "High Risk" if churn_prob > 0.6 else "Medium Risk" if churn_prob > 0.3 else "Low Risk"
            
            st.markdown(f"""
            <div class="prediction-card {risk_class}">
                <h3>🚨 Churn Risk</h3>
                <div class="prediction-value">{churn_prob:.1%}</div>
                <p>{risk_text}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # CLV estimate card
            st.markdown(f"""
            <div class="prediction-card">
                <h3>💎 Estimated CLV</h3>
                <div class="prediction-value">${clv_estimate:.0f}</div>
                <p>Lifetime Value</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Recommendations based on prediction
        st.markdown("### 💡 Recommendations")
        
        if churn_prob > 0.6:
            st.error("🚨 **High Churn Risk** - Immediate action required!")
            st.markdown("""
            **Recommended Actions:**
            - Send personalized retention offer
            - Schedule customer service call
            - Provide exclusive discount
            - Survey for satisfaction issues
            """)
        elif churn_prob > 0.3:
            st.warning("⚠️ **Medium Churn Risk** - Monitor closely")
            st.markdown("""
            **Recommended Actions:**
            - Send re-engagement campaign
            - Offer loyalty program enrollment
            - Provide product recommendations
            - Monitor purchase behavior
            """)
        else:
            st.success("✅ **Low Churn Risk** - Focus on growth")
            st.markdown("""
            **Recommended Actions:**
            - Cross-sell complementary products
            - Invite to VIP program
            - Request reviews and referrals
            - Increase purchase frequency
            """)

def display_model_insights():
    """Display insights about prediction models"""
    st.subheader("🧠 Model Insights & Best Practices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🚨 Churn Prediction
        
        **Key Factors:**
        - Recency of last purchase
        - Purchase frequency patterns
        - Customer satisfaction scores
        - Engagement with marketing
        
        **Business Impact:**
        - 5x cheaper to retain than acquire
        - Early detection enables intervention
        - Targeted retention campaigns
        - Improved customer satisfaction
        """)
    
    with col2:
        st.markdown("""
        ### 💎 Customer Lifetime Value
        
        **Key Factors:**
        - Historical purchase behavior
        - Average order value
        - Purchase frequency
        - Customer demographics
        
        **Business Impact:**
        - Optimize marketing spend
        - Identify high-value customers
        - Personalize service levels
        - Improve resource allocation
        """)
    
    st.markdown("""
    ### 📈 Model Performance Tips
    
    - **Data Quality**: Ensure clean, consistent data for better predictions
    - **Feature Engineering**: Create meaningful features from raw data
    - **Regular Updates**: Retrain models with new data regularly
    - **A/B Testing**: Validate model performance with controlled experiments
    - **Monitoring**: Track model performance and data drift over time
    """)

def main():
    """Main function for predictions page"""
    st.title("🔮 Predictive Analytics")
    
    st.markdown("""
    Leverage machine learning to predict customer behavior, identify risks, and estimate value.
    Get actionable insights to drive business decisions and improve customer retention.
    """)
    
    # Load data
    with st.spinner("Loading data and training models..."):
        df = load_data()
        
        # Prepare data for different prediction tasks
        churn_df = prepare_churn_data(df)
        clv_df = prepare_clv_data(df)
    
    if df is None or df.empty:
        st.error("No data available for predictive analysis.")
        return
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Prediction Settings")
        
        prediction_type = st.selectbox(
            "Analysis Type",
            options=['Overview', 'Churn Prediction', 'Customer Lifetime Value', 'Individual Predictor'],
            help="Choose the type of predictive analysis"
        )
        
        if prediction_type in ['Churn Prediction', 'Customer Lifetime Value']:
            auto_train = st.checkbox("Auto-train Models", value=True)
        
        st.markdown("---")
        st.markdown("### 📊 Data Overview")
        st.info(f"**Total Records**: {len(df):,}")
        if 'Customer ID' in df.columns:
            st.info(f"**Unique Customers**: {df['Customer ID'].nunique():,}")
    
    # Main content based on selection
    if prediction_type == 'Overview':
        display_model_insights()
        
        # Quick stats
        st.markdown("### 📊 Prediction Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Customers Analyzed", f"{len(df):,}")
        
        with col2:
            if 'Churn' in churn_df.columns:
                churn_rate = churn_df['Churn'].mean()
                st.metric("Predicted Churn Rate", f"{churn_rate:.1%}")
        
        with col3:
            if 'CLV' in clv_df.columns:
                avg_clv = clv_df['CLV'].mean()
                st.metric("Average CLV", f"${avg_clv:.0f}")
        
        with col4:
            st.metric("Models Available", "2", help="Churn and CLV models")
    
    elif prediction_type == 'Churn Prediction':
        if auto_train:
            with st.spinner("Training churn prediction model..."):
                model, scaler, encoders, model_results, features = build_churn_model(churn_df)
            
            if model is not None:
                display_churn_predictions(model_results, features)
            else:
                st.error("Failed to train churn prediction model")
        else:
            st.info("Enable 'Auto-train Models' in the sidebar to see churn predictions")
    
    elif prediction_type == 'Customer Lifetime Value':
        if auto_train:
            with st.spinner("Training CLV prediction model..."):
                model, scaler, encoders, model_results, features = build_clv_model(clv_df)
            
            if model is not None:
                display_clv_predictions(model_results, features)
            else:
                st.error("Failed to train CLV prediction model")
        else:
            st.info("Enable 'Auto-train Models' in the sidebar to see CLV predictions")
    
    elif prediction_type == 'Individual Predictor':
        create_customer_predictor()
    
    # Additional information
    with st.expander("ℹ️ About Predictive Models"):
        st.markdown("""
        **Churn Prediction Model:**
        - Uses Random Forest classifier
        - Predicts probability of customer churning
        - Based on purchase behavior and demographics
        - Helps identify at-risk customers for retention
        
        **Customer Lifetime Value Model:**
        - Uses Random Forest regressor
        - Estimates total customer value over time
        - Considers purchase history and patterns
        - Helps prioritize customer service and marketing
        
        **Note:** These are demonstration models using sample data. 
        In production, models would be trained on historical data with proper validation.
        """)

if __name__ == "__main__":
    main()