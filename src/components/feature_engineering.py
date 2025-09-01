# src/components/feature_engineering.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from datetime import datetime, timedelta
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineering:
    """
    Feature engineering component for creating and selecting features
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
    def create_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create customer-level aggregated features
        
        Args:
            df: Input dataframe with transaction-level data
            
        Returns:
            DataFrame with customer-level features
        """
        logger.info("Creating customer-level features...")
        
        try:
            # Group by Customer ID to create customer-level features
            customer_features = df.groupby('Customer ID').agg({
                # Demographic features
                'Age': 'first',
                'Gender': 'first',
                'Location': 'first',
                
                # Purchase behavior features
                'Purchase Amount (USD)': ['sum', 'mean', 'std', 'min', 'max', 'count'],
                'Review Rating': ['mean', 'std', 'count'],
                'Previous Purchases': 'first',
                
                # Product preferences
                'Category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
                'Season': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
                'Payment Method': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
                
                # Subscription and loyalty features
                'Subscription Status': 'first',
                'Discount Applied': lambda x: (x == 'Yes').sum(),
                'Promo Code Used': lambda x: (x == 'Yes').sum(),
            }).reset_index()
            
            # Flatten column names
            customer_features.columns = [
                'Customer_ID', 'Age', 'Gender', 'Location',
                'Total_Spent', 'Avg_Spent', 'Spending_Std', 'Min_Spent', 'Max_Spent', 'Purchase_Count',
                'Avg_Rating', 'Rating_Std', 'Rating_Count',
                'Previous_Purchases', 'Preferred_Category', 'Preferred_Season', 'Preferred_Payment',
                'Subscription_Status', 'Discount_Count', 'Promo_Count'
            ]
            
            # Handle missing values in std calculations
            customer_features['Spending_Std'].fillna(0, inplace=True)
            customer_features['Rating_Std'].fillna(0, inplace=True)
            
            # Create derived features
            customer_features['Spending_Range'] = customer_features['Max_Spent'] - customer_features['Min_Spent']
            customer_features['Spending_Consistency'] = customer_features['Spending_Std'] / (customer_features['Avg_Spent'] + 1e-6)
            customer_features['Total_Purchases'] = customer_features['Previous_Purchases'] + customer_features['Purchase_Count']
            customer_features['Discount_Rate'] = customer_features['Discount_Count'] / customer_features['Purchase_Count']
            customer_features['Promo_Rate'] = customer_features['Promo_Count'] / customer_features['Purchase_Count']
            
            # Subscription binary encoding
            customer_features['Is_Subscriber'] = (customer_features['Subscription_Status'] == 'Yes').astype(int)
            
            logger.info(f"Customer features created for {len(customer_features)} customers")
            return customer_features
            
        except Exception as e:
            logger.error(f"Error creating customer features: {e}")
            raise
    
    def create_rfm_features(self, df: pd.DataFrame, 
                          customer_id_col: str = 'Customer ID',
                          date_col: str = 'Purchase Date',
                          amount_col: str = 'Purchase Amount (USD)') -> pd.DataFrame:
        """
        Create RFM (Recency, Frequency, Monetary) features
        
        Args:
            df: Input dataframe
            customer_id_col: Customer ID column name
            date_col: Date column name
            amount_col: Amount column name
            
        Returns:
            DataFrame with RFM features
        """
        logger.info("Creating RFM features...")
        
        try:
            # Ensure date column is datetime
            df[date_col] = pd.to_datetime(df[date_col])
            reference_date = df[date_col].max() + pd.Timedelta(days=1)
            
            # Calculate RFM metrics
            rfm = df.groupby(customer_id_col).agg({
                date_col: lambda x: (reference_date - x.max()).days,  # Recency
                customer_id_col: 'count',  # Frequency  
                amount_col: 'sum'  # Monetary
            }).reset_index()
            
            rfm.columns = [customer_id_col, 'Recency', 'Frequency', 'Monetary']
            
            # Create RFM scores using quartiles
            rfm['R_Score'] = pd.qcut(rfm['Recency'].rank(method='first'), 4, labels=[4,3,2,1])
            rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1,2,3,4])
            rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 4, labels=[1,2,3,4])
            
            # Convert to numeric
            rfm['R_Score'] = rfm['R_Score'].astype(int)
            rfm['F_Score'] = rfm['F_Score'].astype(int)
            rfm['M_Score'] = rfm['M_Score'].astype(int)
            
            # Create combined RFM score
            rfm['RFM_Score'] = rfm['R_Score'] * 100 + rfm['F_Score'] * 10 + rfm['M_Score']
            
            # Create RFM segments
            def rfm_segment(row):
                if row['RFM_Score'] >= 444:
                    return 'Champions'
                elif row['RFM_Score'] >= 334:
                    return 'Loyal Customers'
                elif row['RFM_Score'] >= 313:
                    return 'Potential Loyalists'
                elif row['RFM_Score'] >= 212:
                    return 'New Customers'
                elif row['RFM_Score'] >= 142:
                    return 'Promising'
                elif row['RFM_Score'] >= 122:
                    return 'Need Attention'
                elif row['RFM_Score'] >= 113:
                    return 'About to Sleep'
                elif row['RFM_Score'] >= 112:
                    return 'At Risk'
                elif row['RFM_Score'] >= 111:
                    return 'Cannot Lose Them'
                else:
                    return 'Lost'
            
            rfm['RFM_Segment'] = rfm.apply(rfm_segment, axis=1)
            
            # Additional RFM-based features
            rfm['Days_Between_Purchases'] = rfm['Recency'] / rfm['Frequency']
            rfm['Avg_Order_Value'] = rfm['Monetary'] / rfm['Frequency']
            
            # Customer lifetime value estimation (simplified)
            rfm['Estimated_CLV'] = rfm['Avg_Order_Value'] * rfm['Frequency'] * 12  # Annual estimate
            
            logger.info(f"RFM features created for {len(rfm)} customers")
            return rfm
            
        except Exception as e:
            logger.error(f"Error creating RFM features: {e}")
            raise
    
    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create behavioral and preference features
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with behavioral features
        """
        logger.info("Creating behavioral features...")
        
        try:
            behavioral_features = df.groupby('Customer ID').agg({
                # Category diversity
                'Category': 'nunique',
                
                # Seasonal behavior
                'Season': ['nunique', lambda x: x.mode().iloc[0]],
                
                # Payment behavior
                'Payment Method': 'nunique',
                
                # Size and color preferences
                'Size': lambda x: x.mode().iloc[0] if 'Size' in df.columns else 'Unknown',
                'Color': lambda x: x.mode().iloc[0] if 'Color' in df.columns else 'Unknown',
                
                # Shipping preferences
                'Shipping Type': lambda x: x.mode().iloc[0] if 'Shipping Type' in df.columns else 'Standard',
                
                # Review behavior
                'Review Rating': ['count', 'std'],
            }).reset_index()
            
            # Flatten column names
            behavioral_features.columns = [
                'Customer_ID', 'Category_Diversity', 'Season_Diversity', 'Preferred_Season',
                'Payment_Diversity', 'Preferred_Size', 'Preferred_Color', 'Preferred_Shipping',
                'Review_Count', 'Review_Variability'
            ]
            
            # Fill missing values
            behavioral_features['Review_Variability'].fillna(0, inplace=True)
            
            # Create behavioral scores
            behavioral_features['Diversity_Score'] = (
                behavioral_features['Category_Diversity'] + 
                behavioral_features['Season_Diversity'] + 
                behavioral_features['Payment_Diversity']
            ) / 3
            
            # Create loyalty indicators
            behavioral_features['Brand_Loyal'] = (behavioral_features['Category_Diversity'] == 1).astype(int)
            behavioral_features['Season_Consistent'] = (behavioral_features['Season_Diversity'] == 1).astype(int)
            
            logger.info(f"Behavioral features created for {len(behavioral_features)} customers")
            return behavioral_features
            
        except Exception as e:
            logger.error(f"Error creating behavioral features: {e}")
            raise
    
    def create_time_based_features(self, df: pd.DataFrame, date_col: str = 'Purchase Date') -> pd.DataFrame:
        """
        Create time-based features
        
        Args:
            df: Input dataframe
            date_col: Date column name
            
        Returns:
            DataFrame with time-based features
        """
        logger.info("Creating time-based features...")
        
        try:
            # Ensure date column is datetime
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Extract time components
            df['Purchase_Year'] = df[date_col].dt.year
            df['Purchase_Month'] = df[date_col].dt.month
            df['Purchase_Day'] = df[date_col].dt.day
            df['Purchase_DayOfWeek'] = df[date_col].dt.dayofweek
            df['Purchase_Quarter'] = df[date_col].dt.quarter
            df['Purchase_Week'] = df[date_col].dt.isocalendar().week
            
            # Create cyclical features for periodic patterns
            df['Month_Sin'] = np.sin(2 * np.pi * df['Purchase_Month'] / 12)
            df['Month_Cos'] = np.cos(2 * np.pi * df['Purchase_Month'] / 12)
            df['Day_Sin'] = np.sin(2 * np.pi * df['Purchase_DayOfWeek'] / 7)
            df['Day_Cos'] = np.cos(2 * np.pi * df['Purchase_DayOfWeek'] / 7)
            
            # Time-based aggregations by customer
            time_features = df.groupby('Customer ID').agg({
                'Purchase_Month': ['nunique', 'std'],
                'Purchase_DayOfWeek': ['nunique', 'mean'],
                'Purchase_Quarter': 'nunique',
                date_col: ['min', 'max', 'count']
            }).reset_index()
            
            # Flatten column names
            time_features.columns = [
                'Customer_ID', 'Month_Diversity', 'Month_Variability',
                'Day_Diversity', 'Avg_Purchase_Day', 'Quarter_Diversity',
                'First_Purchase', 'Last_Purchase', 'Total_Purchases'
            ]
            
            # Calculate customer tenure
            time_features['Customer_Tenure_Days'] = (
                time_features['Last_Purchase'] - time_features['First_Purchase']
            ).dt.days
            
            # Purchase frequency (purchases per day)
            time_features['Purchase_Frequency_Daily'] = (
                time_features['Total_Purchases'] / (time_features['Customer_Tenure_Days'] + 1)
            )
            
            # Fill missing values
            time_features['Month_Variability'].fillna(0, inplace=True)
            time_features['Customer_Tenure_Days'].fillna(0, inplace=True)
            
            logger.info(f"Time-based features created for {len(time_features)} customers")
            return time_features
            
        except Exception as e:
            logger.error(f"Error creating time-based features: {e}")
            raise
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                  categorical_columns: List[str] = None,
                                  encoding_method: str = 'label') -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df: Input dataframe
            categorical_columns: List of categorical columns to encode
            encoding_method: 'label', 'onehot', or 'target'
            
        Returns:
            DataFrame with encoded features
        """
        logger.info(f"Encoding categorical features using {encoding_method} encoding...")
        
        try:
            df_encoded = df.copy()
            
            if categorical_columns is None:
                categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
                # Remove ID columns
                categorical_columns = [col for col in categorical_columns 
                                     if 'ID' not in col.upper() and 'id' not in col]
            
            if encoding_method == 'label':
                for col in categorical_columns:
                    if col in df_encoded.columns:
                        if col not in self.encoders:
                            self.encoders[col] = LabelEncoder()
                        
                        # Handle missing values
                        df_encoded[col].fillna('Unknown', inplace=True)
                        df_encoded[f'{col}_encoded'] = self.encoders[col].fit_transform(df_encoded[col])
            
            elif encoding_method == 'onehot':
                for col in categorical_columns:
                    if col in df_encoded.columns:
                        # Create dummy variables
                        dummies = pd.get_dummies(df_encoded[col], prefix=col, dummy_na=True)
                        df_encoded = pd.concat([df_encoded, dummies], axis=1)
                        df_encoded.drop(col, axis=1, inplace=True)
            
            logger.info(f"Encoded {len(categorical_columns)} categorical features")
            return df_encoded
            
        except Exception as e:
            logger.error(f"Error encoding categorical features: {e}")
            raise
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                  feature_pairs: List[Tuple[str, str]] = None) -> pd.DataFrame:
        """
        Create interaction features between selected variables
        
        Args:
            df: Input dataframe
            feature_pairs: List of feature pairs to create interactions for
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features...")
        
        try:
            df_interactions = df.copy()
            
            if feature_pairs is None:
                # Default interaction pairs for retail data
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                feature_pairs = [
                    ('Age', 'Total_Spent'),
                    ('Purchase_Count', 'Avg_Rating'),
                    ('Frequency', 'Monetary'),
                    ('Spending_Consistency', 'Category_Diversity')
                ]
                # Filter pairs that exist in the dataframe
                feature_pairs = [(col1, col2) for col1, col2 in feature_pairs 
                               if col1 in numeric_cols and col2 in numeric_cols]
            
            for col1, col2 in feature_pairs:
                if col1 in df_interactions.columns and col2 in df_interactions.columns:
                    # Multiplicative interaction
                    interaction_name = f'{col1}_x_{col2}'
                    df_interactions[interaction_name] = df_interactions[col1] * df_interactions[col2]
                    
                    # Ratio interaction (if col2 is not zero)
                    ratio_name = f'{col1}_div_{col2}'
                    df_interactions[ratio_name] = df_interactions[col1] / (df_interactions[col2] + 1e-6)
            
            logger.info(f"Created interaction features for {len(feature_pairs)} feature pairs")
            return df_interactions
            
        except Exception as e:
            logger.error(f"Error creating interaction features: {e}")
            raise
    
    def select_features(self, X: pd.DataFrame, y: pd.Series = None, 
                       method: str = 'correlation', k: int = 20) -> List[str]:
        """
        Select top k features using specified method
        
        Args:
            X: Feature dataframe
            y: Target variable (optional, required for supervised methods)
            method: Feature selection method ('correlation', 'mutual_info', 'f_score')
            k: Number of features to select
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting top {k} features using {method} method...")
        
        try:
            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            X_numeric = X[numeric_features]
            
            if method == 'correlation' and y is not None:
                # Correlation-based selection
                correlations = X_numeric.corrwith(y).abs().sort_values(ascending=False)
                selected_features = correlations.head(k).index.tolist()
            
            elif method == 'mutual_info' and y is not None:
                # Mutual information-based selection
                selector = SelectKBest(score_func=mutual_info_classif, k=min(k, len(numeric_features)))
                selector.fit(X_numeric.fillna(0), y)
                selected_features = X_numeric.columns[selector.get_support()].tolist()
            
            elif method == 'f_score' and y is not None:
                # F-score based selection
                selector = SelectKBest(score_func=f_classif, k=min(k, len(numeric_features)))
                selector.fit(X_numeric.fillna(0), y)
                selected_features = X_numeric.columns[selector.get_support()].tolist()
            
            else:
                # Variance-based selection (unsupervised)
                variances = X_numeric.var().sort_values(ascending=False)
                selected_features = variances.head(k).index.tolist()
            
            logger.info(f"Selected {len(selected_features)} features: {selected_features}")
            return selected_features
            
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            raise
    
    def merge_features(self, *feature_dfs: pd.DataFrame, on: str = 'Customer_ID') -> pd.DataFrame:
        """
        Merge multiple feature dataframes
        
        Args:
            *feature_dfs: Variable number of feature dataframes
            on: Column to merge on
            
        Returns:
            Merged dataframe with all features
        """
        logger.info(f"Merging {len(feature_dfs)} feature dataframes...")
        
        try:
            if len(feature_dfs) == 0:
                raise ValueError("No dataframes provided for merging")
            
            merged_df = feature_dfs[0].copy()
            
            for df in feature_dfs[1:]:
                if on in df.columns:
                    merged_df = merged_df.merge(df, on=on, how='outer')
                else:
                    logger.warning(f"Column '{on}' not found in dataframe, skipping merge")
            
            logger.info(f"Merged dataframe shape: {merged_df.shape}")
            return merged_df
            
        except Exception as e:
            logger.error(f"Error merging features: {e}")
            raise
    
    def get_feature_importance_summary(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create a summary of feature statistics
        
        Args:
            X: Feature dataframe
            
        Returns:
            DataFrame with feature importance summary
        """
        logger.info("Creating feature importance summary...")
        
        try:
            summary_data = []
            
            for col in X.columns:
                if X[col].dtype in [np.int64, np.float64]:
                    summary_data.append({
                        'feature': col,
                        'type': 'numeric',
                        'missing_pct': X[col].isnull().sum() / len(X) * 100,
                        'unique_values': X[col].nunique(),
                        'mean': X[col].mean(),
                        'std': X[col].std(),
                        'variance': X[col].var(),
                        'skewness': X[col].skew(),
                        'kurtosis': X[col].kurtosis()
                    })
                else:
                    summary_data.append({
                        'feature': col,
                        'type': 'categorical',
                        'missing_pct': X[col].isnull().sum() / len(X) * 100,
                        'unique_values': X[col].nunique(),
                        'most_frequent': X[col].mode().iloc[0] if len(X[col].mode()) > 0 else None
                    })
            
            summary_df = pd.DataFrame(summary_data)
            logger.info("Feature importance summary created")
            return summary_df
            
        except Exception as e:
            logger.error(f"Error creating feature importance summary: {e}")
            raise