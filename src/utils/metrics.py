# src/utils/metrics.py

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

def calculate_rfm_metrics(df: pd.DataFrame, 
                         customer_col: str = 'Customer ID',
                         date_col: str = 'Purchase Date',
                         amount_col: str = 'Purchase Amount (USD)') -> pd.DataFrame:
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics for customers
    
    Args:
        df: Transaction dataframe
        customer_col: Customer ID column name
        date_col: Date column name
        amount_col: Amount column name
        
    Returns:
        DataFrame with RFM metrics
    """
    logger.info("Calculating RFM metrics...")
    
    try:
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        reference_date = df[date_col].max() + pd.Timedelta(days=1)
        
        # Calculate RFM metrics
        rfm = df.groupby(customer_col).agg({
            date_col: lambda x: (reference_date - x.max()).days,  # Recency
            customer_col: 'count',  # Frequency
            amount_col: 'sum'  # Monetary
        }).reset_index()
        
        rfm.columns = [customer_col, 'Recency', 'Frequency', 'Monetary']
        
        # Calculate RFM scores (1-5 scale)
        rfm['R_Score'] = pd.qcut(rfm['Recency'].rank(method='first'), 5, labels=[5,4,3,2,1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5])
        
        # Convert to numeric
        rfm['R_Score'] = rfm['R_Score'].astype(int)
        rfm['F_Score'] = rfm['F_Score'].astype(int)
        rfm['M_Score'] = rfm['M_Score'].astype(int)
        
        # Combined RFM score
        rfm['RFM_Score'] = rfm['R_Score'] * 100 + rfm['F_Score'] * 10 + rfm['M_Score']
        
        logger.info(f"RFM metrics calculated for {len(rfm)} customers")
        return rfm
        
    except Exception as e:
        logger.error(f"Error calculating RFM metrics: {e}")
        raise

def calculate_clv(df: pd.DataFrame, 
                 customer_col: str = 'Customer ID',
                 amount_col: str = 'Purchase Amount (USD)',
                 frequency_col: str = 'Frequency',
                 method: str = 'simple') -> pd.Series:
    """
    Calculate Customer Lifetime Value (CLV)
    
    Args:
        df: Customer dataframe with aggregated metrics
        customer_col: Customer ID column name
        amount_col: Amount column name
        frequency_col: Frequency column name
        method: CLV calculation method ('simple', 'traditional')
        
    Returns:
        Series with CLV values
    """
    logger.info(f"Calculating CLV using {method} method...")
    
    try:
        if method == 'simple':
            # Simple CLV = Average Order Value * Purchase Frequency * Customer Lifespan (assumed 2 years)
            avg_order_value = df[amount_col] / df[frequency_col]
            clv = avg_order_value * df[frequency_col] * 2  # 2-year lifespan assumption
            
        elif method == 'traditional':
            # Traditional CLV = (Average Order Value * Purchase Frequency * Gross Margin) - Customer Acquisition Cost
            # Simplified version without acquisition cost
            avg_order_value = df[amount_col] / df[frequency_col]
            gross_margin = 0.3  # Assumed 30% margin
            clv = (avg_order_value * df[frequency_col] * gross_margin)
            
        else:
            raise ValueError(f"Unknown CLV method: {method}")
        
        logger.info(f"CLV calculated for {len(clv)} customers")
        return clv
        
    except Exception as e:
        logger.error(f"Error calculating CLV: {e}")
        raise

def calculate_churn_probability(df: pd.DataFrame,
                              recency_col: str = 'Recency',
                              frequency_col: str = 'Frequency',
                              monetary_col: str = 'Monetary') -> pd.Series:
    """
    Calculate churn probability based on RFM metrics
    
    Args:
        df: DataFrame with RFM metrics
        recency_col: Recency column name
        frequency_col: Frequency column name
        monetary_col: Monetary column name
        
    Returns:
        Series with churn probabilities
    """
    logger.info("Calculating churn probabilities...")
    
    try:
        # Normalize metrics to 0-1 scale
        recency_norm = (df[recency_col] - df[recency_col].min()) / (df[recency_col].max() - df[recency_col].min())
        frequency_norm = (df[frequency_col] - df[frequency_col].min()) / (df[frequency_col].max() - df[frequency_col].min())
        monetary_norm = (df[monetary_col] - df[monetary_col].min()) / (df[monetary_col].max() - df[monetary_col].min())
        
        # Simple churn probability formula
        # Higher recency = higher churn risk
        # Lower frequency and monetary = higher churn risk
        churn_prob = (0.5 * recency_norm + 
                     0.3 * (1 - frequency_norm) + 
                     0.2 * (1 - monetary_norm))
        
        # Ensure values are between 0 and 1
        churn_prob = np.clip(churn_prob, 0, 1)
        
        logger.info(f"Churn probabilities calculated for {len(churn_prob)} customers")
        return churn_prob
        
    except Exception as e:
        logger.error(f"Error calculating churn probability: {e}")
        raise

def segment_performance_metrics(df: pd.DataFrame, 
                              segment_col: str = 'cluster_label',
                              amount_col: str = 'Purchase Amount (USD)',
                              frequency_col: str = 'Frequency') -> Dict[str, Any]:
    """
    Calculate performance metrics for customer segments
    
    Args:
        df: DataFrame with customer data and segments
        segment_col: Segment label column name
        amount_col: Amount column name
        frequency_col: Frequency column name
        
    Returns:
        Dictionary with segment performance metrics
    """
    logger.info("Calculating segment performance metrics...")
    
    try:
        segment_metrics = {}
        
        for segment in df[segment_col].unique():
            segment_data = df[df[segment_col] == segment]
            
            metrics = {
                'customer_count': len(segment_data),
                'percentage_of_total': len(segment_data) / len(df) * 100,
                'avg_purchase_amount': segment_data[amount_col].mean(),
                'total_revenue': segment_data[amount_col].sum(),
                'avg_frequency': segment_data[frequency_col].mean(),
                'revenue_per_customer': segment_data[amount_col].sum() / len(segment_data)
            }
            
            # Additional metrics if available
            if 'Age' in segment_data.columns:
                metrics['avg_age'] = segment_data['Age'].mean()
            
            if 'Review Rating' in segment_data.columns:
                metrics['avg_rating'] = segment_data['Review Rating'].mean()
            
            segment_metrics[f'segment_{segment}'] = metrics
        
        logger.info(f"Performance metrics calculated for {len(segment_metrics)} segments")
        return segment_metrics
        
    except Exception as e:
        logger.error(f"Error calculating segment performance: {e}")
        raise

def recommendation_metrics(actual_items: List[str], 
                         recommended_items: List[str],
                         k: int = None) -> Dict[str, float]:
    """
    Calculate recommendation system metrics
    
    Args:
        actual_items: List of actual items purchased/liked
        recommended_items: List of recommended items
        k: Top-k recommendations to consider
        
    Returns:
        Dictionary with recommendation metrics
    """
    try:
        if k is not None:
            recommended_items = recommended_items[:k]
        
        # Convert to sets for easier calculation
        actual_set = set(actual_items)
        recommended_set = set(recommended_items)
        
        # Calculate metrics
        intersection = len(actual_set.intersection(recommended_set))
        
        # Precision: Relevant recommended items / Total recommended items
        precision = intersection / len(recommended_set) if recommended_set else 0
        
        # Recall: Relevant recommended items / Total actual items
        recall = intersection / len(actual_set) if actual_set else 0
        
        # F1 Score: Harmonic mean of precision and recall
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Coverage: Percentage of unique items recommended
        coverage = len(recommended_set) / len(recommended_items) if recommended_items else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'coverage': coverage,
            'intersection_size': intersection
        }
        
    except Exception as e:
        logger.error(f"Error calculating recommendation metrics: {e}")
        return {'precision': 0, 'recall': 0, 'f1_score': 0, 'coverage': 0, 'intersection_size': 0}

def business_impact_metrics(df_before: pd.DataFrame, 
                          df_after: pd.DataFrame,
                          amount_col: str = 'Purchase Amount (USD)',
                          frequency_col: str = 'Frequency') -> Dict[str, Any]:
    """
    Calculate business impact metrics comparing before and after scenarios
    
    Args:
        df_before: DataFrame with before scenario
        df_after: DataFrame with after scenario
        amount_col: Amount column name
        frequency_col: Frequency column name
        
    Returns:
        Dictionary with business impact metrics
    """
    logger.info("Calculating business impact metrics...")
    
    try:
        # Revenue metrics
        revenue_before = df_before[amount_col].sum()
        revenue_after = df_after[amount_col].sum()
        revenue_change = ((revenue_after - revenue_before) / revenue_before) * 100
        
        # Frequency metrics
        avg_frequency_before = df_before[frequency_col].mean()
        avg_frequency_after = df_after[frequency_col].mean()
        frequency_change = ((avg_frequency_after - avg_frequency_before) / avg_frequency_before) * 100
        
        # Customer count metrics
        customers_before = len(df_before)
        customers_after = len(df_after)
        customer_change = ((customers_after - customers_before) / customers_before) * 100
        
        # Average order value
        aov_before = df_before[amount_col].mean()
        aov_after = df_after[amount_col].mean()
        aov_change = ((aov_after - aov_before) / aov_before) * 100
        
        impact_metrics = {
            'revenue_change_percent': revenue_change,
            'revenue_change_absolute': revenue_after - revenue_before,
            'frequency_change_percent': frequency_change,
            'customer_change_percent': customer_change,
            'aov_change_percent': aov_change,
            'total_impact_score': (revenue_change + frequency_change + aov_change) / 3
        }
        
        logger.info("Business impact metrics calculated successfully")
        return impact_metrics
        
    except Exception as e:
        logger.error(f"Error calculating business impact metrics: {e}")
        raise

def calculate_market_basket_metrics(df: pd.DataFrame,
                                  customer_col: str = 'Customer ID',
                                  item_col: str = 'Category') -> Dict[str, Any]:
    """
    Calculate market basket analysis metrics
    
    Args:
        df: Transaction dataframe
        customer_col: Customer ID column name
        item_col: Item/Category column name
        
    Returns:
        Dictionary with market basket metrics
    """
    logger.info("Calculating market basket metrics...")
    
    try:
        # Create customer-item matrix
        basket = df.groupby([customer_col, item_col]).size().unstack(fill_value=0)
        basket = basket.applymap(lambda x: 1 if x > 0 else 0)
        
        # Calculate support for each item
        item_support = basket.sum(axis=0) / len(basket)
        
        # Calculate frequent itemsets (items appearing together)
        frequent_pairs = {}
        items = basket.columns
        
        for i, item1 in enumerate(items):
            for item2 in items[i+1:]:
                # Support for the pair
                pair_support = ((basket[item1] == 1) & (basket[item2] == 1)).sum() / len(basket)
                
                if pair_support > 0:
                    # Confidence: P(item2|item1)
                    confidence1 = pair_support / item_support[item1] if item_support[item1] > 0 else 0
                    confidence2 = pair_support / item_support[item2] if item_support[item2] > 0 else 0
                    
                    # Lift
                    lift = pair_support / (item_support[item1] * item_support[item2]) if (item_support[item1] * item_support[item2]) > 0 else 0
                    
                    frequent_pairs[f"{item1}-{item2}"] = {
                        'support': pair_support,
                        'confidence_1_2': confidence1,
                        'confidence_2_1': confidence2,
                        'lift': lift
                    }
        
        market_basket_metrics = {
            'item_support': item_support.to_dict(),
            'frequent_pairs': frequent_pairs,
            'basket_size_avg': basket.sum(axis=1).mean(),
            'total_unique_items': len(items)
        }
        
        logger.info(f"Market basket metrics calculated for {len(items)} items")
        return market_basket_metrics
        
    except Exception as e:
        logger.error(f"Error calculating market basket metrics: {e}")
        raise

def calculate_cohort_metrics(df: pd.DataFrame,
                           customer_col: str = 'Customer ID',
                           date_col: str = 'Purchase Date',
                           amount_col: str = 'Purchase Amount (USD)') -> pd.DataFrame:
    """
    Calculate cohort analysis metrics
    
    Args:
        df: Transaction dataframe
        customer_col: Customer ID column name
        date_col: Date column name
        amount_col: Amount column name
        
    Returns:
        DataFrame with cohort metrics
    """
    logger.info("Calculating cohort metrics...")
    
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Determine customer's first purchase month (cohort)
        df['order_period'] = df[date_col].dt.to_period('M')
        cohort_data = df.groupby(customer_col)[date_col].min().reset_index()
        cohort_data.columns = [customer_col, 'cohort_group']
        cohort_data['cohort_group'] = cohort_data['cohort_group'].dt.to_period('M')
        
        # Merge back to original data
        df_cohort = df.merge(cohort_data, on=customer_col)
        
        # Calculate periods
        df_cohort['period_number'] = (df_cohort['order_period'] - df_cohort['cohort_group']).apply(attrgetter('n'))
        
        # Create cohort table
        cohort_data_revenue = df_cohort.groupby(['cohort_group', 'period_number'])[amount_col].sum().reset_index()
        cohort_sizes = df_cohort.groupby('cohort_group')[customer_col].nunique()
        
        cohort_table_revenue = cohort_data_revenue.pivot(index='cohort_group', 
                                                        columns='period_number', 
                                                        values=amount_col)
        
        # Calculate cohort metrics
        cohort_metrics = {
            'cohort_sizes': cohort_sizes.to_dict(),
            'cohort_table_revenue': cohort_table_revenue.fillna(0),
            'avg_revenue_per_cohort': cohort_table_revenue.mean(axis=1).to_dict()
        }
        
        logger.info(f"Cohort metrics calculated for {len(cohort_sizes)} cohorts")
        return cohort_metrics
        
    except Exception as e:
        logger.error(f"Error calculating cohort metrics: {e}")
        raise

def calculate_retention_metrics(df: pd.DataFrame,
                              customer_col: str = 'Customer ID',
                              date_col: str = 'Purchase Date',
                              period: str = 'M') -> Dict[str, Any]:
    """
    Calculate customer retention metrics
    
    Args:
        df: Transaction dataframe
        customer_col: Customer ID column name
        date_col: Date column name
        period: Time period for analysis ('D', 'M', 'Q')
        
    Returns:
        Dictionary with retention metrics
    """
    logger.info(f"Calculating retention metrics for period: {period}")
    
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Create period column
        df['period'] = df[date_col].dt.to_period(period)
        
        # Get customer activity by period
        customer_periods = df.groupby([customer_col, 'period']).size().unstack(fill_value=0)
        customer_periods = customer_periods.applymap(lambda x: 1 if x > 0 else 0)
        
        # Calculate retention rates
        retention_rates = {}
        periods = customer_periods.columns
        
        for i, current_period in enumerate(periods[:-1]):
            next_period = periods[i + 1]
            
            # Customers active in current period
            current_customers = customer_periods[current_period].sum()
            
            # Customers active in both current and next period
            retained_customers = ((customer_periods[current_period] == 1) & 
                                (customer_periods[next_period] == 1)).sum()
            
            # Retention rate
            retention_rate = retained_customers / current_customers if current_customers > 0 else 0
            retention_rates[f"{current_period}_to_{next_period}"] = retention_rate
        
        # Overall metrics
        total_customers = len(customer_periods)
        active_periods = customer_periods.sum(axis=1)
        
        retention_metrics = {
            'period_retention_rates': retention_rates,
            'avg_retention_rate': np.mean(list(retention_rates.values())),
            'total_customers': total_customers,
            'avg_active_periods_per_customer': active_periods.mean(),
            'customer_lifetime_periods': active_periods.to_dict()
        }
        
        logger.info(f"Retention metrics calculated for {total_customers} customers")
        return retention_metrics
        
    except Exception as e:
        logger.error(f"Error calculating retention metrics: {e}")
        raise

# Additional utility functions
def normalize_metrics(series: pd.Series, method: str = 'min_max') -> pd.Series:
    """
    Normalize metrics series
    
    Args:
        series: Series to normalize
        method: Normalization method ('min_max', 'z_score')
        
    Returns:
        Normalized series
    """
    if method == 'min_max':
        return (series - series.min()) / (series.max() - series.min())
    elif method == 'z_score':
        return (series - series.mean()) / series.std()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def calculate_percentile_ranks(series: pd.Series) -> pd.Series:
    """
    Calculate percentile ranks for a series
    
    Args:
        series: Input series
        
    Returns:
        Series with percentile ranks
    """
    return series.rank(pct=True) * 100