# streamlit_app/components/data_tables.py

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode, DataReturnMode
import plotly.express as px
import plotly.graph_objects as go

def create_interactive_table(df: pd.DataFrame, 
                           title: str = "Data Table",
                           key: Optional[str] = None,
                           height: int = 400,
                           enable_selection: bool = True,
                           enable_filtering: bool = True,
                           enable_sorting: bool = True,
                           pagination: bool = True,
                           page_size: int = 20) -> pd.DataFrame:
    """
    Create an interactive data table with AgGrid
    
    Args:
        df: DataFrame to display
        title: Table title
        key: Unique key for the component
        height: Table height in pixels
        enable_selection: Enable row selection
        enable_filtering: Enable column filtering
        enable_sorting: Enable column sorting
        pagination: Enable pagination
        page_size: Rows per page
        
    Returns:
        DataFrame with selected rows (if selection enabled)
    """
    st.subheader(title)
    
    # Configure grid options
    gb = GridOptionsBuilder.from_dataframe(df)
    
    if enable_selection:
        gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren=True, 
                              groupSelectsFiltered=True)
    
    if enable_filtering:
        gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
        gb.configure_side_bar()
    
    if enable_sorting:
        gb.configure_default_column(sorteable=True)
    
    if pagination:
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=page_size)
    
    gb.configure_grid_options(domLayout='normal')
    grid_options = gb.build()
    
    # Display grid
    grid_response = AgGrid(
        df,
        gridOptions=grid_options,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=True,
        height=height,
        width='100%',
        theme='streamlit',
        key=key
    )
    
    selected_data = pd.DataFrame()
    if enable_selection and grid_response['selected_rows'] is not None:
        selected_data = pd.DataFrame(grid_response['selected_rows'])
        
        if not selected_data.empty:
            st.success(f"Selected {len(selected_data)} rows")
    
    return selected_data

def create_summary_table(df: pd.DataFrame, 
                        group_by_cols: List[str],
                        agg_cols: Dict[str, List[str]],
                        title: str = "Summary Table") -> pd.DataFrame:
    """
    Create a summary table with grouped aggregations
    
    Args:
        df: Input DataFrame
        group_by_cols: Columns to group by
        agg_cols: Dictionary with column names and aggregation functions
        title: Table title
        
    Returns:
        Aggregated DataFrame
    """
    st.subheader(title)
    
    try:
        # Perform groupby aggregation
        summary_df = df.groupby(group_by_cols).agg(agg_cols).round(2)
        
        # Flatten column names if multi-level
        if isinstance(summary_df.columns, pd.MultiIndex):
            summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
        
        summary_df = summary_df.reset_index()
        
        # Display with formatting
        st.dataframe(
            summary_df.style.format(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x),
            use_container_width=True
        )
        
        return summary_df
        
    except Exception as e:
        st.error(f"Error creating summary table: {e}")
        return pd.DataFrame()

def create_pivot_table(df: pd.DataFrame,
                      index_cols: List[str],
                      columns_col: str,
                      values_col: str,
                      aggfunc: str = 'sum',
                      title: str = "Pivot Table",
                      show_totals: bool = True) -> pd.DataFrame:
    """
    Create a pivot table with customizable aggregation
    
    Args:
        df: Input DataFrame
        index_cols: Columns for index
        columns_col: Column for pivot columns
        values_col: Column for values
        aggfunc: Aggregation function
        title: Table title
        show_totals: Whether to show row/column totals
        
    Returns:
        Pivot table DataFrame
    """
    st.subheader(title)
    
    try:
        # Create pivot table
        pivot_df = pd.pivot_table(
            df,
            index=index_cols,
            columns=columns_col,
            values=values_col,
            aggfunc=aggfunc,
            fill_value=0,
            margins=show_totals
        )
        
        # Display with formatting
        if pivot_df.select_dtypes(include=[np.number]).shape[1] > 0:
            st.dataframe(
                pivot_df.style.format("{:,.2f}"),
                use_container_width=True
            )
        else:
            st.dataframe(pivot_df, use_container_width=True)
        
        return pivot_df
        
    except Exception as e:
        st.error(f"Error creating pivot table: {e}")
        return pd.DataFrame()

def create_comparison_table(df1: pd.DataFrame, df2: pd.DataFrame,
                          on: str, suffixes: tuple = ('_current', '_previous'),
                          title: str = "Comparison Table") -> pd.DataFrame:
    """
    Create a comparison table between two DataFrames
    
    Args:
        df1: First DataFrame (current)
        df2: Second DataFrame (previous)  
        on: Column to join on
        suffixes: Suffixes for overlapping columns
        title: Table title
        
    Returns:
        Merged comparison DataFrame
    """
    st.subheader(title)
    
    try:
        # Merge DataFrames
        comparison_df = pd.merge(df1, df2, on=on, how='outer', suffixes=suffixes)
        
        # Calculate differences for numeric columns
        numeric_cols = df1.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != on:
                current_col = f"{col}{suffixes[0]}"
                previous_col = f"{col}{suffixes[1]}"
                
                if current_col in comparison_df.columns and previous_col in comparison_df.columns:
                    comparison_df[f"{col}_change"] = (
                        comparison_df[current_col] - comparison_df[previous_col]
                    )
                    comparison_df[f"{col}_change_pct"] = (
                        (comparison_df[current_col] - comparison_df[previous_col]) / 
                        comparison_df[previous_col] * 100
                    ).fillna(0)
        
        # Style the comparison
        def highlight_changes(row):
            styles = [''] * len(row)
            for i, col in enumerate(comparison_df.columns):
                if '_change' in col and not col.endswith('_pct'):
                    if pd.notna(row.iloc[i]) and row.iloc[i] != 0:
                        if row.iloc[i] > 0:
                            styles[i] = 'background-color: #d4edda'  # Green for positive
                        else:
                            styles[i] = 'background-color: #f8d7da'  # Red for negative
            return styles
        
        styled_df = comparison_df.style.apply(highlight_changes, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        return comparison_df
        
    except Exception as e:
        st.error(f"Error creating comparison table: {e}")
        return pd.DataFrame()

def create_ranking_table(df: pd.DataFrame, 
                        rank_by: str,
                        display_cols: List[str],
                        ascending: bool = False,
                        top_n: int = 10,
                        title: str = "Rankings") -> pd.DataFrame:
    """
    Create a ranking table with top N records
    
    Args:
        df: Input DataFrame
        rank_by: Column to rank by
        display_cols: Columns to display
        ascending: Sort order
        top_n: Number of top records to show
        title: Table title
        
    Returns:
        Ranked DataFrame
    """
    st.subheader(f"{title} - Top {top_n}")
    
    try:
        # Create ranking
        ranked_df = df.nlargest(top_n, rank_by) if not ascending else df.nsmallest(top_n, rank_by)
        ranked_df = ranked_df[display_cols].reset_index(drop=True)
        ranked_df.index += 1  # Start ranking from 1
        
        # Add medal emojis for top 3
        if len(ranked_df) >= 1:
            ranked_df.loc[1, 'Rank'] = "🥇 1"
        if len(ranked_df) >= 2:
            ranked_df.loc[2, 'Rank'] = "🥈 2"
        if len(ranked_df) >= 3:
            ranked_df.loc[3, 'Rank'] = "🥉 3"
        for i in range(4, len(ranked_df) + 1):
            if i in ranked_df.index:
                ranked_df.loc[i, 'Rank'] = f"🏅 {i}"
        
        # Reorder columns to put Rank first
        if 'Rank' in ranked_df.columns:
            cols = ['Rank'] + [col for col in ranked_df.columns if col != 'Rank']
            ranked_df = ranked_df[cols]
        
        st.dataframe(ranked_df, use_container_width=True)
        
        return ranked_df
        
    except Exception as e:
        st.error(f"Error creating ranking table: {e}")
        return pd.DataFrame()

def create_conditional_formatting_table(df: pd.DataFrame,
                                       format_rules: Dict[str, Dict[str, Any]],
                                       title: str = "Formatted Table") -> None:
    """
    Create a table with conditional formatting based on rules
    
    Args:
        df: Input DataFrame
        format_rules: Dictionary with column names and formatting rules
        title: Table title
    """
    st.subheader(title)
    
    try:
        styled_df = df.copy()
        
        for col_name, rules in format_rules.items():
            if col_name in styled_df.columns:
                if rules['type'] == 'color_scale':
                    # Apply color scale based on values
                    styled_df = styled_df.style.background_gradient(
                        subset=[col_name],
                        cmap=rules.get('colormap', 'RdYlGn')
                    )
                elif rules['type'] == 'threshold':
                    # Apply threshold-based coloring
                    def apply_threshold_color(val):
                        if val >= rules['high_threshold']:
                            return f"background-color: {rules.get('high_color', '#d4edda')}"
                        elif val >= rules['medium_threshold']:
                            return f"background-color: {rules.get('medium_color', '#fff3cd')}"
                        else:
                            return f"background-color: {rules.get('low_color', '#f8d7da')}"
                    
                    styled_df = styled_df.style.applymap(
                        apply_threshold_color,
                        subset=[col_name]
                    )
        
        st.dataframe(styled_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error applying conditional formatting: {e}")
        st.dataframe(df, use_container_width=True)

def create_statistical_summary_table(df: pd.DataFrame,
                                   columns: Optional[List[str]] = None,
                                   title: str = "Statistical Summary") -> pd.DataFrame:
    """
    Create a statistical summary table for numeric columns
    
    Args:
        df: Input DataFrame
        columns: Specific columns to analyze (default: all numeric)
        title: Table title
        
    Returns:
        Summary statistics DataFrame
    """
    st.subheader(title)
    
    try:
        # Select columns to analyze
        if columns is None:
            numeric_df = df.select_dtypes(include=[np.number])
        else:
            numeric_df = df[columns].select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            st.warning("No numeric columns found for statistical summary")
            return pd.DataFrame()
        
        # Calculate statistics
        stats_df = numeric_df.describe().T
        
        # Add additional statistics
        stats_df['missing'] = numeric_df.isnull().sum()
        stats_df['missing_pct'] = (stats_df['missing'] / len(df)) * 100
        stats_df['unique'] = numeric_df.nunique()
        stats_df['skewness'] = numeric_df.skew()
        stats_df['kurtosis'] = numeric_df.kurtosis()
        
        # Round values
        stats_df = stats_df.round(3)
        
        # Display with formatting
        st.dataframe(
            stats_df.style.format({
                'missing_pct': '{:.2f}%',
                'skewness': '{:.3f}',
                'kurtosis': '{:.3f}'
            }),
            use_container_width=True
        )
        
        return stats_df
        
    except Exception as e:
        st.error(f"Error creating statistical summary: {e}")
        return pd.DataFrame()

def create_correlation_table(df: pd.DataFrame,
                           columns: Optional[List[str]] = None,
                           method: str = 'pearson',
                           title: str = "Correlation Matrix") -> pd.DataFrame:
    """
    Create a correlation matrix table
    
    Args:
        df: Input DataFrame
        columns: Specific columns to analyze
        method: Correlation method ('pearson', 'spearman', 'kendall')
        title: Table title
        
    Returns:
        Correlation matrix DataFrame
    """
    st.subheader(title)
    
    try:
        # Select columns
        if columns is None:
            numeric_df = df.select_dtypes(include=[np.number])
        else:
            numeric_df = df[columns].select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            st.warning("No numeric columns found for correlation analysis")
            return pd.DataFrame()
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr(method=method)
        
        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        corr_matrix_masked = corr_matrix.mask(mask)
        
        # Apply conditional formatting
        styled_corr = corr_matrix_masked.style.background_gradient(
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1
        ).format('{:.3f}')
        
        st.dataframe(styled_corr, use_container_width=True)
        
        return corr_matrix
        
    except Exception as e:
        st.error(f"Error creating correlation table: {e}")
        return pd.DataFrame()

def create_data_profile_table(df: pd.DataFrame,
                            title: str = "Data Profile") -> pd.DataFrame:
    """
    Create a comprehensive data profile table
    
    Args:
        df: Input DataFrame
        title: Table title
        
    Returns:
        Data profile DataFrame
    """
    st.subheader(title)
    
    try:
        profile_data = []
        
        for col in df.columns:
            col_data = {
                'Column': col,
                'Data Type': str(df[col].dtype),
                'Non-Null Count': df[col].count(),
                'Null Count': df[col].isnull().sum(),
                'Null %': f"{(df[col].isnull().sum() / len(df)) * 100:.1f}%",
                'Unique Values': df[col].nunique(),
                'Unique %': f"{(df[col].nunique() / len(df)) * 100:.1f}%"
            }
            
            # Add type-specific statistics
            if df[col].dtype in ['int64', 'float64']:
                col_data.update({
                    'Min': df[col].min(),
                    'Max': df[col].max(),
                    'Mean': df[col].mean(),
                    'Std': df[col].std()
                })
            elif df[col].dtype == 'object':
                col_data.update({
                    'Min Length': df[col].astype(str).str.len().min(),
                    'Max Length': df[col].astype(str).str.len().max(),
                    'Most Frequent': df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A',
                    'Least Frequent': df[col].value_counts().index[-1] if len(df[col].value_counts()) > 0 else 'N/A'
                })
            
            profile_data.append(col_data)
        
        profile_df = pd.DataFrame(profile_data)
        
        # Apply conditional formatting
        def highlight_nulls(row):
            styles = [''] * len(row)
            if 'Null %' in row.index:
                null_pct = float(row['Null %'].rstrip('%'))
                if null_pct > 50:
                    styles[row.index.get_loc('Null %')] = 'background-color: #f8d7da'
                elif null_pct > 20:
                    styles[row.index.get_loc('Null %')] = 'background-color: #fff3cd'
            return styles
        
        styled_profile = profile_df.style.apply(highlight_nulls, axis=1)
        st.dataframe(styled_profile, use_container_width=True)
        
        return profile_df
        
    except Exception as e:
        st.error(f"Error creating data profile: {e}")
        return pd.DataFrame()

def export_table_options(df: pd.DataFrame, filename_prefix: str = "data") -> None:
    """
    Provide export options for tables
    
    Args:
        df: DataFrame to export
        filename_prefix: Prefix for the filename
    """
    st.markdown("### 📥 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv,
            file_name=f"{filename_prefix}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        excel_buffer = BytesIO()
        df.to_excel(excel_buffer, index=False)
        st.download_button(
            label="📊 Download Excel",
            data=excel_buffer.getvalue(),
            file_name=f"{filename_prefix}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col3:
        json_str = df.to_json(orient='records', date_format='iso')
        st.download_button(
            label="🔗 Download JSON",
            data=json_str,
            file_name=f"{filename_prefix}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Import required for Excel export
from io import BytesIO