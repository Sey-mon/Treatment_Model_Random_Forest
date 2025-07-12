"""
Streamlit data type fixes for PyArrow compatibility
"""

import pandas as pd
import numpy as np

def fix_dataframe_for_streamlit(df):
    """
    Fix DataFrame data types to prevent PyArrow serialization errors
    """
    if df is None or df.empty:
        return df
        
    df_fixed = df.copy()
    
    # Convert all object columns to strings to avoid mixed type issues
    for col in df_fixed.columns:
        if df_fixed[col].dtype == 'object':
            df_fixed[col] = df_fixed[col].astype(str)
    
    # Handle NaN values
    df_fixed = df_fixed.fillna('N/A')
    
    # Ensure numeric columns are properly typed
    numeric_columns = ['age_months', 'weight', 'height', 'whz_score', 'bmi', 'total_household', 'adults', 'children', 'twins']
    for col in numeric_columns:
        if col in df_fixed.columns:
            try:
                # Convert to numeric, then back to string for display
                numeric_vals = pd.to_numeric(df_fixed[col], errors='coerce')
                df_fixed[col] = numeric_vals.apply(
                    lambda x: f"{x:.2f}" if pd.notnull(x) and isinstance(x, (int, float)) else 'N/A'
                )
            except:
                df_fixed[col] = df_fixed[col].astype(str)
    
    return df_fixed

def create_display_dataframe(data_dict):
    """
    Create a display-friendly DataFrame from a dictionary
    """
    if isinstance(data_dict, dict):
        # Convert to list of tuples for consistent display
        data_list = []
        for key, value in data_dict.items():
            if isinstance(value, (int, float)) and not pd.isna(value):
                if isinstance(value, float):
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
            else:
                formatted_value = str(value) if value is not None else 'N/A'
            
            data_list.append([str(key), formatted_value])
        
        df = pd.DataFrame(data_list, columns=['Field', 'Value'])
    else:
        df = pd.DataFrame(data_dict)
    
    return fix_dataframe_for_streamlit(df)

def safe_numeric_format(value, decimals=2):
    """
    Safely format numeric values for display
    """
    if pd.isna(value) or value is None:
        return 'N/A'
    
    try:
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                return f"{value:.{decimals}f}"
            else:
                return str(value)
        else:
            # Try to convert to float
            float_val = float(value)
            return f"{float_val:.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)

def prepare_batch_results(results_list):
    """
    Prepare batch results for safe display
    """
    if not results_list:
        return pd.DataFrame()
    
    # Convert all values to strings for safe display
    safe_results = []
    for result in results_list:
        safe_result = {}
        for key, value in result.items():
            if key in ['age_months', 'weight', 'height', 'whz_score', 'bmi']:
                safe_result[key] = safe_numeric_format(value)
            else:
                safe_result[key] = str(value) if value is not None else 'N/A'
        safe_results.append(safe_result)
    
    df = pd.DataFrame(safe_results)
    return fix_dataframe_for_streamlit(df)
