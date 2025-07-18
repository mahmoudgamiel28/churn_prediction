"""
Utility functions for the Customer Churn Prediction ML Pipeline
"""

import pandas as pd
import numpy as np
from typing import Union


def convert_unix_to_timestamp(series: pd.Series) -> pd.Series:
    """
    Converts a pandas Series of Unix timestamps (in milliseconds) to datetime format.
    
    Args:
        series (pd.Series): A Series of Unix timestamps (int, in milliseconds).
        
    Returns:
        pd.Series: A Series of pandas datetime objects.
    """
    return pd.to_datetime(series, unit='ms')


def extract_device_type(user_agent: str) -> str:
    """
    Extract device type from user agent string.
    
    Args:
        user_agent (str): User agent string
        
    Returns:
        str: Device type ('Mobile', 'Tablet', 'Desktop', 'Unknown')
    """
    if pd.isna(user_agent):
        return 'Unknown'
    
    ua_lower = user_agent.lower()
    if 'mobile' in ua_lower or 'android' in ua_lower or 'iphone' in ua_lower:
        return 'Mobile'
    elif 'tablet' in ua_lower or 'ipad' in ua_lower:
        return 'Tablet'
    else:
        return 'Desktop'


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that a DataFrame contains all required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        
    Returns:
        bool: True if all required columns are present
        
    Raises:
        ValueError: If required columns are missing
    """
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return True


def print_data_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Print basic information about a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        name (str): Name to display for the DataFrame
    """
    print(f"\n{name} Info:")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Null values: {df.isnull().sum().sum()}")
    if 'churn' in df.columns:
        print(f"Churn distribution:\n{df['churn'].value_counts()}")