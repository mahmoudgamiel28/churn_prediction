"""
Data preprocessing module for Customer Churn Prediction
"""

import pandas as pd
from typing import Tuple, Optional
import logging

from config import config
from src.utils.helpers import convert_unix_to_timestamp, print_data_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    try:
        df = pd.read_json(file_path, lines=True)
        logger.info(f"Successfully loaded data from {file_path}")
        print_data_info(df, "Raw Data")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw data by removing invalid entries and handling missing values.
    
    Args:
        df (pd.DataFrame): Raw DataFrame
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    logger.info("Starting data cleaning...")
    
    # Remove rows with empty userId
    initial_shape = df.shape
    df = df[df['userId'] != ''].copy()
    logger.info(f"Removed {initial_shape[0] - df.shape[0]} rows with empty userId")
    
    # Convert timestamps
    df['registration'] = convert_unix_to_timestamp(df['registration'])
    df['ts'] = convert_unix_to_timestamp(df['ts'])
    logger.info("Converted timestamps to datetime format")
    
    # Drop unnecessary columns for initial analysis
    columns_to_drop = ['artist', 'song', 'location']
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)
        logger.info(f"Dropped columns: {existing_cols_to_drop}")
    
    # Handle missing values in length column
    if 'length' in df.columns:
        df['length'] = df['length'].fillna(0)
        logger.info("Filled missing values in 'length' column with 0")
    
    print_data_info(df, "Cleaned Data")
    return df


def create_churn_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create churn labels based on last activity and cancellation confirmations.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with churn labels per user
    """
    logger.info("Creating churn labels...")
    
    # Get last activity per user
    last_activity = df.groupby('userId')['ts'].max().reset_index()
    last_activity.columns = ['userId', 'last_seen']
    
    # Calculate days since last seen
    current_date = df['ts'].max()
    last_activity['days_since_last_seen'] = (current_date - last_activity['last_seen']).dt.days
    
    # Create inactive churn (users inactive for more than threshold days)
    last_activity['inactive_churn'] = (
        last_activity['days_since_last_seen'] >= config.CHURN_DAYS_THRESHOLD
    ).astype(int)
    
    # Create explicit churn (users who confirmed cancellation)
    canceled_users = df[df['page'] == 'Cancellation Confirmation']['userId'].unique()
    last_activity['explicit_churn'] = last_activity['userId'].isin(canceled_users).astype(int)
    
    # Combine both types of churn
    last_activity['churn'] = (
        (last_activity['explicit_churn'] == 1) | (last_activity['inactive_churn'] == 1)
    ).astype(int)
    
    logger.info(f"Created churn labels for {len(last_activity)} users")
    logger.info(f"Churn distribution: {last_activity['churn'].value_counts().to_dict()}")
    
    return last_activity


def preprocess_data(file_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main preprocessing pipeline.
    
    Args:
        file_path (str, optional): Path to data file. Uses config default if None.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (cleaned_data, churn_labels)
    """
    if file_path is None:
        file_path = config.DATA_PATH
    
    logger.info("Starting data preprocessing pipeline...")
    
    # Load and clean data
    raw_data = load_data(file_path)
    cleaned_data = clean_data(raw_data)
    
    # Create churn labels
    churn_labels = create_churn_labels(cleaned_data)
    
    # Save preprocessed data
    import os
    os.makedirs("data/preprocessed_data", exist_ok=True)
    
    # Save cleaned data
    cleaned_data.to_parquet("data/preprocessed_data/cleaned_data.parquet", index=False)
    logger.info("Saved cleaned data to data/preprocessed_data/cleaned_data.parquet")
    
    # Save churn labels
    churn_labels.to_parquet("data/preprocessed_data/churn_labels.parquet", index=False)
    logger.info("Saved churn labels to data/preprocessed_data/churn_labels.parquet")
    
    logger.info("Data preprocessing completed successfully!")
    return cleaned_data, churn_labels