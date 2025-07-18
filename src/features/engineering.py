"""
Feature engineering module for Customer Churn Prediction
"""

import pandas as pd
import numpy as np
import logging

from config import config
from src.utils.helpers import extract_device_type

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Required page types for event counting
required_pages = [
    'NextSong', 'Logout', 'Thumbs Up', 'Thumbs Down', 'Add to Playlist',
    'Upgrade', 'Submit Upgrade', 'Roll Advert', 'Help', 'Add Friend',
    'About', 'Settings', 'Submit Downgrade', 'Error', 'Save Settings'
]


def create_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create session-based features for each user.
    
    Args:
        df (pd.DataFrame): Cleaned user activity data
        
    Returns:
        pd.DataFrame: Session features per user
    """
    logger.info("Creating session features...")
    
    # Basic session aggregations
    session_features = df.groupby('userId').agg({
        'sessionId': 'nunique',  # num_sessions
        'ts': ['min', 'max'],    # for session duration calculation
        'registration': 'first'  # registration time
    }).reset_index()
    
    session_features.columns = ['userId', 'num_sessions', 'first_activity', 'last_activity', 'registration']
    
    # Calculate songs per session
    songs_per_session = df[df['page'] == 'NextSong'].groupby(['userId', 'sessionId']).size().reset_index(name='songs_in_session')
    avg_songs_per_session = songs_per_session.groupby('userId')['songs_in_session'].mean().reset_index()
    avg_songs_per_session.columns = ['userId', 'avg_songs_listen_per_session']
    session_features = session_features.merge(avg_songs_per_session, on='userId', how='left')
    session_features['avg_songs_listen_per_session'] = session_features['avg_songs_listen_per_session'].fillna(0)
    
    # Calculate average session duration (in hours)
    session_durations = df.groupby(['userId', 'sessionId'])['ts'].agg(['min', 'max']).reset_index()
    session_durations['session_duration'] = (session_durations['max'] - session_durations['min']).dt.total_seconds() / 3600
    avg_session_duration = session_durations.groupby('userId')['session_duration'].mean().reset_index()
    session_features = session_features.merge(avg_session_duration, on='userId', how='left')
    session_features['avg_session_duration'] = session_features['session_duration'].fillna(0)
    
    # Calculate days_active_ratio using activity span
    # Use the range of activity dates instead of registration-based lifespan
    session_features['activity_span'] = (session_features['last_activity'] - session_features['first_activity']).dt.days + 1
    session_features['activity_span'] = session_features['activity_span'].clip(lower=1)
    
    # Calculate days_active as count of unique days user was active
    df_with_date = df.copy()
    df_with_date['date'] = df_with_date['ts'].dt.date
    days_active = df_with_date.groupby('userId')['date'].nunique().reset_index()
    days_active.columns = ['userId', 'days_active']
    session_features = session_features.merge(days_active, on='userId', how='left')
    session_features['days_active'] = session_features['days_active'].fillna(0)
    
    session_features['days_active_ratio'] = session_features['days_active'] / session_features['activity_span']
    session_features['days_active_ratio'] = session_features['days_active_ratio'].fillna(0)
    # Cap the ratio at 1.0 to avoid extreme values
    session_features['days_active_ratio'] = session_features['days_active_ratio'].clip(upper=1.0)
    
    # Remove days_active column as it's not needed in final features
    session_features = session_features.drop('days_active', axis=1)
    
    logger.info(f"Created session features for {len(session_features)} users")
    return session_features[['userId', 'num_sessions', 
                           'avg_songs_listen_per_session', 'avg_session_duration', 
                           'days_active_ratio']]


def create_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create demographic features for each user.
    
    Args:
        df (pd.DataFrame): Cleaned user activity data
        
    Returns:
        pd.DataFrame: Demographic features per user
    """
    logger.info("Creating demographic features...")
    
    demo_features = df.groupby('userId').agg({
        'gender': 'first',
        'level': lambda x: (x == 'paid').mean(),  # level_paid_ratio
        'userAgent': 'first'
    }).reset_index()
    
    demo_features.columns = ['userId', 'gender', 'level_paid_ratio', 'userAgent']
    
    # Transform gender to binary (1 for F, 0 for M)
    demo_features['gender'] = (demo_features['gender'] == 'F').astype(int)
    
    # Extract device type from userAgent
    demo_features['device_type'] = demo_features['userAgent'].apply(extract_device_type)
    demo_features = demo_features.drop('userAgent', axis=1)
    
    logger.info(f"Created demographic features for {len(demo_features)} users")
    return demo_features


def create_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create event count features for each user.
    
    Args:
        df (pd.DataFrame): Cleaned user activity data
        
    Returns:
        pd.DataFrame: Event count features per user
    """
    logger.info("Creating event count features...")
    
    # Create page count matrix
    page_counts = df.groupby(['userId', 'page']).size().unstack(fill_value=0).reset_index()
    
    # Ensure all required columns exist
    for page in required_pages:
        if page not in page_counts.columns:
            page_counts[page] = 0
    
    # Select only the required columns
    event_features = page_counts[['userId'] + required_pages].copy()
    
    # Rename columns to match the feature names
    event_features.columns = ['userId'] + [f'count_{page.replace(" ", "_").lower()}' for page in required_pages]
    
    logger.info(f"Created event count features for {len(event_features)} users")
    return event_features


def create_skip_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create skip-related features for each user.
    
    Args:
        df (pd.DataFrame): Cleaned user activity data
        
    Returns:
        pd.DataFrame: Skip features per user
    """
    logger.info("Creating skip features...")
    
    # Filter for NextSong events only
    songs_df = df[df['page'] == 'NextSong'].copy()
    songs_df = songs_df.sort_values(by=['userId', 'sessionId', 'ts'])
    
    # Calculate play duration by looking at time to next event
    songs_df['next_ts'] = songs_df.groupby(['userId', 'sessionId'])['ts'].shift(-1)
    songs_df['play_duration'] = (songs_df['next_ts'] - songs_df['ts']).dt.total_seconds()
    songs_df['song_length'] = songs_df['length']
    
    # Handle edge cases for played_ratio calculation
    songs_df['played_ratio'] = np.where(
        (songs_df['song_length'] > 0) & (songs_df['play_duration'].notna()),
        songs_df['play_duration'] / songs_df['song_length'],
        0
    )
    # Cap played_ratio to avoid extreme values
    songs_df['played_ratio'] = songs_df['played_ratio'].clip(upper=10.0)
    
    # Consider a song skipped if played less than threshold % of its length
    songs_df['is_skip'] = (
        (songs_df['played_ratio'] < config.PLAY_RATIO_THRESHOLD) & 
        (songs_df['play_duration'].notna()) & 
        (songs_df['song_length'] > 0)
    )
    
    # Aggregate skip features per user
    skip_agg = songs_df.groupby('userId').agg({
        'is_skip': ['sum', 'mean'],  # total skips and skip rate
        'played_ratio': ['mean', 'std'],  # avg completion rate and consistency
        'song_length': 'count'  # total songs listened
    }).reset_index()
    
    skip_agg.columns = ['userId', 'total_skips', 'skip_rate', 'avg_completion_rate', 'completion_consistency', 'total_songs']
    
    # Fill NaN values for consistency (when user has only 1 song)
    skip_agg['completion_consistency'] = skip_agg['completion_consistency'].fillna(0)
    
    logger.info(f"Created skip features for {len(skip_agg)} users")
    return skip_agg


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main feature engineering pipeline.
    
    Args:
        df (pd.DataFrame): Cleaned user activity data
        
    Returns:
        pd.DataFrame: Complete feature set per user
    """
    logger.info("Starting feature engineering pipeline...")
    
    # Create different feature groups
    session_features = create_session_features(df)
    demo_features = create_demographic_features(df)
    event_features = create_event_features(df)
    skip_features = create_skip_features(df)
    
    # Merge all features
    final_features = session_features
    final_features = final_features.merge(demo_features, on='userId', how='left')
    final_features = final_features.merge(event_features, on='userId', how='left')
    final_features = final_features.merge(skip_features, on='userId', how='left')
    
    # Fill any missing values with 0
    final_features = final_features.fillna(0)
    
    # Handle any remaining infinite values
    final_features = final_features.replace([np.inf, -np.inf], 0)
    
    # Remove userId from features for model training
    features_for_training = final_features.drop('userId', axis=1)
    
    # Log any extreme values for debugging
    numeric_cols = final_features.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'userId':
            max_val = final_features[col].max()
            min_val = final_features[col].min()
            if max_val > 1e6 or min_val < -1e6:
                logger.warning(f"Extreme values detected in {col}: min={min_val}, max={max_val}")
                # Cap extreme values
                final_features[col] = final_features[col].clip(lower=-1e6, upper=1e6)
    
    # Save engineered features
    import os
    os.makedirs("data/preprocessed_data", exist_ok=True)
    
    # Save features with userId for reference
    final_features.to_parquet("data/preprocessed_data/engineered_features_with_id.parquet", index=False)
    logger.info("Saved engineered features with userId to data/preprocessed_data/engineered_features_with_id.parquet")
    
    # Save features without userId for model training
    features_for_training.to_parquet("data/preprocessed_data/engineered_features.parquet", index=False)
    logger.info("Saved engineered features for training to data/preprocessed_data/engineered_features.parquet")
    
    # Also save individual feature groups for analysis
    session_features.to_parquet("data/preprocessed_data/session_features.parquet", index=False)
    demo_features.to_parquet("data/preprocessed_data/demographic_features.parquet", index=False)
    event_features.to_parquet("data/preprocessed_data/event_features.parquet", index=False)
    skip_features.to_parquet("data/preprocessed_data/skip_features.parquet", index=False)
    logger.info("Saved individual feature groups to data/preprocessed_data/")
    
    logger.info(f"Feature engineering completed! Final shape: {features_for_training.shape}")
    logger.info(f"Features created: {list(features_for_training.columns)}")
    
    return features_for_training
