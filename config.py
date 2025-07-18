"""
Configuration file for Customer Churn Prediction ML Pipeline
"""

import os
from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    """Configuration class for the ML pipeline"""
    
    # Data paths
    DATA_PATH: str = 'raw_data/customer_churn.json'
    MODEL_DIR: str = 'models'
    TEST_DATA_DIR: str = 'data/test_data'
    
    # Model parameters
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    
    # XGBoost parameters for imbalanced data
    N_ESTIMATORS: int = 200
    MAX_DEPTH: int = 6
    LEARNING_RATE: float = 0.1
    SUBSAMPLE: float = 0.8
    COLSAMPLE_BYTREE: float = 0.8
    MIN_CHILD_WEIGHT: int = 1
    GAMMA: float = 0.1
    REG_ALPHA: float = 0.1
    REG_LAMBDA: float = 1.0
    
    # Imbalanced data handling parameters
    SCALE_POS_WEIGHT: float = 2.89  # Primary imbalanced data parameter (333/115 ≈ 2.89)
    MAX_DELTA_STEP: float = 1.0  # Helps with imbalanced data convergence
    
    # Additional parameters that help with imbalanced data
    EARLY_STOPPING_ROUNDS: int = 10  # Prevent overfitting on majority class
    EVAL_METRIC: str = 'auc'  # Better metric for imbalanced data than logloss
    
    # Cross-validation parameters
    CV_FOLDS: int = 5
    CV_RANDOM_STATE: int = 42
    
    # MLflow configuration
    MLFLOW_TRACKING_URI: str = os.getenv('MLFLOW_TRACKING_URI', "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME: str = "customer_churn_prediction"
    MLFLOW_MODEL_NAME: str = "churn_predictor"
    
    
    # Feature engineering parameters
    SKIP_THRESHOLD: float = 30.0  # seconds
    CHURN_DAYS_THRESHOLD: int = 30
    PLAY_RATIO_THRESHOLD: float = 0.5
    CHURN_THRESHOLD: float = 0.5  # Threshold for binary churn prediction
    
    # Categorical columns
    CATEGORICAL_COLS: List[str] = None
    
    def __post_init__(self):
        if self.CATEGORICAL_COLS is None:
            self.CATEGORICAL_COLS = ['location', 'device_type']


# Global config instance
config = Config()