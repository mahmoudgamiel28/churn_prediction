"""
Main pipeline script for Customer Churn Prediction
"""

import os
import sys
import logging
from pathlib import Path

from src.data.preprocessing import preprocess_data
from src.features.engineering import engineer_features
from src.models.training import ChurnPredictor

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main pipeline execution function.
    """
    logger.info("Starting Customer Churn Prediction Pipeline...")
    
    try:
        # Step 1: Data Preprocessing
        logger.info("=" * 50)
        logger.info("STEP 1: Data Preprocessing")
        logger.info("=" * 50)
        
        cleaned_data, churn_labels = preprocess_data()
        
        # Step 2: Feature Engineering
        logger.info("=" * 50)
        logger.info("STEP 2: Feature Engineering")
        logger.info("=" * 50)
        
        user_features = engineer_features(cleaned_data)
        
        # Step 3: Model Training and Evaluation
        logger.info("=" * 50)
        logger.info("STEP 3: Model Training and Evaluation")
        logger.info("=" * 50)
        
        # Initialize model
        predictor = ChurnPredictor()
        
        # Prepare features
        X, y = predictor.prepare_features(user_features, churn_labels)
        
        # Split data
        X_train, X_test, y_train, y_test = predictor.train_test_split(X, y)
        
        # Scale features
        X_train_scaled, X_test_scaled = predictor.scale_features(X_train, X_test)
        
        # Train model
        predictor.train_model(X_train_scaled, y_train)
        
        # Evaluate model
        metrics = predictor.evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Save model
        model_path = "models/churn_predictor.joblib"
        os.makedirs("models", exist_ok=True)
        predictor.save_model(model_path)
        
        logger.info("=" * 50)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 50)
        
        return predictor, metrics
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise


def predict_churn(model_path: str, features_df):
    """
    Make predictions using a trained model.
    
    Args:
        model_path (str): Path to the saved model
        features_df: DataFrame with user features
        
    Returns:
        predictions and probabilities
    """
    predictor = ChurnPredictor()
    predictor.load_model(model_path)
    
    predictions = predictor.predict(features_df)
    probabilities = predictor.predict_proba(features_df)
    
    return predictions, probabilities


if __name__ == "__main__":
    # Run the main pipeline
    trained_model, evaluation_metrics = main()
    
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print("✅ Data preprocessing completed")
    print("✅ Feature engineering completed")
    print("✅ Model training completed")
    print("✅ Model evaluation completed")
    print("✅ Model saved to models/churn_predictor.joblib")
    print("\nKey Metrics:")
    print(f"  - Test Accuracy: {evaluation_metrics['test_accuracy']:.4f}")
    print(f"  - Test AUC: {evaluation_metrics['test_auc']:.4f}")
    print("=" * 60)