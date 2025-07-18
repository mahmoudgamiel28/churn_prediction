"""
FastAPI application for Customer Churn Prediction
"""
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import logging
from typing import Optional
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from datetime import datetime

from .models import (
    PredictionRequest, PredictionResponse, HealthResponse
)
from pydantic import BaseModel
from typing import List, Dict, Any

# Response model for batch predictions
class BatchPredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    total_records: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    filename: str
from ..features.engineering import engineer_features
from ..data.preprocessing import clean_data
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn based on user activity data",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
model_version = None
model_stage = None

def load_model():
    """Load the trained model from MLflow registry or fallback to local file"""
    global model, model_version, model_stage
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    try:
        # Try to load latest model from MLflow registry
        try:
            latest_version = client.get_latest_versions(
                config.MLFLOW_MODEL_NAME, 
                stages=["Production", "Staging", "None"]
            )
            
            if latest_version:
                # Get the latest version (prefer Production, then Staging, then None)
                model_version_info = None
                for stage in ["Production", "Staging", "None"]:
                    for version in latest_version:
                        if version.current_stage == stage:
                            model_version_info = version
                            break
                    if model_version_info:
                        break
                
                if model_version_info:
                    model_uri = f"models:/{config.MLFLOW_MODEL_NAME}/{model_version_info.version}"
                    model = mlflow.sklearn.load_model(model_uri)
                    model_version = model_version_info.version
                    model_stage = model_version_info.current_stage
                    logger.info(f"Model loaded from MLflow registry: {model_uri} (stage: {model_stage})")
                    return
        except Exception as mlflow_error:
            logger.warning(f"Failed to load model from MLflow registry: {mlflow_error}")
        
        # Fallback to local model file
        model_path = os.path.join(config.MODEL_DIR, "churn_predictor.joblib")
        if os.path.exists(model_path):
            from src.models.training import ChurnPredictor
            model = ChurnPredictor()
            model.load_model(model_path)
            model_version = "local"
            model_stage = "local"
            logger.info(f"Model loaded from local file: {model_path}")
        else:
            logger.warning(f"No model found at {model_path}")
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not_loaded"
    return HealthResponse(
        status="healthy",
        model_status=model_status
    )


@app.post("/predict_from_test_data", response_model=BatchPredictionResponse)
async def predict_from_test_data(file_path: str):
    """
    Predict churn probabilities for a batch of users from test data file
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Determine the file path in the test_data directory
        file_path = os.path.join(config.TEST_DATA_DIR, file_path)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Test data file not found")

        # Load data from file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path, lines=True)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Clean and engineer features
        cleaned_df = clean_data(df)
        features_df = engineer_features(cleaned_df)

        # Remove userId column for prediction
        feature_columns = [col for col in features_df.columns if col != 'userId']
        X = features_df[feature_columns]

        # Make predictions (handle both MLflow model and ChurnPredictor)
        predictions = []
        high_risk_count = 0
        medium_risk_count = 0
        low_risk_count = 0
        
        if hasattr(model, 'feature_names') and hasattr(model, 'scaler'):
            # ChurnPredictor instance from local file
            churn_probabilities = model.predict_proba(features_df)[:, 1]
        else:
            # Direct sklearn model from MLflow
            churn_probabilities = model.predict_proba(X)[:, 1]

        for churn_probability in churn_probabilities:
            churn_prediction = int(churn_probability > config.CHURN_THRESHOLD)
            risk_level = "high" if churn_probability > 0.7 else "medium" if churn_probability > 0.3 else "low"
            
            predictions.append({
                "churn_probability": float(churn_probability),
                "churn_prediction": churn_prediction,
                "risk_level": risk_level
            })

            if risk_level == "high":
                high_risk_count += 1
            elif risk_level == "medium":
                medium_risk_count += 1
            else:
                low_risk_count += 1

        return BatchPredictionResponse(
            predictions=predictions,
            total_records=len(predictions),
            high_risk_count=high_risk_count,
            medium_risk_count=medium_risk_count,
            low_risk_count=low_risk_count,
            filename=os.path.basename(file_path)
        )

    except Exception as e:
        logger.error(f"Error during batch prediction from test data: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "features_count": getattr(model, 'n_features_in_', 'unknown'),
        "model_loaded": True,
        "model_version": model_version,
        "model_stage": model_stage,
        "source": "mlflow_registry" if model_version != "local" else "local_file"
    }





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)