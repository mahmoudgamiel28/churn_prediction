"""
Pydantic models for API request/response schemas
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class PredictionRequest(BaseModel):
    """Request model for churn prediction"""
    userId: str = Field(..., description="User ID")
    gender: str = Field(..., description="User gender (M/F)")
    level: str = Field(..., description="User subscription level (free/paid)")
    userAgent: str = Field(..., description="User agent string")
    sessionId: int = Field(..., description="Session ID")
    page: str = Field(..., description="Page/action type")
    ts: int = Field(..., description="Timestamp (Unix)")
    registration: int = Field(..., description="Registration timestamp (Unix)")
    length: Optional[float] = Field(None, description="Song length in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "userId": "123456",
                "gender": "M",
                "level": "paid",
                "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "sessionId": 789,
                "page": "NextSong",
                "ts": 1543537327796,
                "registration": 1540919166796,
                "length": 207.77751
            }
        }


class PredictionResponse(BaseModel):
    """Response model for churn prediction"""
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    churn_prediction: int = Field(..., description="Binary churn prediction (0/1)")
    risk_level: str = Field(..., description="Risk level (low/medium/high)")
    
    class Config:
        schema_extra = {
            "example": {
                "churn_probability": 0.75,
                "churn_prediction": 1,
                "risk_level": "high"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="API status")
    model_status: str = Field(..., description="Model loading status")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_status": "loaded"
            }
        }


