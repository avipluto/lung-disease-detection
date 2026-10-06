"""
Pydantic schemas for the Lung Disease Detection API.
"""
from pydantic import BaseModel, Field
from typing import List


class PredictionResult(BaseModel):
    """Individual disease prediction result."""
    disease: str = Field(..., description="Name of the detected disease")
    confidence: float = Field(..., ge=0, le=100, description="Confidence percentage (0-100)")
    description: str = Field(..., description="Brief medical description of the disease")


class PredictionResponse(BaseModel):
    """Response from the prediction endpoint."""
    status: str = Field(default="success", description="Status of the prediction")
    filename: str = Field(..., description="Name of the uploaded file")
    predictions: List[PredictionResult] = Field(..., description="List of disease predictions")


class HealthResponse(BaseModel):
    """Response from the health check endpoint."""
    status: str = Field(default="healthy", description="Service status")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    device: str = Field(..., description="Compute device (cpu/cuda)")
    model_name: str = Field(default="DenseNet121-CheXNet", description="Model architecture name")
