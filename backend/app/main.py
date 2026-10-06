"""
FastAPI application for Lung Disease Detection.

Provides REST API endpoints for chest X-ray analysis using
DenseNet121 (CheXNet) deep learning model.
"""
import os
import uuid
import logging
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from app.model import get_model, predict_image
from app.schemas import PredictionResponse, PredictionResult, HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Allowed image extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Initialize FastAPI app
app = FastAPI(
    title="LungScan AI - Lung Disease Detection API",
    description="Detect lung diseases from chest X-ray images using DenseNet121 (CheXNet) deep learning model.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount uploads directory for serving images
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.on_event("startup")
async def startup_event():
    """Initialize the model on application startup."""
    logger.info("Starting LungScan AI backend...")
    try:
        model = get_model()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("The API will start but predictions will fail until the model is loaded.")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to LungScan AI - Lung Disease Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint returning model and service status."""
    try:
        model = get_model()
        return HealthResponse(
            status="healthy",
            model_loaded=model.is_loaded,
            device=str(model.device),
            model_name="DenseNet121-CheXNet"
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            device="unknown",
            model_name="DenseNet121-CheXNet"
        )


@app.post("/api/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(..., description="Chest X-ray image file")):
    """
    Analyze a chest X-ray image for lung diseases.

    Upload a chest X-ray image and receive predictions for 14 thoracic diseases
    with confidence scores and medical descriptions.

    - **file**: Chest X-ray image (JPEG, PNG, BMP, TIFF, WebP)
    - Returns top 5 disease predictions sorted by confidence
    """
    # Validate file type
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file_ext}'. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Read and validate file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB."
        )

    # Save uploaded file
    unique_filename = f"{uuid.uuid4().hex}{file_ext}"
    file_path = UPLOAD_DIR / unique_filename

    try:
        with open(file_path, "wb") as f:
            f.write(contents)
        logger.info(f"Saved uploaded file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

    # Run prediction
    try:
        predictions = predict_image(str(file_path), top_k=5)

        prediction_results = [
            PredictionResult(
                disease=pred["disease"],
                confidence=pred["confidence"],
                description=pred["description"]
            )
            for pred in predictions
        ]

        logger.info(f"Prediction completed for {file.filename}: top result = {predictions[0]['disease']} ({predictions[0]['confidence']:.1f}%)")

        return PredictionResponse(
            status="success",
            filename=file.filename,
            predictions=prediction_results
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        # Clean up uploaded file on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again."}
    )
