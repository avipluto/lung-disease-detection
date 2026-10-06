"""
DenseNet121 (CheXNet) Model for Lung Disease Detection.

This module provides a pretrained DenseNet121 model modified for 14-class
chest X-ray disease classification, following the CheXNet architecture.
"""
import os
import logging
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)

# 14 disease classes from NIH ChestX-ray14 dataset
DISEASE_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia"
]

# Medical descriptions for each disease
DISEASE_DESCRIPTIONS: Dict[str, str] = {
    "Atelectasis": "Partial or complete collapse of the lung or a section of the lung, resulting in reduced gas exchange.",
    "Cardiomegaly": "Enlargement of the heart, often indicating underlying heart disease or heart failure.",
    "Effusion": "Abnormal accumulation of fluid in the pleural space between the lungs and chest wall.",
    "Infiltration": "Substance denser than air (fluid, pus, blood, cells) that lingers within lung tissue.",
    "Mass": "A lesion seen on chest X-ray as an opacity greater than 3 cm, may indicate lung cancer.",
    "Nodule": "A small rounded opacity in the lung, less than 3 cm in diameter, requires monitoring.",
    "Pneumonia": "Infection that inflames the air sacs in one or both lungs, causing cough and difficulty breathing.",
    "Pneumothorax": "Collapsed lung caused by air leaking into the space between the lung and chest wall.",
    "Consolidation": "Region of lung tissue filled with liquid instead of air, commonly seen in pneumonia.",
    "Edema": "Excess fluid in the lungs (pulmonary edema), often related to heart failure.",
    "Emphysema": "Chronic condition where air sacs in the lungs are damaged, causing shortness of breath.",
    "Fibrosis": "Scarring and thickening of lung tissue, leading to progressive breathing difficulty.",
    "Pleural Thickening": "Thickening of the pleural lining of the lungs, often due to asbestos exposure or infection.",
    "Hernia": "Protrusion of an organ through the diaphragm into the chest cavity."
}

# Image preprocessing pipeline (ImageNet normalization)
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class CheXNetModel:
    """
    CheXNet-style DenseNet121 model for chest X-ray disease detection.
    
    Supports 14 thoracic disease classes from the NIH ChestX-ray14 dataset.
    Uses ImageNet pretrained weights with a modified classifier head.
    """

    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize the CheXNet model.

        Args:
            model_path: Path to fine-tuned model weights (.pth file).
                       If None, uses ImageNet pretrained weights.
            device: Compute device ('cpu', 'cuda', or 'cuda:0').
                   If None, auto-detects GPU availability.
        """
        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Build the model
        self.model = self._build_model()

        # Load fine-tuned weights if available
        if model_path and os.path.exists(model_path):
            self._load_weights(model_path)
            logger.info(f"Loaded fine-tuned model weights from: {model_path}")
        else:
            logger.info("Using ImageNet pretrained weights (no fine-tuned weights found)")
            logger.info("For better accuracy, train the model using train.py")

        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

        logger.info(f"CheXNet model initialized successfully on {self.device}")

    def _build_model(self) -> nn.Module:
        """Build DenseNet121 with modified classifier for 14 disease classes."""
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        # Replace classifier for 14-class multi-label classification
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_features, 14),
            nn.Sigmoid()  # Sigmoid for multi-label classification
        )

        return model

    def _load_weights(self, model_path: str) -> None:
        """Load fine-tuned model weights from a .pth file."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
            logger.info("Falling back to ImageNet pretrained weights")

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess an image for model inference.

        Args:
            image_path: Path to the input image file.

        Returns:
            Preprocessed image tensor with batch dimension.
        """
        image = Image.open(image_path).convert("RGB")
        image_tensor = IMAGE_TRANSFORM(image)
        # Add batch dimension
        return image_tensor.unsqueeze(0)

    @torch.no_grad()
    def predict(self, image_path: str, top_k: int = 5) -> List[Dict]:
        """
        Run prediction on a chest X-ray image.

        Args:
            image_path: Path to the chest X-ray image.
            top_k: Number of top predictions to return.

        Returns:
            List of prediction dicts sorted by confidence (descending).
            Each dict contains: disease, confidence (%), description.
        """
        # Preprocess
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)

        # Inference
        outputs = self.model(image_tensor)
        probabilities = outputs.squeeze().cpu().numpy()

        # Build results sorted by confidence
        results = []
        for idx in np.argsort(probabilities)[::-1][:top_k]:
            disease_name = DISEASE_CLASSES[idx]
            confidence = float(probabilities[idx]) * 100  # Convert to percentage

            results.append({
                "disease": disease_name,
                "confidence": round(confidence, 2),
                "description": DISEASE_DESCRIPTIONS.get(disease_name, "No description available.")
            })

        return results


# Global model instance
_model_instance: CheXNetModel = None


def get_model() -> CheXNetModel:
    """Get or create the global model instance (singleton)."""
    global _model_instance
    if _model_instance is None:
        model_path = os.environ.get("MODEL_PATH", "models/model.pth")
        device = os.environ.get("DEVICE", None)
        _model_instance = CheXNetModel(model_path=model_path, device=device)
    return _model_instance


def predict_image(image_path: str, top_k: int = 5) -> List[Dict]:
    """
    Convenience function to predict diseases from a chest X-ray image.

    Args:
        image_path: Path to the chest X-ray image.
        top_k: Number of top predictions to return.

    Returns:
        List of prediction results.
    """
    model = get_model()
    return model.predict(image_path, top_k=top_k)
