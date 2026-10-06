"""
Training script for DenseNet121 (CheXNet) on the NIH ChestX-ray14 dataset.

Usage:
    python train.py --data_dir data/ --epochs 20 --batch_size 32 --lr 0.0001

Dataset structure expected:
    data/
    ├── images/           # All chest X-ray images
    └── Data_Entry_2017.csv  # Labels file from NIH
"""

import os
import argparse
import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import roc_auc_score

try:
    from tqdm import tqdm
except ImportError:
    print("Install tqdm for progress bars: pip install tqdm")
    tqdm = lambda x, **kwargs: x

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("TensorBoard not available. Install with: pip install tensorboard")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Disease classes
DISEASE_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
NUM_CLASSES = len(DISEASE_CLASSES)


class ChestXrayDataset(Dataset):
    """
    Custom dataset for NIH ChestX-ray14.
    
    Reads the Data_Entry_2017.csv file and loads corresponding images
    with multi-label binary targets.
    """

    def __init__(self, image_dir: str, csv_path: str, transform=None):
        """
        Args:
            image_dir: Path to directory containing X-ray images.
            csv_path: Path to Data_Entry_2017.csv file.
            transform: Optional image transform pipeline.
        """
        import pandas as pd

        self.image_dir = image_dir
        self.transform = transform

        # Load CSV
        df = pd.read_csv(csv_path)
        self.image_names = df['Image Index'].tolist()
        self.labels_raw = df['Finding Labels'].tolist()

        # Build multi-label binary targets
        self.targets = []
        for label_str in self.labels_raw:
            target = np.zeros(NUM_CLASSES, dtype=np.float32)
            findings = [f.strip() for f in label_str.split('|')]
            for finding in findings:
                finding_clean = finding.replace(' ', '_')
                if finding_clean in DISEASE_CLASSES:
                    idx = DISEASE_CLASSES.index(finding_clean)
                    target[idx] = 1.0
            self.targets.append(target)

        logger.info(f"Loaded {len(self.image_names)} images from {csv_path}")

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        target = torch.FloatTensor(self.targets[idx])
        return image, target


def build_model(pretrained: bool = True) -> nn.Module:
    """Build DenseNet121 model with modified classifier for 14 classes."""
    if pretrained:
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    else:
        model = models.densenet121(weights=None)

    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, NUM_CLASSES),
        nn.Sigmoid()
    )
    return model


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Get training and validation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def compute_auc(targets: np.ndarray, predictions: np.ndarray) -> dict:
    """Compute AUC-ROC for each disease class."""
    aucs = {}
    for i, disease in enumerate(DISEASE_CLASSES):
        try:
            if len(np.unique(targets[:, i])) > 1:
                auc = roc_auc_score(targets[:, i], predictions[:, i])
                aucs[disease] = auc
            else:
                aucs[disease] = float('nan')
        except ValueError:
            aucs[disease] = float('nan')
    return aucs


def train_one_epoch(model, dataloader, criterion, optimizer, device) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, targets in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, criterion, device) -> Tuple[float, dict]:
    """Validate model and return loss and per-class AUC."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_targets = []
    all_predictions = []

    for images, targets in tqdm(dataloader, desc="Validating", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        num_batches += 1

        all_targets.append(targets.cpu().numpy())
        all_predictions.append(outputs.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    aucs = compute_auc(all_targets, all_predictions)
    avg_loss = total_loss / max(num_batches, 1)

    return avg_loss, aucs


def main():
    parser = argparse.ArgumentParser(description="Train CheXNet (DenseNet121) for Lung Disease Detection")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="backend/models", help="Path to save model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--no_pretrained", action="store_true", help="Train from scratch")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Paths
    image_dir = os.path.join(args.data_dir, "images")
    csv_path = os.path.join(args.data_dir, "Data_Entry_2017.csv")
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        logger.info("Download the NIH ChestX-ray14 dataset from:")
        logger.info("https://nihcc.app.box.com/v/ChestXray-NIHCC")
        return

    # Transforms
    train_transform, val_transform = get_transforms()

    # Dataset & splits
    logger.info("Loading dataset...")
    full_dataset = ChestXrayDataset(image_dir, csv_path, transform=train_transform)

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Override val transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    logger.info(f"Train: {train_size} | Val: {val_size}")

    # Model
    model = build_model(pretrained=not args.no_pretrained).to(device)
    logger.info("Model built: DenseNet121 with 14-class classifier")

    # Loss, optimizer, scheduler
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # TensorBoard
    writer = SummaryWriter(os.path.join(args.output_dir, "runs")) if HAS_TENSORBOARD else None

    # Training loop
    best_val_loss = float('inf')
    best_auc = 0.0

    logger.info(f"Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{args.epochs} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, aucs = validate(model, val_loader, criterion, device)

        # Mean AUC (excluding NaN)
        valid_aucs = [v for v in aucs.values() if not np.isnan(v)]
        mean_auc = np.mean(valid_aucs) if valid_aucs else 0.0

        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Mean AUC: {mean_auc:.4f}")

        # Per-class AUC
        for disease, auc in aucs.items():
            logger.info(f"  {disease:.<25} AUC: {auc:.4f}" if not np.isnan(auc) else f"  {disease:.<25} AUC: N/A")

        # Scheduler step
        scheduler.step(val_loss)

        # TensorBoard logging
        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("AUC/mean", mean_auc, epoch)
            for disease, auc in aucs.items():
                if not np.isnan(auc):
                    writer.add_scalar(f"AUC/{disease}", auc, epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_auc = mean_auc
            checkpoint_path = os.path.join(args.output_dir, "model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "mean_auc": mean_auc,
                "disease_aucs": aucs,
            }, checkpoint_path)
            logger.info(f"✓ Saved best model (val_loss: {val_loss:.4f}, AUC: {mean_auc:.4f})")

    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete! Best Val Loss: {best_val_loss:.4f} | Best AUC: {best_auc:.4f}")
    logger.info(f"Model saved to: {os.path.join(args.output_dir, 'model.pth')}")

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
