#!/usr/bin/env python3
"""Training script for iris recognition models."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import create_data_loaders
from src.models import IrisCNN, IrisModelTrainer
from src.utils import get_device, load_config, set_seed, setup_logging


def train_model(
    config_path: str,
    data_dir: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> Dict[str, Any]:
    """Train an iris recognition model.
    
    Args:
        config_path: Path to configuration file.
        data_dir: Directory containing training data.
        output_dir: Directory to save trained model.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate for optimizer.
        
    Returns:
        Training history dictionary.
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logging(config)
    
    # Set random seed
    set_seed(42)
    
    # Get device
    device = get_device(config)
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dir=data_dir,
        test_dir=data_dir,  # Using same directory for simplicity
        batch_size=batch_size,
        num_workers=4,
        validation_split=0.2
    )
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Initialize model
    model = IrisCNN(
        input_channels=1,
        num_classes=1000,  # Large number for embedding space
        embedding_dim=512
    )
    
    # Initialize trainer
    trainer = IrisModelTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate
    )
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping_patience=10
    )
    
    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "best_model.pth"
    trainer.save_model(str(model_path))
    
    logger.info(f"Model saved to {model_path}")
    
    return history


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train iris recognition model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing training data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Train model
    history = train_model(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    print("✅ Training completed successfully!")
    print(f"Final training accuracy: {history['train_accuracies'][-1]:.2f}%")
    print(f"Final validation accuracy: {history['val_accuracies'][-1]:.2f}%")


if __name__ == "__main__":
    main()
