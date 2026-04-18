#!/usr/bin/env python3
"""Script to generate synthetic iris data for testing."""

import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Tuple

import cv2
from PIL import Image, ImageDraw


def generate_synthetic_iris(
    size: Tuple[int, int] = (256, 256),
    num_rings: int = 8,
    noise_level: float = 0.1,
    seed: int = None
) -> np.ndarray:
    """Generate a synthetic iris image.
    
    Args:
        size: Image size (height, width).
        num_rings: Number of concentric rings.
        noise_level: Level of noise to add.
        seed: Random seed for reproducibility.
        
    Returns:
        Synthetic iris image as numpy array.
    """
    if seed is not None:
        np.random.seed(seed)
    
    height, width = size
    center_x, center_y = width // 2, height // 2
    
    # Create base image
    image = np.zeros((height, width), dtype=np.float32)
    
    # Generate concentric rings
    for i in range(num_rings):
        radius = (i + 1) * min(height, width) // (2 * num_rings)
        
        # Create ring pattern
        y, x = np.ogrid[:height, :width]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Ring intensity varies with angle
        angle = np.arctan2(y - center_y, x - center_x)
        ring_intensity = 0.5 + 0.5 * np.sin(angle * (i + 1) * 2)
        
        # Apply ring
        mask = (distance >= radius - 2) & (distance <= radius + 2)
        image[mask] = ring_intensity[mask]
    
    # Add pupil (dark center)
    pupil_radius = min(height, width) // 8
    pupil_mask = np.sqrt((x - center_x)**2 + (y - center_y)**2) <= pupil_radius
    image[pupil_mask] = 0.1
    
    # Add noise
    noise = np.random.normal(0, noise_level, image.shape)
    image = np.clip(image + noise, 0, 1)
    
    # Add eyelid occlusion (top)
    eyelid_y = height // 4
    eyelid_mask = y < eyelid_y
    image[eyelid_mask] *= 0.3
    
    # Add eyelid occlusion (bottom)
    eyelid_y = 3 * height // 4
    eyelid_mask = y > eyelid_y
    image[eyelid_mask] *= 0.3
    
    return image


def generate_synthetic_dataset(
    output_dir: str,
    num_samples: int = 100,
    num_subjects: int = 10,
    samples_per_subject: int = 10
) -> None:
    """Generate a synthetic iris dataset.
    
    Args:
        output_dir: Output directory for the dataset.
        num_samples: Total number of samples to generate.
        num_subjects: Number of different subjects.
        samples_per_subject: Number of samples per subject.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    train_dir = output_path / "train"
    test_dir = output_path / "test"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    logging.info(f"Generating {num_samples} synthetic iris images...")
    
    subject_id = 0
    sample_id = 0
    
    for i in range(num_samples):
        # Generate synthetic iris
        iris_image = generate_synthetic_iris(seed=i)
        
        # Convert to uint8
        iris_image_uint8 = (iris_image * 255).astype(np.uint8)
        
        # Determine if training or test
        is_test = (i % 5 == 0)  # 20% test data
        target_dir = test_dir if is_test else train_dir
        
        # Create filename
        filename = f"subject_{subject_id:03d}_sample_{sample_id:02d}.png"
        filepath = target_dir / filename
        
        # Save image
        cv2.imwrite(str(filepath), iris_image_uint8)
        
        # Update counters
        sample_id += 1
        if sample_id >= samples_per_subject:
            sample_id = 0
            subject_id += 1
        
        if (i + 1) % 10 == 0:
            logging.info(f"Generated {i + 1}/{num_samples} images")
    
    # Create metadata file
    metadata = {
        "num_samples": num_samples,
        "num_subjects": num_subjects,
        "samples_per_subject": samples_per_subject,
        "image_size": [256, 256],
        "format": "PNG",
        "description": "Synthetic iris dataset for testing iris recognition system"
    }
    
    import json
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Dataset generated successfully in {output_dir}")
    logging.info(f"Training samples: {len(list(train_dir.glob('*.png')))}")
    logging.info(f"Test samples: {len(list(test_dir.glob('*.png')))}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate synthetic iris dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/synthetic",
        help="Output directory for the dataset"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Total number of samples to generate"
    )
    parser.add_argument(
        "--num_subjects",
        type=int,
        default=10,
        help="Number of different subjects"
    )
    parser.add_argument(
        "--samples_per_subject",
        type=int,
        default=10,
        help="Number of samples per subject"
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
    
    # Generate dataset
    generate_synthetic_dataset(
        args.output_dir,
        args.num_samples,
        args.num_subjects,
        args.samples_per_subject
    )


if __name__ == "__main__":
    main()
