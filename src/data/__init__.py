"""Data preprocessing and augmentation utilities for iris recognition."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)


class IrisPreprocessor:
    """Iris image preprocessing pipeline."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        normalize: bool = True,
        enhance_contrast: bool = True
    ):
        """Initialize the iris preprocessor.
        
        Args:
            target_size: Target size for resizing images (height, width).
            normalize: Whether to normalize pixel values to [0, 1].
            enhance_contrast: Whether to enhance contrast using CLAHE.
        """
        self.target_size = target_size
        self.normalize = normalize
        self.enhance_contrast = enhance_contrast
        
        # Initialize CLAHE for contrast enhancement
        if enhance_contrast:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def preprocess(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Preprocess an iris image.
        
        Args:
            image: Input image as file path or numpy array.
            
        Returns:
            Preprocessed image as numpy array.
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image: {image}")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize image
        image = cv2.resize(image, self.target_size[::-1])  # OpenCV uses (width, height)
        
        # Enhance contrast if enabled
        if self.enhance_contrast:
            image = self.clahe.apply(image)
        
        # Normalize if enabled
        if self.normalize:
            image = image.astype(np.float32) / 255.0
        
        return image
    
    def detect_iris_region(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """Detect iris region using circular Hough transform.
        
        Args:
            image: Preprocessed iris image.
            
        Returns:
            Tuple of (x, y, radius) for the detected iris circle.
        """
        # Convert to uint8 if normalized
        if image.dtype == np.float32:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image_uint8, (5, 5), 0)
        
        # Detect circles using Hough transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=30,
            maxRadius=min(image.shape) // 2
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Return the largest circle (most likely iris)
            largest_circle = circles[np.argmax(circles[:, 2])]
            return largest_circle[0], largest_circle[1], largest_circle[2]
        else:
            # Fallback: assume iris is in the center
            h, w = image.shape
            center_x, center_y = w // 2, h // 2
            radius = min(w, h) // 3
            return center_x, center_y, radius


class IrisAugmentation:
    """Data augmentation for iris images."""
    
    def __init__(
        self,
        rotation_range: float = 15.0,
        brightness_range: float = 0.2,
        contrast_range: float = 0.2,
        noise_level: float = 0.01,
        blur_probability: float = 0.1
    ):
        """Initialize augmentation parameters.
        
        Args:
            rotation_range: Maximum rotation angle in degrees.
            brightness_range: Range for brightness adjustment.
            contrast_range: Range for contrast adjustment.
            noise_level: Standard deviation for Gaussian noise.
            blur_probability: Probability of applying blur.
        """
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_level = noise_level
        self.blur_probability = blur_probability
    
    def augment(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentations to an image.
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            Augmented image.
        """
        augmented = image.copy()
        
        # Random rotation
        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            h, w = augmented.shape
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented = cv2.warpAffine(augmented, rotation_matrix, (w, h))
        
        # Random brightness adjustment
        if self.brightness_range > 0:
            brightness_factor = np.random.uniform(
                1 - self.brightness_range, 
                1 + self.brightness_range
            )
            augmented = np.clip(augmented * brightness_factor, 0, 1)
        
        # Random contrast adjustment
        if self.contrast_range > 0:
            contrast_factor = np.random.uniform(
                1 - self.contrast_range, 
                1 + self.contrast_range
            )
            mean = np.mean(augmented)
            augmented = np.clip((augmented - mean) * contrast_factor + mean, 0, 1)
        
        # Random Gaussian noise
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, augmented.shape)
            augmented = np.clip(augmented + noise, 0, 1)
        
        # Random blur
        if np.random.random() < self.blur_probability:
            kernel_size = np.random.choice([3, 5])
            augmented = cv2.GaussianBlur(augmented, (kernel_size, kernel_size), 0)
        
        return augmented


class IrisDataset(Dataset):
    """Dataset class for iris images."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        preprocessor: Optional[IrisPreprocessor] = None,
        augmentation: Optional[IrisAugmentation] = None,
        is_training: bool = True
    ):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing iris images.
            preprocessor: Image preprocessor instance.
            augmentation: Data augmentation instance.
            is_training: Whether this is training data (affects augmentation).
        """
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor or IrisPreprocessor()
        self.augmentation = augmentation
        self.is_training = is_training
        
        # Find all image files
        self.image_paths = self._find_image_files()
        
        if len(self.image_paths) == 0:
            logger.warning(f"No images found in {data_dir}")
    
    def _find_image_files(self) -> List[Path]:
        """Find all image files in the data directory.
        
        Returns:
            List of image file paths.
        """
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = []
        
        for ext in supported_extensions:
            image_paths.extend(self.data_dir.glob(f"*{ext}"))
            image_paths.extend(self.data_dir.glob(f"*{ext.upper()}"))
        
        return sorted(image_paths)
    
    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """Get an item from the dataset.
        
        Args:
            idx: Index of the item.
            
        Returns:
            Tuple of (preprocessed_image, filename).
        """
        image_path = self.image_paths[idx]
        
        # Preprocess the image
        image = self.preprocessor.preprocess(image_path)
        
        # Apply augmentation if training
        if self.is_training and self.augmentation is not None:
            image = self.augmentation.augment(image)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        
        return image_tensor, image_path.stem


def create_data_loaders(
    train_dir: Union[str, Path],
    test_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    validation_split: float = 0.2
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create data loaders for training, validation, and testing.
    
    Args:
        train_dir: Directory containing training images.
        test_dir: Directory containing test images.
        batch_size: Batch size for data loaders.
        num_workers: Number of worker processes.
        validation_split: Fraction of training data to use for validation.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Create preprocessor and augmentation
    preprocessor = IrisPreprocessor()
    augmentation = IrisAugmentation()
    
    # Create datasets
    train_dataset = IrisDataset(train_dir, preprocessor, augmentation, is_training=True)
    test_dataset = IrisDataset(test_dir, preprocessor, None, is_training=False)
    
    # Split training data into train/validation
    train_size = int((1 - validation_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset_split, val_dataset_split = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset_split,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset_split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
