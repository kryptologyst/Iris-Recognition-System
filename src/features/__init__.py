"""Feature extraction methods for iris recognition."""

import logging
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cosine, euclidean

logger = logging.getLogger(__name__)


class GaborFilterBank:
    """Bank of Gabor filters for iris feature extraction."""
    
    def __init__(
        self,
        orientations: int = 8,
        frequencies: List[float] = None,
        kernel_size: int = 31,
        sigma: float = 2.0
    ):
        """Initialize Gabor filter bank.
        
        Args:
            orientations: Number of orientation angles.
            frequencies: List of frequencies for filters.
            kernel_size: Size of the filter kernel.
            sigma: Standard deviation of the Gaussian envelope.
        """
        self.orientations = orientations
        self.frequencies = frequencies or [0.1, 0.2, 0.3, 0.4]
        self.kernel_size = kernel_size
        self.sigma = sigma
        
        # Generate filter bank
        self.filters = self._generate_filters()
    
    def _generate_filters(self) -> List[np.ndarray]:
        """Generate the Gabor filter bank.
        
        Returns:
            List of Gabor filter kernels.
        """
        filters = []
        
        for frequency in self.frequencies:
            for orientation in range(self.orientations):
                angle = orientation * np.pi / self.orientations
                
                # Create Gabor kernel
                kernel = cv2.getGaborKernel(
                    (self.kernel_size, self.kernel_size),
                    self.sigma,
                    angle,
                    2 * np.pi * frequency,
                    0.5,  # gamma
                    0,    # psi
                    ktype=cv2.CV_32F
                )
                
                filters.append(kernel)
        
        return filters
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract Gabor features from an image.
        
        Args:
            image: Input iris image.
            
        Returns:
            Concatenated Gabor features.
        """
        features = []
        
        for kernel in self.filters:
            # Apply Gabor filter
            filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
            
            # Compute magnitude
            magnitude = np.abs(filtered)
            
            # Compute statistics
            mean_val = np.mean(magnitude)
            std_val = np.std(magnitude)
            
            features.extend([mean_val, std_val])
        
        return np.array(features)


class PhaseEncoder:
    """Phase encoding for iris template generation."""
    
    def __init__(self, quantization_levels: int = 2):
        """Initialize phase encoder.
        
        Args:
            quantization_levels: Number of quantization levels for phase.
        """
        self.quantization_levels = quantization_levels
    
    def encode(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode iris image using phase information.
        
        Args:
            image: Input iris image.
            
        Returns:
            Tuple of (phase_code, mask) where mask indicates valid regions.
        """
        # Convert to complex representation
        complex_image = image.astype(np.complex128)
        
        # Compute phase
        phase = np.angle(complex_image)
        
        # Quantize phase
        phase_quantized = self._quantize_phase(phase)
        
        # Create mask for valid regions (non-zero pixels)
        mask = (image > 0).astype(np.uint8)
        
        return phase_quantized, mask
    
    def _quantize_phase(self, phase: np.ndarray) -> np.ndarray:
        """Quantize phase values.
        
        Args:
            phase: Phase values in radians.
            
        Returns:
            Quantized phase values.
        """
        # Normalize phase to [0, 2π]
        phase_normalized = (phase + np.pi) / (2 * np.pi)
        
        # Quantize
        quantized = np.floor(phase_normalized * self.quantization_levels)
        quantized = np.clip(quantized, 0, self.quantization_levels - 1)
        
        return quantized.astype(np.uint8)


class IrisFeatureExtractor:
    """Main feature extractor for iris recognition."""
    
    def __init__(
        self,
        gabor_orientations: int = 8,
        gabor_frequencies: List[float] = None,
        phase_quantization: int = 2,
        histogram_bins: int = 32
    ):
        """Initialize feature extractor.
        
        Args:
            gabor_orientations: Number of Gabor orientations.
            gabor_frequencies: List of Gabor frequencies.
            phase_quantization: Phase quantization levels.
            histogram_bins: Number of histogram bins.
        """
        self.gabor_bank = GaborFilterBank(
            orientations=gabor_orientations,
            frequencies=gabor_frequencies
        )
        self.phase_encoder = PhaseEncoder(quantization_levels=phase_quantization)
        self.histogram_bins = histogram_bins
    
    def extract_all_features(self, image: np.ndarray) -> dict:
        """Extract all types of features from iris image.
        
        Args:
            image: Preprocessed iris image.
            
        Returns:
            Dictionary containing different feature types.
        """
        features = {}
        
        # Gabor features
        features['gabor'] = self.gabor_bank.extract_features(image)
        
        # Phase encoding
        phase_code, mask = self.phase_encoder.encode(image)
        features['phase_code'] = phase_code
        features['mask'] = mask
        
        # Histogram features
        features['histogram'] = self._extract_histogram_features(image)
        
        # Texture features
        features['texture'] = self._extract_texture_features(image)
        
        # Shape features
        features['shape'] = self._extract_shape_features(image)
        
        return features
    
    def _extract_histogram_features(self, image: np.ndarray) -> np.ndarray:
        """Extract histogram-based features.
        
        Args:
            image: Input image.
            
        Returns:
            Histogram features.
        """
        # Compute histogram
        hist, _ = np.histogram(image, bins=self.histogram_bins, range=(0, 1))
        
        # Normalize
        hist = hist / np.sum(hist)
        
        # Compute statistics
        mean_val = np.mean(image)
        std_val = np.std(image)
        skewness = self._compute_skewness(image)
        kurtosis = self._compute_kurtosis(image)
        
        return np.concatenate([hist, [mean_val, std_val, skewness, kurtosis]])
    
    def _extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Extract texture features using Local Binary Patterns.
        
        Args:
            image: Input image.
            
        Returns:
            Texture features.
        """
        # Convert to uint8
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Compute LBP
        lbp = self._compute_lbp(image_uint8)
        
        # Compute LBP histogram
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)
        
        return hist
    
    def _extract_shape_features(self, image: np.ndarray) -> np.ndarray:
        """Extract shape-based features.
        
        Args:
            image: Input image.
            
        Returns:
            Shape features.
        """
        # Convert to binary
        binary = (image > 0.5).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return np.zeros(5)
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Compute shape features
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Circularity
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Convex hull
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        return np.array([area, perimeter, circularity, aspect_ratio, solidity])
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data."""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0
        return np.mean(((data - mean_val) / std_val) ** 3)
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data."""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0
        return np.mean(((data - mean_val) / std_val) ** 4) - 3
    
    def _compute_lbp(self, image: np.ndarray) -> np.ndarray:
        """Compute Local Binary Pattern."""
        h, w = image.shape
        lbp = np.zeros_like(image)
        
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = image[i, j]
                binary_string = ""
                
                # 8-neighborhood
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                
                for neighbor in neighbors:
                    binary_string += "1" if neighbor >= center else "0"
                
                lbp[i, j] = int(binary_string, 2)
        
        return lbp


class IrisTemplate:
    """Iris template for storage and comparison."""
    
    def __init__(self, features: dict, identifier: str = None):
        """Initialize iris template.
        
        Args:
            features: Extracted features dictionary.
            identifier: Optional identifier for the template.
        """
        self.features = features
        self.identifier = identifier
        self.template_vector = self._create_template_vector()
    
    def _create_template_vector(self) -> np.ndarray:
        """Create a single vector representation of all features.
        
        Returns:
            Concatenated feature vector.
        """
        vectors = []
        
        for feature_type, feature_data in self.features.items():
            if isinstance(feature_data, np.ndarray):
                vectors.append(feature_data.flatten())
        
        return np.concatenate(vectors) if vectors else np.array([])
    
    def compare(self, other: 'IrisTemplate', method: str = 'cosine') -> float:
        """Compare this template with another.
        
        Args:
            other: Other iris template to compare with.
            method: Comparison method ('cosine', 'euclidean', 'hamming').
            
        Returns:
            Similarity score.
        """
        if method == 'cosine':
            return 1 - cosine(self.template_vector, other.template_vector)
        elif method == 'euclidean':
            distance = euclidean(self.template_vector, other.template_vector)
            return 1 / (1 + distance)  # Convert to similarity
        elif method == 'hamming':
            # For binary features (phase codes)
            if 'phase_code' in self.features and 'phase_code' in other.features:
                mask1 = self.features['mask']
                mask2 = other.features['mask']
                code1 = self.features['phase_code']
                code2 = other.features['phase_code']
                
                # Only compare valid regions
                valid_mask = mask1 & mask2
                if np.sum(valid_mask) == 0:
                    return 0.0
                
                hamming_distance = np.sum((code1 != code2) & valid_mask) / np.sum(valid_mask)
                return 1 - hamming_distance
            else:
                return 0.0
        else:
            raise ValueError(f"Unknown comparison method: {method}")
    
    def get_template_size(self) -> int:
        """Get the size of the template vector.
        
        Returns:
            Size of the template vector.
        """
        return len(self.template_vector)
    
    def save(self, filepath: str) -> None:
        """Save template to file.
        
        Args:
            filepath: Path to save the template.
        """
        np.savez(filepath, **self.features, identifier=self.identifier)
    
    @classmethod
    def load(cls, filepath: str) -> 'IrisTemplate':
        """Load template from file.
        
        Args:
            filepath: Path to load the template from.
            
        Returns:
            Loaded iris template.
        """
        data = np.load(filepath, allow_pickle=True)
        features = {key: data[key] for key in data.files if key != 'identifier'}
        identifier = str(data['identifier']) if 'identifier' in data.files else None
        
        return cls(features, identifier)
