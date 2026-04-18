"""Iris Recognition System - A modern biometric authentication research platform."""

__version__ = "1.0.0"
__author__ = "Security Research Team"
__email__ = "research@example.com"

from .data import IrisDataset, IrisPreprocessor, IrisAugmentation
from .features import IrisFeatureExtractor, IrisTemplate, GaborFilterBank, PhaseEncoder
from .models import IrisRecognizer, IrisCNN, IrisSiameseNetwork, IrisModelTrainer
from .eval import BiometricEvaluator, IrisVisualizer
from .utils import (
    setup_logging, set_seed, get_device, load_config, save_config,
    anonymize_identifier, validate_image_path, create_directory_structure
)

__all__ = [
    # Data
    "IrisDataset",
    "IrisPreprocessor", 
    "IrisAugmentation",
    
    # Features
    "IrisFeatureExtractor",
    "IrisTemplate",
    "GaborFilterBank",
    "PhaseEncoder",
    
    # Models
    "IrisRecognizer",
    "IrisCNN",
    "IrisSiameseNetwork", 
    "IrisModelTrainer",
    
    # Evaluation
    "BiometricEvaluator",
    "IrisVisualizer",
    
    # Utils
    "setup_logging",
    "set_seed",
    "get_device",
    "load_config",
    "save_config",
    "anonymize_identifier",
    "validate_image_path",
    "create_directory_structure",
]
