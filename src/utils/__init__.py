"""Utility functions for the iris recognition system."""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def setup_logging(config: DictConfig) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        config: Configuration dictionary containing logging settings.
        
    Returns:
        Configured logger instance.
    """
    # Create logs directory if it doesn't exist
    log_file = Path(config.logging.file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format=config.logging.format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config: DictConfig) -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        config: Configuration dictionary containing device settings.
        
    Returns:
        PyTorch device object.
    """
    preferred = config.device.preferred.lower()
    fallback_order = config.device.fallback_order
    
    for device_name in fallback_order:
        device_name = device_name.lower()
        if device_name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device_name == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif device_name == "cpu":
            return torch.device("cpu")
    
    # Default fallback
    return torch.device("cpu")


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        OmegaConf configuration object.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, output_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save.
        output_path: Path where to save the configuration.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_path)


def anonymize_identifier(identifier: str) -> str:
    """Anonymize an identifier by hashing it.
    
    Args:
        identifier: Original identifier to anonymize.
        
    Returns:
        Hashed identifier.
    """
    import hashlib
    return hashlib.sha256(identifier.encode()).hexdigest()[:16]


def validate_image_path(image_path: Union[str, Path]) -> bool:
    """Validate that an image path exists and has a supported format.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        True if the image path is valid, False otherwise.
    """
    image_path = Path(image_path)
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    if not image_path.exists():
        return False
    
    if image_path.suffix.lower() not in supported_formats:
        return False
    
    return True


def create_directory_structure(base_path: Union[str, Path]) -> None:
    """Create the standard directory structure for the project.
    
    Args:
        base_path: Base path where to create the structure.
    """
    base_path = Path(base_path)
    directories = [
        "data/raw",
        "data/processed", 
        "data/train",
        "data/test",
        "data/validation",
        "models",
        "logs",
        "assets/plots",
        "assets/models",
        "configs",
        "scripts",
        "notebooks",
        "tests",
        "demo"
    ]
    
    for directory in directories:
        (base_path / directory).mkdir(parents=True, exist_ok=True)


def calculate_file_hash(file_path: Union[str, Path]) -> str:
    """Calculate SHA256 hash of a file.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        SHA256 hash of the file.
    """
    import hashlib
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    
    return hash_sha256.hexdigest()


def format_confidence_score(score: float, decimals: int = 3) -> str:
    """Format a confidence score for display.
    
    Args:
        score: Confidence score to format.
        decimals: Number of decimal places.
        
    Returns:
        Formatted confidence score string.
    """
    return f"{score:.{decimals}f}"


def get_system_info() -> Dict[str, Any]:
    """Get system information for logging and debugging.
    
    Returns:
        Dictionary containing system information.
    """
    import platform
    import sys
    
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }
    
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    
    return info
