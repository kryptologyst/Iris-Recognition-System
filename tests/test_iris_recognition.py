"""Test suite for the iris recognition system."""

import numpy as np
import pytest
import torch
from pathlib import Path
import tempfile
import shutil

from src.data import IrisPreprocessor, IrisAugmentation, IrisDataset
from src.features import IrisFeatureExtractor, IrisTemplate, GaborFilterBank, PhaseEncoder
from src.models import IrisRecognizer, IrisCNN, IrisSiameseNetwork
from src.eval import BiometricEvaluator
from src.utils import set_seed, get_device, anonymize_identifier, validate_image_path


class TestIrisPreprocessor:
    """Test cases for iris preprocessing."""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = IrisPreprocessor()
        assert preprocessor.target_size == (256, 256)
        assert preprocessor.normalize is True
        assert preprocessor.enhance_contrast is True
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        preprocessor = IrisPreprocessor()
        
        # Create synthetic image
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # Preprocess
        processed = preprocessor.preprocess(image)
        
        # Check output
        assert processed.shape == (256, 256)
        assert processed.dtype == np.float32
        assert processed.min() >= 0.0
        assert processed.max() <= 1.0
    
    def test_iris_region_detection(self):
        """Test iris region detection."""
        preprocessor = IrisPreprocessor()
        
        # Create synthetic iris image
        image = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(image, (128, 128), 80, 255, -1)  # Iris
        cv2.circle(image, (128, 128), 30, 0, -1)    # Pupil
        
        # Detect iris region
        x, y, radius = preprocessor.detect_iris_region(image)
        
        # Check results
        assert isinstance(x, int)
        assert isinstance(y, int)
        assert isinstance(radius, int)
        assert radius > 0


class TestIrisAugmentation:
    """Test cases for data augmentation."""
    
    def test_augmentation_initialization(self):
        """Test augmentation initialization."""
        augmentation = IrisAugmentation()
        assert augmentation.rotation_range == 15.0
        assert augmentation.brightness_range == 0.2
        assert augmentation.contrast_range == 0.2
        assert augmentation.noise_level == 0.01
    
    def test_augment_image(self):
        """Test image augmentation."""
        augmentation = IrisAugmentation()
        
        # Create synthetic image
        image = np.random.random((256, 256)).astype(np.float32)
        
        # Augment
        augmented = augmentation.augment(image)
        
        # Check output
        assert augmented.shape == image.shape
        assert augmented.dtype == np.float32
        assert augmented.min() >= 0.0
        assert augmented.max() <= 1.0


class TestGaborFilterBank:
    """Test cases for Gabor filter bank."""
    
    def test_filter_bank_initialization(self):
        """Test filter bank initialization."""
        filter_bank = GaborFilterBank()
        assert filter_bank.orientations == 8
        assert len(filter_bank.frequencies) == 4
        assert filter_bank.kernel_size == 31
    
    def test_feature_extraction(self):
        """Test feature extraction."""
        filter_bank = GaborFilterBank()
        
        # Create synthetic image
        image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        
        # Extract features
        features = filter_bank.extract_features(image)
        
        # Check output
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert features.dtype == np.float64


class TestPhaseEncoder:
    """Test cases for phase encoding."""
    
    def test_phase_encoder_initialization(self):
        """Test phase encoder initialization."""
        encoder = PhaseEncoder()
        assert encoder.quantization_levels == 2
    
    def test_phase_encoding(self):
        """Test phase encoding."""
        encoder = PhaseEncoder()
        
        # Create synthetic image
        image = np.random.random((256, 256)).astype(np.float32)
        
        # Encode
        phase_code, mask = encoder.encode(image)
        
        # Check output
        assert phase_code.shape == image.shape
        assert mask.shape == image.shape
        assert phase_code.dtype == np.uint8
        assert mask.dtype == np.uint8


class TestIrisFeatureExtractor:
    """Test cases for iris feature extractor."""
    
    def test_feature_extractor_initialization(self):
        """Test feature extractor initialization."""
        extractor = IrisFeatureExtractor()
        assert extractor.gabor_bank is not None
        assert extractor.phase_encoder is not None
        assert extractor.histogram_bins == 32
    
    def test_feature_extraction(self):
        """Test feature extraction."""
        extractor = IrisFeatureExtractor()
        
        # Create synthetic image
        image = np.random.random((256, 256)).astype(np.float32)
        
        # Extract features
        features = extractor.extract_all_features(image)
        
        # Check output
        assert isinstance(features, dict)
        assert 'gabor' in features
        assert 'phase_code' in features
        assert 'mask' in features
        assert 'histogram' in features
        assert 'texture' in features
        assert 'shape' in features


class TestIrisTemplate:
    """Test cases for iris template."""
    
    def test_template_creation(self):
        """Test template creation."""
        # Create synthetic features
        features = {
            'gabor': np.random.random(64),
            'phase_code': np.random.randint(0, 2, (256, 256)),
            'mask': np.ones((256, 256), dtype=np.uint8),
            'histogram': np.random.random(32)
        }
        
        template = IrisTemplate(features, "test_user")
        
        assert template.identifier == "test_user"
        assert template.template_vector is not None
        assert len(template.template_vector) > 0
    
    def test_template_comparison(self):
        """Test template comparison."""
        # Create two templates
        features1 = {
            'gabor': np.random.random(64),
            'phase_code': np.random.randint(0, 2, (256, 256)),
            'mask': np.ones((256, 256), dtype=np.uint8),
            'histogram': np.random.random(32)
        }
        
        features2 = {
            'gabor': np.random.random(64),
            'phase_code': np.random.randint(0, 2, (256, 256)),
            'mask': np.ones((256, 256), dtype=np.uint8),
            'histogram': np.random.random(32)
        }
        
        template1 = IrisTemplate(features1, "user1")
        template2 = IrisTemplate(features2, "user2")
        
        # Compare templates
        similarity = template1.compare(template2, method='cosine')
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0


class TestIrisCNN:
    """Test cases for iris CNN model."""
    
    def test_cnn_initialization(self):
        """Test CNN initialization."""
        model = IrisCNN()
        assert model.embedding_dim == 512
    
    def test_cnn_forward_pass(self):
        """Test CNN forward pass."""
        model = IrisCNN()
        
        # Create synthetic input
        x = torch.randn(1, 1, 256, 256)
        
        # Forward pass
        output = model(x)
        
        # Check output
        assert output.shape == (1, 1000)  # num_classes
    
    def test_cnn_embedding(self):
        """Test CNN embedding extraction."""
        model = IrisCNN()
        
        # Create synthetic input
        x = torch.randn(1, 1, 256, 256)
        
        # Get embedding
        embedding = model.get_embedding(x)
        
        # Check output
        assert embedding.shape == (1, 512)  # embedding_dim


class TestIrisRecognizer:
    """Test cases for iris recognizer."""
    
    def test_recognizer_initialization(self):
        """Test recognizer initialization."""
        recognizer = IrisRecognizer()
        assert recognizer.threshold == 0.5
        assert recognizer.device is not None
    
    def test_enrollment(self):
        """Test iris enrollment."""
        recognizer = IrisRecognizer()
        
        # Create synthetic image
        image = np.random.random((256, 256)).astype(np.float32)
        
        # Enroll
        template = recognizer.enroll(image, "test_user")
        
        assert template.identifier == "test_user"
        assert template.get_template_size() > 0
    
    def test_authentication(self):
        """Test iris authentication."""
        recognizer = IrisRecognizer()
        
        # Create synthetic images
        image1 = np.random.random((256, 256)).astype(np.float32)
        image2 = np.random.random((256, 256)).astype(np.float32)
        
        # Enroll first image
        template = recognizer.enroll(image1, "test_user")
        
        # Authenticate second image
        similarity = recognizer.authenticate(image2, template)
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0


class TestBiometricEvaluator:
    """Test cases for biometric evaluator."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = BiometricEvaluator()
        assert len(evaluator.scores) == 0
        assert len(evaluator.labels) == 0
    
    def test_add_scores(self):
        """Test adding scores."""
        evaluator = BiometricEvaluator()
        
        scores = [0.8, 0.3, 0.9, 0.2]
        labels = [True, False, True, False]
        
        evaluator.add_scores(scores, labels)
        
        assert len(evaluator.scores) == 4
        assert len(evaluator.labels) == 4
    
    def test_eer_computation(self):
        """Test EER computation."""
        evaluator = BiometricEvaluator()
        
        # Add test data
        scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]  # First 3 genuine, last 3 impostor
        labels = [True, True, True, False, False, False]
        
        evaluator.add_scores(scores, labels)
        
        # Compute EER
        eer_threshold, eer_value = evaluator.compute_eer()
        
        assert isinstance(eer_threshold, float)
        assert isinstance(eer_value, float)
        assert 0.0 <= eer_value <= 1.0
    
    def test_roc_metrics(self):
        """Test ROC metrics computation."""
        evaluator = BiometricEvaluator()
        
        # Add test data
        scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        labels = [True, True, True, False, False, False]
        
        evaluator.add_scores(scores, labels)
        
        # Compute ROC metrics
        roc_metrics = evaluator.compute_roc_metrics()
        
        assert 'auc' in roc_metrics
        assert 'fpr' in roc_metrics
        assert 'tpr' in roc_metrics
        assert 0.0 <= roc_metrics['auc'] <= 1.0


class TestUtils:
    """Test cases for utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate random numbers
        np_rand1 = np.random.random()
        torch_rand1 = torch.rand(1).item()
        
        # Reset seed
        set_seed(42)
        
        # Generate random numbers again
        np_rand2 = np.random.random()
        torch_rand2 = torch.rand(1).item()
        
        # Should be the same
        assert np_rand1 == np_rand2
        assert torch_rand1 == torch_rand2
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_anonymize_identifier(self):
        """Test identifier anonymization."""
        identifier = "test_user_123"
        anonymized = anonymize_identifier(identifier)
        
        assert isinstance(anonymized, str)
        assert len(anonymized) == 16
        assert anonymized != identifier
    
    def test_validate_image_path(self):
        """Test image path validation."""
        # Test with non-existent file
        assert not validate_image_path("nonexistent.jpg")
        
        # Test with temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(b"fake image data")
            tmp_path = tmp.name
        
        try:
            assert validate_image_path(tmp_path)
        finally:
            Path(tmp_path).unlink()


# Integration tests
class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Initialize recognizer
        recognizer = IrisRecognizer()
        
        # Create synthetic images
        image1 = np.random.random((256, 256)).astype(np.float32)
        image2 = np.random.random((256, 256)).astype(np.float32)
        
        # Enroll first image
        template = recognizer.enroll(image1, "user1")
        
        # Authenticate second image
        similarity = recognizer.authenticate(image2, template)
        is_match = recognizer.is_match(similarity)
        
        # Check results
        assert isinstance(similarity, float)
        assert isinstance(is_match, bool)
        assert 0.0 <= similarity <= 1.0
    
    def test_evaluation_workflow(self):
        """Test evaluation workflow."""
        # Initialize evaluator
        evaluator = BiometricEvaluator()
        
        # Add test data
        scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        labels = [True, True, True, False, False, False]
        
        evaluator.add_scores(scores, labels)
        
        # Generate report
        report = evaluator.generate_report()
        
        # Check report structure
        assert 'eer' in report
        assert 'mindcf' in report
        assert 'roc' in report
        assert 'precision_recall' in report
        assert 'metrics' in report


if __name__ == "__main__":
    pytest.main([__file__])
