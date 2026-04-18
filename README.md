# Iris Recognition System

A research-focused iris recognition system for biometric authentication education and defensive security research.

## DISCLAIMER

**This is a research and educational demonstration system. It is NOT intended for production security operations or real-world biometric authentication. The system may be inaccurate and should not be used for actual security decisions.**

## Overview

This project implements a comprehensive iris recognition system with modern computer vision techniques, including:

- Advanced iris segmentation and preprocessing
- Gabor wavelet feature extraction
- Phase encoding and template matching
- Biometric evaluation metrics (EER, minDCF, ROC/DET curves)
- Interactive demo for enrollment and authentication workflows
- Privacy-preserving techniques and ethical considerations

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Iris-Recognition-Implementation.git
cd Iris-Recognition-Implementation

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.models.iris_recognizer import IrisRecognizer

# Initialize the recognizer
recognizer = IrisRecognizer()

# Enroll a new iris
template = recognizer.enroll("path/to/enrolled_iris.jpg")

# Authenticate against enrolled template
score = recognizer.authenticate("path/to/input_iris.jpg", template)
is_match = recognizer.is_match(score)
```

### Demo Application

```bash
# Launch the Streamlit demo
streamlit run demo/app.py
```

## Dataset Schema

The system expects iris images in standard formats (JPG, PNG) with the following characteristics:

- **Resolution**: Minimum 200x200 pixels (higher resolution preferred)
- **Format**: Grayscale or color images (automatically converted)
- **Quality**: Clear iris patterns with minimal occlusion
- **Privacy**: All sample data is synthetic or anonymized

### Data Generation

Synthetic iris datasets can be generated using:

```bash
python scripts/generate_synthetic_data.py --output_dir data/synthetic --num_samples 1000
```

## Training and Evaluation

### Training a Model

```bash
python scripts/train.py --config configs/training_config.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --model_path models/best_model.pth --test_data data/test/
```

## Metrics and Performance

The system provides comprehensive biometric evaluation metrics:

- **EER (Equal Error Rate)**: Primary biometric accuracy metric
- **minDCF (Minimum Detection Cost Function)**: Cost-sensitive evaluation
- **ROC/DET Curves**: Visualization of performance trade-offs
- **FAR/FRR**: False Accept/Reject Rates at operating points
- **Template Size**: Storage efficiency metrics

## Limitations

- **Research Focus**: Designed for educational purposes, not production use
- **Accuracy**: May not match commercial iris recognition systems
- **Privacy**: Ensure compliance with local biometric data regulations
- **Security**: This is a defensive research tool, not a security product

## Ethical Considerations

- **Consent**: Always obtain explicit consent for biometric data collection
- **Data Retention**: Implement appropriate data retention policies
- **Privacy**: Use privacy-preserving techniques where applicable
- **Bias**: Be aware of potential demographic bias in biometric systems
- **Surveillance**: Consider implications of biometric surveillance systems

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For questions or issues, please open a GitHub issue or contact the research team.
# Iris-Recognition-System
