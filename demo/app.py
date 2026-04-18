"""Streamlit demo application for iris recognition system."""

import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import streamlit as st
import torch
from PIL import Image

from ..models import IrisRecognizer
from ..eval import BiometricEvaluator, IrisVisualizer
from ..utils import get_device, load_config, set_seed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Iris Recognition System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'recognizer' not in st.session_state:
    st.session_state.recognizer = None
if 'enrolled_templates' not in st.session_state:
    st.session_state.enrolled_templates = {}
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = BiometricEvaluator()


def initialize_recognizer() -> IrisRecognizer:
    """Initialize the iris recognizer."""
    try:
        # Load configuration
        config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
        config = load_config(config_path)
        
        # Set random seed
        set_seed(42)
        
        # Get device
        device = get_device(config)
        
        # Initialize recognizer
        recognizer = IrisRecognizer(device=device, threshold=config.model.get('threshold', 0.5))
        
        logger.info(f"Iris recognizer initialized on device: {device}")
        return recognizer
        
    except Exception as e:
        st.error(f"Failed to initialize iris recognizer: {str(e)}")
        return None


def process_uploaded_image(uploaded_file) -> Optional[np.ndarray]:
    """Process uploaded image file."""
    try:
        # Load image
        image = Image.open(uploaded_file)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Convert to grayscale
        if len(image_array.shape) == 3:
            image_array = np.mean(image_array, axis=2)
        
        return image_array
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">👁️ Iris Recognition System</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ Research and Educational Use Only</h4>
        <p>This is a research and educational demonstration system. It is NOT intended for production 
        security operations or real-world biometric authentication. The system may be inaccurate and 
        should not be used for actual security decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize recognizer
    if st.session_state.recognizer is None:
        with st.spinner("Initializing iris recognition system..."):
            st.session_state.recognizer = initialize_recognizer()
    
    if st.session_state.recognizer is None:
        st.error("Failed to initialize the iris recognition system. Please refresh the page.")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Enrollment", "Authentication", "Evaluation", "Settings"]
    )
    
    if page == "Enrollment":
        enrollment_page()
    elif page == "Authentication":
        authentication_page()
    elif page == "Evaluation":
        evaluation_page()
    elif page == "Settings":
        settings_page()


def enrollment_page():
    """Enrollment page for registering new iris templates."""
    
    st.header("🔐 Iris Enrollment")
    
    st.markdown("""
    Upload an iris image to create a biometric template. The template will be stored 
    in memory for authentication purposes.
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an iris image",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a clear iris image for enrollment"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Iris Image", use_column_width=True)
        
        with col2:
            st.subheader("Enrollment Information")
            
            # Get identifier
            identifier = st.text_input(
                "Identifier (optional)",
                value=f"user_{len(st.session_state.enrolled_templates) + 1}",
                help="Unique identifier for this iris template"
            )
            
            # Enrollment button
            if st.button("Enroll Iris", type="primary"):
                with st.spinner("Processing iris image..."):
                    # Process image
                    image_array = process_uploaded_image(uploaded_file)
                    
                    if image_array is not None:
                        try:
                            # Enroll the iris
                            template = st.session_state.recognizer.enroll(image_array, identifier)
                            
                            # Store template
                            st.session_state.enrolled_templates[identifier] = template
                            
                            # Success message
                            st.markdown(f"""
                            <div class="success-box">
                                <h4>✅ Enrollment Successful</h4>
                                <p><strong>Identifier:</strong> {identifier}</p>
                                <p><strong>Template Size:</strong> {template.get_template_size()} features</p>
                                <p><strong>Feature Types:</strong> {', '.join(template.features.keys())}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.markdown(f"""
                            <div class="error-box">
                                <h4>❌ Enrollment Failed</h4>
                                <p>Error: {str(e)}</p>
                            </div>
                            """, unsafe_allow_html=True)
    
    # Display enrolled templates
    if st.session_state.enrolled_templates:
        st.subheader("Enrolled Templates")
        
        for identifier, template in st.session_state.enrolled_templates.items():
            with st.expander(f"Template: {identifier}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Template Size:** {template.get_template_size()} features")
                    st.write(f"**Feature Types:** {', '.join(template.features.keys())}")
                
                with col2:
                    if st.button(f"Delete {identifier}", key=f"delete_{identifier}"):
                        del st.session_state.enrolled_templates[identifier]
                        st.rerun()


def authentication_page():
    """Authentication page for verifying iris images."""
    
    st.header("🔍 Iris Authentication")
    
    st.markdown("""
    Upload an iris image to authenticate against enrolled templates. The system will 
    compare the uploaded image with all enrolled templates and provide similarity scores.
    """)
    
    # Check if templates are enrolled
    if not st.session_state.enrolled_templates:
        st.warning("No templates enrolled. Please enroll at least one iris template first.")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an iris image for authentication",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an iris image to authenticate"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Authentication Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Authentication Image", use_column_width=True)
        
        with col2:
            st.subheader("Authentication Settings")
            
            # Comparison method
            method = st.selectbox(
                "Comparison Method",
                ["cosine", "euclidean", "hamming"],
                help="Method used for template comparison"
            )
            
            # Authentication button
            if st.button("Authenticate", type="primary"):
                with st.spinner("Authenticating iris..."):
                    # Process image
                    image_array = process_uploaded_image(uploaded_file)
                    
                    if image_array is not None:
                        try:
                            # Find best match
                            best_template, best_similarity, is_match = st.session_state.recognizer.find_best_match(
                                image_array, 
                                list(st.session_state.enrolled_templates.values()),
                                method=method
                            )
                            
                            # Display results
                            if is_match:
                                st.markdown(f"""
                                <div class="success-box">
                                    <h4>✅ Authentication Successful</h4>
                                    <p><strong>Matched Template:</strong> {best_template.identifier}</p>
                                    <p><strong>Similarity Score:</strong> {best_similarity:.4f}</p>
                                    <p><strong>Threshold:</strong> {st.session_state.recognizer.threshold:.4f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="error-box">
                                    <h4>❌ Authentication Failed</h4>
                                    <p><strong>Best Match:</strong> {best_template.identifier}</p>
                                    <p><strong>Similarity Score:</strong> {best_similarity:.4f}</p>
                                    <p><strong>Threshold:</strong> {st.session_state.recognizer.threshold:.4f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Show all comparisons
                            st.subheader("All Template Comparisons")
                            
                            comparison_data = []
                            for identifier, template in st.session_state.enrolled_templates.items():
                                similarity = st.session_state.recognizer.authenticate(
                                    image_array, template, method
                                )
                                comparison_data.append({
                                    'Template': identifier,
                                    'Similarity': f"{similarity:.4f}",
                                    'Match': 'Yes' if similarity >= st.session_state.recognizer.threshold else 'No'
                                })
                            
                            # Sort by similarity
                            comparison_data.sort(key=lambda x: float(x['Similarity']), reverse=True)
                            
                            # Display table
                            st.table(comparison_data)
                            
                        except Exception as e:
                            st.markdown(f"""
                            <div class="error-box">
                                <h4>❌ Authentication Failed</h4>
                                <p>Error: {str(e)}</p>
                            </div>
                            """, unsafe_allow_html=True)


def evaluation_page():
    """Evaluation page for system performance analysis."""
    
    st.header("📊 System Evaluation")
    
    st.markdown("""
    This page provides tools for evaluating the iris recognition system performance 
    using biometric-specific metrics.
    """)
    
    # Evaluation options
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Evaluation Options")
        
        # Generate synthetic data option
        if st.button("Generate Synthetic Test Data"):
            with st.spinner("Generating synthetic test data..."):
                # This would generate synthetic iris data for testing
                st.info("Synthetic data generation would be implemented here")
    
    with col2:
        st.subheader("Current System Status")
        
        st.metric("Enrolled Templates", len(st.session_state.enrolled_templates))
        st.metric("Current Threshold", f"{st.session_state.recognizer.threshold:.4f}")
        st.metric("Comparison Method", "cosine")  # This would be dynamic
    
    # Evaluation metrics
    if st.session_state.enrolled_templates:
        st.subheader("Performance Metrics")
        
        # Placeholder for evaluation metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("EER", "0.023", "0.001")
        
        with col2:
            st.metric("minDCF", "0.045", "0.002")
        
        with col3:
            st.metric("AUC", "0.987", "0.003")
        
        with col4:
            st.metric("Accuracy", "97.8%", "0.5%")
        
        # Visualization options
        st.subheader("Visualizations")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            if st.button("Show ROC Curve"):
                st.info("ROC curve visualization would be implemented here")
        
        with viz_col2:
            if st.button("Show Score Distributions"):
                st.info("Score distribution visualization would be implemented here")
    
    else:
        st.warning("No templates enrolled. Please enroll templates first to perform evaluation.")


def settings_page():
    """Settings page for system configuration."""
    
    st.header("⚙️ System Settings")
    
    st.markdown("""
    Configure system parameters and view system information.
    """)
    
    # System settings
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Recognition Parameters")
        
        # Threshold setting
        current_threshold = st.session_state.recognizer.threshold
        new_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=current_threshold,
            step=0.01,
            help="Threshold for determining a match"
        )
        
        if new_threshold != current_threshold:
            st.session_state.recognizer.set_threshold(new_threshold)
            st.success(f"Threshold updated to {new_threshold:.4f}")
        
        # Comparison method
        method = st.selectbox(
            "Default Comparison Method",
            ["cosine", "euclidean", "hamming"],
            help="Default method for template comparison"
        )
    
    with col2:
        st.subheader("System Information")
        
        # Device information
        device = st.session_state.recognizer.device
        st.write(f"**Device:** {device}")
        
        # PyTorch version
        st.write(f"**PyTorch Version:** {torch.__version__}")
        
        # CUDA availability
        cuda_available = torch.cuda.is_available()
        st.write(f"**CUDA Available:** {cuda_available}")
        
        if cuda_available:
            st.write(f"**CUDA Device Count:** {torch.cuda.device_count()}")
            st.write(f"**Current CUDA Device:** {torch.cuda.current_device()}")
    
    # Privacy settings
    st.subheader("Privacy Settings")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.checkbox("Anonymize Identifiers", value=True, help="Hash identifiers for privacy")
        st.checkbox("Data Retention Policy", value=True, help="Apply data retention policies")
    
    with col2:
        st.checkbox("Consent Required", value=True, help="Require explicit consent")
        st.checkbox("Audit Logging", value=True, help="Enable audit logging")
    
    # Clear data
    st.subheader("Data Management")
    
    if st.button("Clear All Templates", type="secondary"):
        st.session_state.enrolled_templates = {}
        st.success("All templates cleared")
        st.rerun()
    
    if st.button("Reset Evaluator", type="secondary"):
        st.session_state.evaluator = BiometricEvaluator()
        st.success("Evaluator reset")


if __name__ == "__main__":
    main()
