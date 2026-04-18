#!/usr/bin/env python3
"""Main entry point for the iris recognition system."""

import argparse
import logging
import sys
from pathlib import Path

from src.models import IrisRecognizer
from src.utils import get_device, load_config, set_seed, setup_logging


def main():
    """Main entry point for the iris recognition system."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Iris Recognition System")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["enroll", "authenticate", "demo"],
        default="demo",
        help="Operation mode"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to iris image"
    )
    parser.add_argument(
        "--template",
        type=str,
        help="Path to iris template"
    )
    parser.add_argument(
        "--identifier",
        type=str,
        help="Identifier for the iris template"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Similarity threshold for matching"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["cosine", "euclidean", "hamming"],
        default="cosine",
        help="Comparison method"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Setup logging
    if args.verbose:
        config.logging.level = "DEBUG"
    
    logger = setup_logging(config)
    
    # Set random seed
    set_seed(42)
    
    # Get device
    device = get_device(config)
    logger.info(f"Using device: {device}")
    
    # Initialize recognizer
    recognizer = IrisRecognizer(device=device, threshold=args.threshold)
    
    if args.mode == "enroll":
        enroll_mode(recognizer, args, logger)
    elif args.mode == "authenticate":
        authenticate_mode(recognizer, args, logger)
    elif args.mode == "demo":
        demo_mode(recognizer, args, logger)


def enroll_mode(recognizer, args, logger):
    """Enrollment mode - enroll a new iris template."""
    
    if not args.image:
        print("Error: --image is required for enrollment mode")
        sys.exit(1)
    
    if not args.identifier:
        print("Error: --identifier is required for enrollment mode")
        sys.exit(1)
    
    try:
        # Enroll the iris
        template = recognizer.enroll(args.image, args.identifier)
        
        # Save template if path provided
        if args.template:
            template.save(args.template)
            logger.info(f"Template saved to {args.template}")
        
        print(f"✅ Enrollment successful!")
        print(f"Identifier: {args.identifier}")
        print(f"Template size: {template.get_template_size()} features")
        print(f"Feature types: {', '.join(template.features.keys())}")
        
    except Exception as e:
        logger.error(f"Enrollment failed: {str(e)}")
        print(f"❌ Enrollment failed: {str(e)}")
        sys.exit(1)


def authenticate_mode(recognizer, args, logger):
    """Authentication mode - authenticate an iris against a template."""
    
    if not args.image:
        print("Error: --image is required for authentication mode")
        sys.exit(1)
    
    if not args.template:
        print("Error: --template is required for authentication mode")
        sys.exit(1)
    
    try:
        # Load template
        template = recognizer.enroll(args.template)  # Load template
        
        # Authenticate
        similarity = recognizer.authenticate(args.image, template, args.method)
        is_match = recognizer.is_match(similarity)
        
        print(f"🔍 Authentication Results:")
        print(f"Similarity Score: {similarity:.4f}")
        print(f"Threshold: {recognizer.threshold:.4f}")
        print(f"Match: {'✅ Yes' if is_match else '❌ No'}")
        
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}")
        print(f"❌ Authentication failed: {str(e)}")
        sys.exit(1)


def demo_mode(recognizer, args, logger):
    """Demo mode - launch the Streamlit application."""
    
    try:
        import streamlit.web.cli as stcli
        import sys
        
        # Launch Streamlit app
        demo_path = Path(__file__).parent / "demo" / "app.py"
        sys.argv = ["streamlit", "run", str(demo_path)]
        stcli.main()
        
    except ImportError:
        print("Error: Streamlit is required for demo mode")
        print("Install with: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Demo launch failed: {str(e)}")
        print(f"❌ Demo launch failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
