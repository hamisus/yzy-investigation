#!/usr/bin/env python3
"""
Example script showing how to use the stego_analysis package.
"""
import os
import sys
import json
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from yzy_investigation.projects.stego_analysis.src.analyzer import detect_steganography
from yzy_investigation.projects.stego_analysis.src.puzzle_integration import analyze_image_with_puzzle_keywords


def main():
    """
    Example usage of the stego analysis tools.
    """
    # Example image path - replace with your actual image path
    image_path = "path/to/your/image.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return 1
    
    print("Running basic steganography detection...")
    results = detect_steganography(image_path)
    print(json.dumps(results, indent=2))
    
    print("\nRunning analysis with puzzle keywords...")
    results_with_keywords = analyze_image_with_puzzle_keywords(image_path)
    print(json.dumps(results_with_keywords, indent=2))
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 