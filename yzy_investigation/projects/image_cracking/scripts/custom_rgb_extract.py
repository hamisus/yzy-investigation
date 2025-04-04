#!/usr/bin/env python3
"""
Script to extract data using custom RGB encoding.

This script extracts data hidden in images using a custom RGB bit allocation scheme,
where data might be stored with a specific bit pattern across color channels.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import base64

from PIL import Image
import numpy as np

from yzy_investigation.projects.image_cracking import (
    StegStrategy,
    CustomRgbEncodingStrategy,
)
from yzy_investigation.core.log_manager import setup_logging


def find_images(input_path: Path) -> List[Path]:
    """
    Find all image files in a directory or return a single image.
    
    Args:
        input_path: Path to image file or directory containing images
        
    Returns:
        List of image file paths
    """
    image_files = []
    
    # If path is a file, just return it
    if input_path.is_file():
        if is_image_file(input_path):
            return [input_path]
        else:
            print(f"Warning: {input_path} does not appear to be an image file.")
            return []
    
    # If path is a directory, find all image files
    for root, _, files in os.walk(input_path):
        for file in files:
            file_path = Path(root) / file
            if is_image_file(file_path):
                image_files.append(file_path)
    
    return image_files


def is_image_file(file_path: Path) -> bool:
    """
    Check if a file is an image based on its extension.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if file appears to be an image, False otherwise
    """
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']
    return file_path.suffix.lower() in image_extensions


def analyze_image(image_path: Path, output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    """
    Analyze a single image with the CustomRgbEncodingStrategy.
    
    Args:
        image_path: Path to image file
        output_dir: Directory to save results
        logger: Logger instance
        
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Analyzing {image_path}")
    print(f"Analyzing {image_path}...")
    
    # Create the strategy
    strategy = CustomRgbEncodingStrategy()
    
    # Set output directory for the strategy
    strategy.set_output_dir(output_dir)
    
    # Create a result container
    result = StegAnalysisResult(image_path)
    
    try:
        # Run the strategy
        detected, data = strategy.analyze(image_path)
        
        # Add results
        result.add_strategy_result(strategy.name, detected, data)
        
        # Save result to JSON file
        results_dir = output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = results_dir / f"{image_path.stem}_custom_rgb_result.json"
        with open(result_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
            
        # Print success/failure message
        if detected:
            print(f"✅ Found hidden data in {image_path.name}")
            
            # Check if file info was found
            if data and "file_info" in data and data["file_info"]:
                print(f"   - Extracted file: {data['file_info']['filename']} ({data['file_info']['filetype']})")
                
            # Check if there's a path to the extracted file
            if data and "extracted_file_path" in data:
                print(f"   - Saved to: {data['extracted_file_path']}")
        else:
            print(f"❌ No custom RGB encoded data found in {image_path.name}")
            
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Error analyzing {image_path}: {e}")
        print(f"Error analyzing {image_path}: {e}")
        return {"error": str(e), "image_path": str(image_path)}


def main() -> None:
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(description="Extract data from images using the Custom RGB 3-3-2 bit encoding strategy")
    parser.add_argument("input", help="Path to image file or directory of images")
    parser.add_argument("-o", "--output", help="Output directory for results", default="results/custom_rgb_extraction")
    parser.add_argument("-v", "--verbose", help="Enable verbose logging", action="store_true")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging("custom_rgb_extract", log_level=log_level, log_dir=output_dir / "logs")
    
    # Find images to process
    input_path = Path(args.input)
    image_files = find_images(input_path)
    
    if not image_files:
        print(f"No image files found in {input_path}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} image(s) to analyze")
    
    # Process each image
    results = []
    for image_path in image_files:
        result = analyze_image(image_path, output_dir, logger)
        results.append(result)
    
    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "total_images": len(image_files),
            "results": results
        }, f, indent=2)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")
    
    # Count successful extractions
    successful = sum(1 for result in results if result.get("has_hidden_data", False))
    print(f"Found hidden data in {successful} out of {len(image_files)} images")


if __name__ == "__main__":
    main() 