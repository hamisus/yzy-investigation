#!/usr/bin/env python3
"""
Script to extract data from images using XOR with specific keywords.

This standalone script extracts potentially hidden data from images
using XOR operations with specific keywords or numbers.
"""

import argparse
import logging
import time
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import json
import base64

from PIL import Image
import numpy as np

from yzy_investigation.projects.image_cracking import (
    StegStrategy,
    KeywordXorStrategy,
)
from yzy_investigation.core.log_manager import setup_logging

def find_images(input_dir: Path) -> List[Path]:
    """
    Find all image files in the input directory.
    
    Args:
        input_dir: Directory to search for images
        
    Returns:
        List of paths to image files
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
    return [
        f for f in input_dir.glob("**/*") 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

def extract_xor_data(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    custom_keywords: Optional[List[str]] = None,
    custom_numbers: Optional[List[int]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Extract hidden data from images using XOR operations.
    
    Args:
        input_dir: Directory containing images to analyze
        output_dir: Directory to store results
        custom_keywords: Additional keywords to use for XOR operations
        custom_numbers: Additional numbers to use for XOR operations
        verbose: Whether to enable verbose logging
        
    Returns:
        Dictionary with extraction results
    """
    # Set up logging
    output_dir = output_dir or Path("results/xor_extraction")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped subfolder for this run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = run_dir / "logs"
    logger = setup_logging(
        "xor_extraction",
        log_level=logging.DEBUG if verbose else logging.INFO,
        log_dir=log_dir
    )
    
    logger.info(f"Starting XOR data extraction from {input_dir}")
    start_time = time.time()
    
    # Find all images
    images = find_images(input_dir)
    logger.info(f"Found {len(images)} images to analyze")
    
    # Set up the strategy
    strategy = KeywordXorStrategy()
    
    # Add custom keywords if provided
    if custom_keywords:
        strategy.KEY_TERMS.extend(custom_keywords)
        logger.info(f"Added custom keywords: {', '.join(custom_keywords)}")
    
    # Add custom numbers if provided
    if custom_numbers:
        strategy.KEY_NUMBERS.extend(custom_numbers)
        logger.info(f"Added custom numbers: {', '.join(map(str, custom_numbers))}")
    
    # Keep track of results
    results = {
        "images_analyzed": len(images),
        "images_with_hidden_data": 0,
        "target_string_found": False,
        "keys_found": [],
        "extraction_time": 0,
        "timestamp": timestamp,
        "run_dir": str(run_dir)
    }
    
    # Create directory for decoded content
    decoded_dir = run_dir / "decoded_content"
    decoded_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    for i, image_path in enumerate(images):
        logger.info(f"Analyzing image {i+1}/{len(images)}: {image_path.name}")
        
        try:
            # Update the strategy to use our timestamped decoded_dir
            strategy._output_dir = decoded_dir

            # Analyze the image
            detected, data = strategy.analyze(image_path)
            
            if detected:
                results["images_with_hidden_data"] += 1
                logger.info(f"Hidden data detected in {image_path.name}")
                
                # Check if target string was found
                for key, value in data.items():
                    if "found_key" in key:
                        results["target_string_found"] = True
                        key_used = key.replace("_found_key", "").replace("xor_", "")
                        results["keys_found"].append(key_used)
                        logger.warning(f"Target string found in {image_path.name} using key: {key_used}")
                    elif "false_positive" in key:
                        # Log false positives but don't count them as successful detections
                        reason = value.get("reason", "Unknown reason")
                        logger.info(f"False positive in {image_path.name}: {reason}")
                        
                        # If this was our only "detection", decrement the count
                        if "images_with_hidden_data" in results and len(data) == 1:
                            results["images_with_hidden_data"] = max(0, results["images_with_hidden_data"] - 1)
            else:
                logger.info(f"No hidden data detected in {image_path.name}")
                
        except Exception as e:
            logger.error(f"Error analyzing {image_path.name}: {e}")
            if verbose:
                import traceback
                logger.debug(traceback.format_exc())
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    results["extraction_time"] = elapsed_time
    
    logger.info(f"XOR extraction completed in {elapsed_time:.2f} seconds")
    logger.info(f"Analyzed {results['images_analyzed']} images")
    logger.info(f"Found hidden data in {results['images_with_hidden_data']} images")
    
    if results["target_string_found"]:
        logger.warning(f"TARGET STRING FOUND using keys: {', '.join(results['keys_found'])}")
        logger.warning(f"Check the '{decoded_dir}' directory for the decoded content")
    else:
        logger.info("Target string not found in any image")
    
    # Save summary
    summary_path = run_dir / "extraction_summary.txt"
    with open(summary_path, "w") as f:
        f.write("XOR DATA EXTRACTION SUMMARY\n")
        f.write("==========================\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Images analyzed: {results['images_analyzed']}\n")
        f.write(f"Images with hidden data: {results['images_with_hidden_data']}\n")
        f.write(f"Target string found: {results['target_string_found']}\n")
        if results["target_string_found"]:
            f.write(f"Keys that revealed target string: {', '.join(results['keys_found'])}\n")
        f.write(f"Total extraction time: {elapsed_time:.2f} seconds\n\n")
        f.write("Note on false positives: When using the key '4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump' for XOR,\n")
        f.write("if the same string appears in the output, it likely indicates that portion of the image contained zeros,\n")
        f.write("not necessarily hidden data (since X ⊕ 0 = X).\n\n")
        f.write(f"Decoded content is available in the '{decoded_dir}' directory\n")
    
    logger.info(f"Summary saved to {summary_path}")
    
    # Also save a copy of the summary to the main output directory for quick reference
    main_summary_path = output_dir / f"latest_run_summary_{timestamp}.txt"
    with open(main_summary_path, "w") as f:
        f.write(f"Latest run: {timestamp}\n\n")
        f.write(f"Images analyzed: {results['images_analyzed']}\n")
        f.write(f"Images with hidden data: {results['images_with_hidden_data']}\n")
        f.write(f"Target string found: {results['target_string_found']}\n")
        if results["target_string_found"]:
            f.write(f"Keys that revealed target string: {', '.join(results['keys_found'])}\n")
        f.write(f"Full results in: {run_dir}\n")
    
    return results

def main() -> None:
    """Main entry point for the XOR extraction script."""
    parser = argparse.ArgumentParser(
        description="Extract hidden data from images using XOR operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract hidden data from images:
  python -m yzy_investigation.projects.puzzle_cracking.scripts.extract_xor_data --input-dir ./data/raw/yews
  
  # Use custom keywords and numbers:
  python -m yzy_investigation.projects.puzzle_cracking.scripts.extract_xor_data --input-dir ./data/raw/yews --keywords Blake Tyger --numbers 4 333
"""
    )
    
    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        required=True,
        help="Directory containing images to analyze"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Directory to store results (default: results/xor_extraction)"
    )
    
    parser.add_argument(
        "--keywords", "-k",
        nargs="+",
        help="Additional keywords to use for XOR operations"
    )
    
    parser.add_argument(
        "--numbers", "-n",
        type=int,
        nargs="+",
        help="Additional numbers to use for XOR operations"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    try:
        results = extract_xor_data(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            custom_keywords=args.keywords,
            custom_numbers=args.numbers,
            verbose=args.verbose
        )
        
        print("\nXOR EXTRACTION COMPLETE")
        print("======================")
        print(f"Results saved to: {results['run_dir']}")
        print(f"Images analyzed: {results['images_analyzed']}")
        print(f"Images with hidden data: {results['images_with_hidden_data']}")
        if results['target_string_found']:
            print(f"Target string found using keys: {', '.join(results['keys_found'])}")
            print(f"Check the decoded content directory for details")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 