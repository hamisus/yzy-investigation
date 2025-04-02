#!/usr/bin/env python3
"""
Script to explore and extract meaningful information from steganography analysis results.

This is a more interactive exploration tool compared to the automated processing
pipeline. It allows deeper investigation of steganography patterns.
"""

import json
import base64
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import argparse

# Add project root to Python path if running script directly
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(project_root))

from yzy_investigation.projects.puzzle_cracking.process_stego_results import StegoResultProcessor


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """
    Load and parse a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing the parsed JSON data
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def display_analysis_summary(summary_path: Path) -> None:
    """
    Display the steganography analysis summary.
    
    Args:
        summary_path: Path to the analysis summary JSON file
    """
    summary = load_json_file(summary_path)
    
    print("\n=== Steganography Analysis Summary ===")
    print(f"Total images analyzed: {summary['total_images']}")
    print(f"Images with hidden data: {summary['images_with_hidden_data']}")
    print("\nStrategy success counts:")
    for strategy, count in summary['strategy_success_counts'].items():
        print(f"  - {strategy}: {count}/{summary['total_images']}")
    
    # Calculate proportion of images with color histogram findings
    color_hist_ratio = summary['strategy_success_counts']['color_histogram_strategy'] / summary['total_images']
    print(f"\nColor histogram detection ratio: {color_hist_ratio:.2f}")
    
    if color_hist_ratio > 0.6 and color_hist_ratio < 0.8:
        print("NOTE: The 42/60 (70%) color histogram ratio might indicate binary data (0/1) across images")


def extract_pattern_from_results(results_dir: Path, pattern_type: str = "color_histogram") -> List[int]:
    """
    Extract a binary pattern from results.
    
    Args:
        results_dir: Directory containing analysis results
        pattern_type: Type of pattern to extract (default: color_histogram)
        
    Returns:
        List of 0s and 1s representing the pattern
    """
    pattern = []
    results_files = sorted([f for f in (results_dir / "results").glob("*.json")])
    
    for result_file in results_files:
        result_data = load_json_file(result_file)
        
        # Extract image number
        image_name = result_data["image_name"]
        image_num = int(image_name.split("_")[1].split(".")[0])
        
        # Check if pattern exists
        if pattern_type == "color_histogram":
            has_pattern = result_data["strategy_results"]["color_histogram_strategy"]["detected"]
            pattern.append((image_num, 1 if has_pattern else 0))
    
    # Sort by image number and extract just the pattern
    pattern.sort(key=lambda x: x[0])
    return [p[1] for p in pattern]


def analyze_binary_pattern(pattern: List[int]) -> None:
    """
    Analyze a binary pattern for possible meanings.
    
    Args:
        pattern: List of 0s and 1s
    """
    print("\n=== Binary Pattern Analysis ===")
    print(f"Pattern length: {len(pattern)}")
    
    # Print the pattern
    pattern_str = ''.join(str(b) for b in pattern)
    print(f"Raw pattern: {pattern_str}")
    
    # Try to interpret as ASCII (8 bits per character)
    if len(pattern) >= 8:
        ascii_chars = []
        for i in range(0, len(pattern) - 7, 8):
            byte = 0
            for j in range(8):
                if i + j < len(pattern):
                    byte |= pattern[i + j] << j
            ascii_chars.append(chr(byte) if 32 <= byte <= 126 else '?')
        
        ascii_text = ''.join(ascii_chars)
        print(f"\nAs ASCII text: {ascii_text}")
    
    # Try to interpret as bytes and look for file signatures
    bytes_data = bytearray()
    for i in range(0, len(pattern) - 7, 8):
        byte = 0
        for j in range(8):
            if i + j < len(pattern):
                byte |= pattern[i + j] << j
        bytes_data.append(byte)
    
    # Check for common file signatures
    signatures = {
        b"\x50\x4B\x03\x04": "ZIP",
        b"\x25\x50\x44\x46": "PDF",
        b"\xFF\xD8\xFF": "JPEG",
        b"\x89\x50\x4E\x47": "PNG"
    }
    
    for sig, file_type in signatures.items():
        if bytes_data.startswith(sig):
            print(f"\nDetected file signature: {file_type}")
            break


def extract_and_decode_data(extracted_data_dir: Path, output_dir: Path) -> None:
    """
    Extract and decode various data formats from the extracted data.
    
    Args:
        extracted_data_dir: Directory containing extracted data
        output_dir: Directory to save decoded files
    """
    output_dir.mkdir(exist_ok=True)
    
    # Process all image directories
    for img_dir in extracted_data_dir.iterdir():
        if not img_dir.is_dir():
            continue
            
        img_name = img_dir.name
        print(f"\nProcessing {img_name}...")
        
        # Handle LSB strategy JSON data
        lsb_json = img_dir / "lsb_strategy_data.json"
        if lsb_json.exists():
            lsb_data = load_json_file(lsb_json)
            if "encoding" in lsb_data and lsb_data["encoding"] == "base64":
                print(f"  Found base64-encoded LSB data")
                try:
                    bin_data = base64.b64decode(lsb_data.get("data", ""))
                    output_file = output_dir / f"{img_name}_lsb_decoded.bin"
                    with open(output_file, "wb") as f:
                        f.write(bin_data)
                    print(f"  Decoded LSB data saved to {output_file}")
                except Exception as e:
                    print(f"  Error decoding LSB data: {e}")
        
        # Handle file signature JSON data
        sig_json = img_dir / "file_signature_strategy_data.json"
        if sig_json.exists():
            sig_data = load_json_file(sig_json)
            if "encoding" in sig_data and sig_data["encoding"] == "base64":
                print(f"  Found base64-encoded file signature data")
                try:
                    bin_data = base64.b64decode(sig_data.get("data", ""))
                    output_file = output_dir / f"{img_name}_signature_decoded.bin"
                    with open(output_file, "wb") as f:
                        f.write(bin_data)
                    print(f"  Decoded signature data saved to {output_file}")
                    
                    # Determine file type if possible
                    file_type = sig_data.get("file_type")
                    if file_type:
                        ext = f".{file_type.lower()}"
                        typed_output = output_dir / f"{img_name}_signature_decoded{ext}"
                        with open(typed_output, "wb") as f:
                            f.write(bin_data)
                        print(f"  File identified as {file_type}, saved to {typed_output}")
                        
                except Exception as e:
                    print(f"  Error decoding signature data: {e}")


def main() -> None:
    """Main entry point for the steganography results explorer."""
    parser = argparse.ArgumentParser(description='Explore steganography analysis results')
    parser.add_argument('--results-dir', type=str, default='results/stego_analysis',
                      help='Directory containing the steganography analysis results')
    parser.add_argument('--output-dir', type=str, default='results/stego_explored',
                      help='Directory to store exploration output')
    parser.add_argument('--binary-pattern', action='store_true',
                      help='Extract and analyze binary pattern from color histogram results')
    parser.add_argument('--decode-data', action='store_true',
                      help='Decode base64 data from extracted JSON files')
    parser.add_argument('--run-processor', action='store_true',
                      help='Run the StegoResultProcessor on the results')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Display the analysis summary
    summary_path = results_dir / "analysis_summary.json"
    if summary_path.exists():
        display_analysis_summary(summary_path)
    
    # Extract binary pattern if requested
    if args.binary_pattern:
        pattern = extract_pattern_from_results(results_dir)
        if pattern:
            analyze_binary_pattern(pattern)
            
            # Save the pattern to a file
            pattern_file = output_dir / "binary_pattern.json"
            with open(pattern_file, "w") as f:
                json.dump({"pattern": pattern, "pattern_string": ''.join(str(b) for b in pattern)}, f, indent=2)
            print(f"\nBinary pattern saved to {pattern_file}")
    
    # Decode base64 data if requested
    if args.decode_data:
        extracted_data_dir = results_dir / "extracted_data"
        if extracted_data_dir.exists():
            decoded_dir = output_dir / "decoded_data"
            extract_and_decode_data(extracted_data_dir, decoded_dir)
    
    # Run the StegoResultProcessor if requested
    if args.run_processor:
        processor = StegoResultProcessor(results_dir, output_dir / "processed")
        results = processor.process_all_images()
        
        print("\n=== Stego Result Processor Results ===")
        print(f"Processed {results['total_images_processed']} images")
        print(f"Discovered {results['files_discovered']} potential files")
        if results['file_types_found']:
            print("\nFile types found:")
            for file_type, count in results['file_types_found'].items():
                print(f"  - {file_type}: {count}")


if __name__ == "__main__":
    main() 