#!/usr/bin/env python3
"""
Script to extract the binary pattern from the 60 news images.

This script extracts a binary pattern from the images in the yews dataset
based on color histogram analysis. Each image is classified as 0 or 1
based on whether it contains a specific pattern in its color distribution.
"""

import os
import sys
import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to Python path if running script directly
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(project_root))

from yzy_investigation.projects.puzzle_cracking.stego_strategies import StegStrategy


# Custom JSON encoder to handle non-serializable types
class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle non-serializable types like numpy arrays and booleans."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return int(obj)
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def analyze_color_histogram(image_path: Path) -> Tuple[bool, Dict[str, Any]]:
    """
    Analyze the color histogram of an image to detect steganography.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple containing:
            - Boolean indicating if a pattern was detected
            - Dictionary with analysis metadata
    """
    try:
        # Open the image
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Check if image is color (RGB)
        if len(img_array.shape) < 3 or img_array.shape[2] < 3:
            return False, {"error": "Not a color image"}
        
        # Extract color channels
        red = img_array[:, :, 0].flatten()
        green = img_array[:, :, 1].flatten()
        blue = img_array[:, :, 2].flatten()
        
        # Calculate histograms
        r_hist, _ = np.histogram(red, bins=256, range=(0, 256))
        g_hist, _ = np.histogram(green, bins=256, range=(0, 256))
        b_hist, _ = np.histogram(blue, bins=256, range=(0, 256))
        
        # Calculate histogram statistics
        r_mean = np.mean(r_hist)
        g_mean = np.mean(g_hist)
        b_mean = np.mean(b_hist)
        
        r_std = np.std(r_hist)
        g_std = np.std(g_hist)
        b_std = np.std(b_hist)
        
        # Calculate peak locations
        r_peaks = np.where(r_hist > r_mean + 2*r_std)[0]
        g_peaks = np.where(g_hist > g_mean + 2*g_std)[0]
        b_peaks = np.where(b_hist > b_mean + 2*b_std)[0]
        
        # Check for the specific pattern:
        # 1. Number of peaks in the histograms
        # 2. Peak position distribution
        # 3. Relationship between color channels
        
        # Pattern detection logic - we're determining if there's an unusual 
        # pattern in the color distribution that might indicate steganography
        
        # For demonstration, we'll use a simple metric: the ratio of peaks
        # in the red channel compared to other channels
        peak_pattern_factor = len(r_peaks) / max(1, (len(g_peaks) + len(b_peaks)) / 2)
        
        # Another factor: distribution of colors in specific ranges
        dark_colors_r = np.sum(r_hist[:85]) / np.sum(r_hist)
        mid_colors_r = np.sum(r_hist[85:170]) / np.sum(r_hist)
        light_colors_r = np.sum(r_hist[170:]) / np.sum(r_hist)
        
        # Create a pattern score based on multiple factors
        pattern_score = peak_pattern_factor
        pattern_score += 2 * abs(dark_colors_r - 0.33)  # Deviation from uniform
        
        # Return results
        has_pattern = pattern_score > 1.5  # Threshold based on analysis of known images
        
        result = {
            "peak_pattern_factor": float(peak_pattern_factor),
            "dark_colors_ratio": float(dark_colors_r),
            "mid_colors_ratio": float(mid_colors_r),
            "light_colors_ratio": float(light_colors_r),
            "pattern_score": float(pattern_score),
            "r_peaks": int(len(r_peaks)),
            "g_peaks": int(len(g_peaks)),
            "b_peaks": int(len(b_peaks))
        }
        
        return has_pattern, result
        
    except Exception as e:
        return False, {"error": str(e)}


def extract_binary_pattern(data_dir: Path, output_file: Path) -> None:
    """
    Extract a binary pattern from all image folders in the data directory.
    
    Args:
        data_dir: Directory containing the yews date folders
        output_file: File to save the extracted pattern
    """
    # First, collect all subdirectories with images
    image_dirs = []
    for time_slot in ["10AM", "3PM", "8PM"]:
        dirs = [d for d in data_dir.glob(f"*_{time_slot}_*") if d.is_dir()]
        for d in dirs:
            image_path = d / "image_01.jpg"
            if image_path.exists():
                image_dirs.append((d, image_path))
    
    print(f"Found {len(image_dirs)} image directories")
    
    # Sort directories by their index
    def get_dir_index(dir_path: Path) -> int:
        name = dir_path.name
        if "_" not in name:
            return 999  # Sort unknown formats to the end
        try:
            index = int(name.split("_")[0])
            return index
        except:
            return 999
    
    image_dirs.sort(key=lambda x: (x[0].name.split("_")[1], get_dir_index(x[0])))
    
    # Analyze each image
    results = []
    for dir_path, image_path in image_dirs:
        dir_name = dir_path.name
        print(f"Analyzing {dir_name}...")
        
        has_pattern, analysis = analyze_color_histogram(image_path)
        
        # Add to results (ensure all values are JSON serializable)
        results.append({
            "directory": str(dir_name),
            "image_path": str(image_path),
            "has_pattern": int(has_pattern),  # Convert bool to int
            "analysis": analysis
        })
    
    # Extract the binary pattern
    binary_pattern = [int(r["has_pattern"]) for r in results]  # Use int not bool
    pattern_string = ''.join(str(b) for b in binary_pattern)
    
    # Group images by time slot to analyze patterns
    time_slots = {}
    for i, r in enumerate(results):
        dir_name = r["directory"]
        if "_10AM_" in dir_name:
            slot = "10AM"
        elif "_3PM_" in dir_name:
            slot = "3PM"
        elif "_8PM_" in dir_name:
            slot = "8PM"
        else:
            slot = "UNKNOWN"
            
        if slot not in time_slots:
            time_slots[slot] = []
        
        time_slots[slot].append((i, int(r["has_pattern"])))  # Ensure int, not bool
    
    # Create a structured pattern by time slot
    structured_patterns = {}
    for slot, items in time_slots.items():
        items.sort()  # Sort by index
        structured_patterns[slot] = [int(b) for _, b in items]  # Use int, not bool
    
    # Prepare the data for JSON serialization
    output_data = {
        "binary_pattern": binary_pattern,
        "pattern_string": pattern_string,
        "by_time_slot": {
            slot: {
                "pattern": pattern,
                "pattern_string": ''.join(str(b) for b in pattern)
            }
            for slot, pattern in structured_patterns.items()
        },
        "detailed_results": results
    }
    
    # Save results to file
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, cls=CustomJSONEncoder)
    
    print(f"Analysis complete. Results saved to {output_file}")
    
    # Print binary pattern
    print("\nExtracted Binary Pattern:")
    print(pattern_string)
    
    # Try to decode as ASCII
    if len(binary_pattern) >= 8:
        ascii_chars = []
        for i in range(0, len(binary_pattern) - 7, 8):
            byte = 0
            for j in range(8):
                if i + j < len(binary_pattern):
                    byte |= binary_pattern[i + j] << j
            ascii_chars.append(chr(byte) if 32 <= byte <= 126 else '?')
        
        ascii_text = ''.join(ascii_chars)
        print(f"\nPattern as ASCII text: {ascii_text}")


def decode_binary_pattern(pattern: List[int]) -> None:
    """
    Attempt to decode a binary pattern in various ways.
    
    Args:
        pattern: List of 0s and 1s representing the binary pattern
    """
    # Ensure all pattern values are integers
    pattern = [int(b) for b in pattern]
    
    pattern_str = ''.join(str(b) for b in pattern)
    print(f"Binary pattern ({len(pattern)} bits): {pattern_str}")
    
    # 1. Try as ASCII (8 bits per character)
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
    
    # 2. Try as ASCII with 7 bits per character
    if len(pattern) >= 7:
        ascii_7bit_chars = []
        for i in range(0, len(pattern) - 6, 7):
            byte = 0
            for j in range(7):
                if i + j < len(pattern):
                    byte |= pattern[i + j] << j
            ascii_7bit_chars.append(chr(byte) if 32 <= byte <= 126 else '?')
        
        ascii_7bit_text = ''.join(ascii_7bit_chars)
        print(f"\nAs 7-bit ASCII text: {ascii_7bit_text}")
    
    # 3. Try with bit reversal (LSB/MSB swap)
    if len(pattern) >= 8:
        reversed_chars = []
        for i in range(0, len(pattern) - 7, 8):
            byte = 0
            for j in range(8):
                if i + j < len(pattern):
                    byte |= pattern[i + j] << (7 - j)
            reversed_chars.append(chr(byte) if 32 <= byte <= 126 else '?')
        
        reversed_text = ''.join(reversed_chars)
        print(f"\nWith bit reversal: {reversed_text}")
    
    # 4. Try as binary pattern representing coordinates or other non-text data
    # For example, converting chunks to decimal
    chunks = [pattern[i:i+8] for i in range(0, len(pattern), 8)]
    decimal_values = []
    for chunk in chunks:
        if len(chunk) == 8:
            value = sum(bit << i for i, bit in enumerate(chunk))
            decimal_values.append(value)
    
    print(f"\nAs decimal values: {decimal_values}")


def analyze_pattern_file(pattern_file: Path) -> None:
    """
    Analyze a previously saved pattern file.
    
    Args:
        pattern_file: Path to the pattern JSON file
    """
    with open(pattern_file, "r") as f:
        data = json.load(f)
    
    binary_pattern = data["binary_pattern"]
    pattern_string = data["pattern_string"]
    
    print(f"Loaded pattern with {len(binary_pattern)} bits")
    
    # Analyze the pattern
    decode_binary_pattern(binary_pattern)
    
    # Analyze by time slot
    for slot, slot_data in data["by_time_slot"].items():
        print(f"\n=== Analysis for {slot} ===")
        slot_pattern = slot_data["pattern"]
        decode_binary_pattern(slot_pattern)


def main() -> None:
    """Main entry point for the pattern extraction script."""
    parser = argparse.ArgumentParser(description='Extract binary pattern from news images')
    parser.add_argument('--data-dir', type=str, default='data/raw/yews/2025-03-27',
                      help='Directory containing the date folder with news images')
    parser.add_argument('--output-file', type=str, default='results/stego_pattern/extracted_pattern.json',
                      help='File to save the extracted pattern')
    parser.add_argument('--analyze-only', type=str, default=None,
                      help='Analyze an existing pattern file instead of extracting a new one')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        analyze_pattern_file(Path(args.analyze_only))
    else:
        # Create output directory if it doesn't exist
        output_path = Path(args.output_file)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Extract the pattern
        extract_binary_pattern(Path(args.data_dir), output_path)


if __name__ == "__main__":
    main() 