#!/usr/bin/env python3
"""
Script to explore the significance of the number 42 in the steganography analysis.

This script explores the special meaning of 42 in the context of our steganography analysis:
1. Checking which specific 42 images have the pattern
2. Looking for patterns in the indices of those images
3. Interpreting '42' as ASCII (it's '*' in ASCII)
4. Checking if the pattern can be interpreted as stars/asterisks
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Tuple


def load_pattern_file(pattern_file: Path) -> dict:
    """
    Load a pattern file created by extract_yews_pattern.py
    
    Args:
        pattern_file: Path to the pattern JSON file
        
    Returns:
        Dictionary with the loaded pattern data
    """
    with open(pattern_file, "r") as f:
        return json.load(f)


def analyze_pattern_indices(data: Dict[str, Any]) -> Dict[str, List[int]]:
    """
    Analyze which indices have 1s and which have 0s in the pattern.
    
    Args:
        data: Pattern data dictionary
        
    Returns:
        Dictionary with indices of 1s and 0s
    """
    pattern = data["binary_pattern"]
    
    # Get indices of 1s and 0s
    indices_1 = [i for i, bit in enumerate(pattern) if bit == 1]
    indices_0 = [i for i, bit in enumerate(pattern) if bit == 0]
    
    # Count by time slot
    time_slots = {}
    for slot, slot_data in data["by_time_slot"].items():
        pattern = slot_data["pattern"]
        time_slots[slot] = {
            "indices_1": [i for i, bit in enumerate(pattern) if bit == 1],
            "indices_0": [i for i, bit in enumerate(pattern) if bit == 0],
            "count_1": sum(pattern),
            "count_0": len(pattern) - sum(pattern)
        }
    
    return {
        "indices_1": indices_1,
        "indices_0": indices_0,
        "count_1": len(indices_1),
        "count_0": len(indices_0),
        "time_slots": time_slots
    }


def try_42_as_ascii() -> None:
    """
    Interpret the number 42 as ASCII and explore its significance.
    """
    print("\n=== Interpreting 42 as ASCII ===")
    
    ascii_42 = chr(42)
    print(f"ASCII value of 42: '{ascii_42}' (asterisk/star)")
    
    # Mention Douglas Adams reference
    print("In 'The Hitchhiker's Guide to the Galaxy', 42 is the 'Answer to the Ultimate Question of Life, the Universe, and Everything'")
    
    # Check ASCII values of related characters
    print(f"ASCII value of '*': {ord('*')} (42)")
    print(f"ASCII value of 'B': {ord('B')} (66, which is 42 in hexadecimal)")
    
    # Check if the count of 1s in our pattern has any significance
    print("\nIf the 42 out of 60 images with the pattern correspond to '*' symbols:")
    print("This could suggest looking for a pattern of 42 asterisks in a 60-character message")


def visualize_pattern_as_stars(pattern: List[int], output_dir: Path) -> None:
    """
    Visualize the pattern as asterisks/stars.
    
    Args:
        pattern: List of 0s and 1s
        output_dir: Output directory for visualizations
    """
    length = len(pattern)
    
    # Create visualizations with different shapes where 1s are stars
    arrangements = [
        (3, 20),  # 3 rows x 20 columns
        (4, 15),  # 4 rows x 15 columns
        (5, 12),  # 5 rows x 12 columns
        (6, 10),  # 6 rows x 10 columns
        (10, 6),  # 10 rows x 6 columns
        (12, 5),  # 12 rows x 5 columns
        (15, 4),  # 15 rows x 4 columns
        (20, 3),  # 20 rows x 3 columns
    ]
    
    for rows, cols in arrangements:
        if rows * cols != length:
            continue
            
        # Create matrix
        matrix = np.array(pattern).reshape(rows, cols)
        
        # Create figure
        plt.figure(figsize=(max(8, cols), max(4, rows)))
        
        # Plot empty grid
        plt.xlim(-0.5, cols - 0.5)
        plt.ylim(-0.5, rows - 0.5)
        plt.grid(True, color='lightgray')
        
        # Add stars for 1s
        for i in range(rows):
            for j in range(cols):
                if matrix[i, j] == 1:
                    plt.text(j, i, "*", fontsize=20, ha='center', va='center', color='blue')
                else:
                    plt.text(j, i, "·", fontsize=20, ha='center', va='center', color='lightgray')
        
        plt.title(f"Pattern as Stars ({rows}x{cols})")
        plt.tight_layout()
        plt.savefig(output_dir / f"stars_{rows}x{cols}.png")
        plt.close()


def analyze_image_details(data: Dict[str, Any]) -> None:
    """
    Analyze the detailed results to see which specific images have the pattern.
    
    Args:
        data: Pattern data dictionary
    """
    print("\n=== Analyzing Images with Pattern ===")
    
    results = data.get("detailed_results", [])
    
    # Group by pattern presence and time slot
    with_pattern = []
    without_pattern = []
    
    for result in sorted(results, key=lambda x: x["directory"]):
        if result["has_pattern"]:
            with_pattern.append(result["directory"])
        else:
            without_pattern.append(result["directory"])
    
    # Print summary
    print(f"Images with pattern ({len(with_pattern)}):")
    for i, name in enumerate(with_pattern):
        print(f"  {name}")
        if i >= 15:  # Limit output if too many
            print(f"  ... and {len(with_pattern) - 16} more")
            break
    
    print(f"\nImages without pattern ({len(without_pattern)}):")
    for i, name in enumerate(without_pattern):
        print(f"  {name}")
        if i >= 15:  # Limit output if too many
            print(f"  ... and {len(without_pattern) - 16} more")
            break
    
    # Check for title patterns
    print("\nAnalyzing image titles:")
    
    # Extract titles (skip the time slot part)
    with_pattern_titles = [name.split("_", 2)[2] if "_" in name else name for name in with_pattern]
    without_pattern_titles = [name.split("_", 2)[2] if "_" in name else name for name in without_pattern]
    
    # Look for common words or patterns
    common_words_with = {}
    for title in with_pattern_titles:
        for word in title.split("_"):
            if word.lower() in ["untitled"]:
                continue
            if word.lower() not in common_words_with:
                common_words_with[word.lower()] = 0
            common_words_with[word.lower()] += 1
    
    common_words_without = {}
    for title in without_pattern_titles:
        for word in title.split("_"):
            if word.lower() in ["untitled"]:
                continue
            if word.lower() not in common_words_without:
                common_words_without[word.lower()] = 0
            common_words_without[word.lower()] += 1
    
    # Print most common words
    print("\nMost common words in titles WITH pattern:")
    for word, count in sorted(common_words_with.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {word}: {count}")
    
    print("\nMost common words in titles WITHOUT pattern:")
    for word, count in sorted(common_words_without.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {word}: {count}")


def check_for_constellations(pattern: List[int], output_dir: Path) -> None:
    """
    Check if the pattern resembles known star constellations.
    
    Args:
        pattern: List of 0s and 1s
        output_dir: Output directory for visualizations
    """
    print("\n=== Checking for Constellations ===")
    
    # Most promising arrangements for recognizing constellations
    arrangements = [
        (6, 10),  # 6 rows x 10 columns
        (10, 6),  # 10 rows x 6 columns
        (5, 12),  # 5 rows x 12 columns
        (12, 5),  # 12 rows x 5 columns
    ]
    
    for rows, cols in arrangements:
        if rows * cols != len(pattern):
            continue
            
        # Create matrix
        matrix = np.array(pattern).reshape(rows, cols)
        
        # Plot as stars
        plt.figure(figsize=(max(8, cols), max(6, rows)))
        
        # Black background
        plt.gca().set_facecolor('black')
        
        # No axes or grid
        plt.axis('off')
        
        # Plot 1s as stars
        for i in range(rows):
            for j in range(cols):
                if matrix[i, j] == 1:
                    plt.plot(j, i, 'o', color='white', markersize=10)
        
        plt.title(f"Possible Constellation ({rows}x{cols})", color='white')
        plt.tight_layout()
        plt.savefig(output_dir / f"constellation_{rows}x{cols}.png")
        plt.close()
    
    print("Generated constellation visualizations")


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Explore the significance of 42 in steganography analysis')
    parser.add_argument('--pattern-file', type=str, default='results/stego_pattern/extracted_pattern.json',
                      help='Path to the pattern JSON file')
    parser.add_argument('--output-dir', type=str, default='results/stego_pattern/42_analysis',
                      help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Load pattern file
    pattern_file = Path(args.pattern_file)
    data = load_pattern_file(pattern_file)
    
    # Extract the binary pattern
    pattern = data["binary_pattern"]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Print basic pattern information
    print(f"Loaded pattern with {len(pattern)} bits")
    print(f"Pattern: {data['pattern_string']}")
    
    # Analyze pattern indices
    indices_analysis = analyze_pattern_indices(data)
    
    print(f"\nPattern has {indices_analysis['count_1']} bits set to 1 and {indices_analysis['count_0']} bits set to 0")
    
    for slot, slot_data in indices_analysis["time_slots"].items():
        print(f"  {slot}: {slot_data['count_1']} bits set to 1, {slot_data['count_0']} bits set to 0")
    
    # Try 42 as ASCII
    try_42_as_ascii()
    
    # Visualize pattern as stars
    visualize_pattern_as_stars(pattern, output_dir)
    
    # Analyze image details
    analyze_image_details(data)
    
    # Check for constellations
    check_for_constellations(pattern, output_dir)
    
    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    main() 