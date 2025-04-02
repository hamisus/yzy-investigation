#!/usr/bin/env python3
"""
Script to visualize the binary pattern extracted from steganography analysis.

This script provides various visualizations of the binary pattern to help
identify hidden messages or patterns in the data.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


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


def visualize_as_image(pattern: list, output_file: Path) -> None:
    """
    Visualize the binary pattern as a 2D image.
    
    Args:
        pattern: List of 0s and 1s
        output_file: Path to save the visualization
    """
    # Determine dimensions - make it as square as possible
    length = len(pattern)
    width = int(np.sqrt(length))
    height = (length // width) + (1 if length % width else 0)
    
    # Create image array and fill with pattern
    img = np.zeros((height, width), dtype=np.uint8)
    for i, bit in enumerate(pattern):
        if i < height * width:
            row = i // width
            col = i % width
            img[row, col] = bit * 255
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='binary')
    plt.title(f"Binary Pattern ({len(pattern)} bits)")
    plt.grid(True, alpha=0.2)
    
    # Add bit values as text for small patterns
    if len(pattern) <= 100:
        for i, bit in enumerate(pattern):
            if i < height * width:
                row = i // width
                col = i % width
                plt.text(col, row, str(bit), ha="center", va="center", 
                         color="red" if bit else "blue", fontweight="bold")
    
    plt.savefig(output_file)
    print(f"Saved pattern visualization to {output_file}")


def visualize_as_time_slots(data: dict, output_file: Path) -> None:
    """
    Visualize the pattern broken down by time slots.
    
    Args:
        data: Pattern data dictionary
        output_file: Path to save the visualization
    """
    time_slots = data["by_time_slot"]
    
    # Determine number of rows based on available time slots
    nrows = len(time_slots)
    
    plt.figure(figsize=(12, 2 * nrows))
    
    for i, (slot_name, slot_data) in enumerate(sorted(time_slots.items())):
        pattern = slot_data["pattern"]
        
        # Create subplot
        plt.subplot(nrows, 1, i + 1)
        
        # Plot as binary sequence
        x = np.arange(len(pattern))
        plt.bar(x, pattern, width=0.6, color=['black' if bit else 'white' for bit in pattern], 
                edgecolor='gray')
        
        # Add grid and labels
        plt.grid(True, alpha=0.2)
        plt.title(f"Time Slot: {slot_name} ({len(pattern)} bits)")
        plt.xlabel("Position")
        plt.ylabel("Bit Value")
        
        # Add bit values as text
        for j, bit in enumerate(pattern):
            plt.text(j, bit/2, str(bit), ha="center", va="center", 
                     color="white" if bit else "black", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved time slot visualization to {output_file}")


def visualize_pattern_grid(data: dict, output_file: Path) -> None:
    """
    Visualize the pattern as a grid with highlighted bits.
    
    Args:
        data: Pattern data dictionary
        output_file: Path to save the visualization
    """
    # Get all time slots
    time_slots = sorted(data["by_time_slot"].keys())
    patterns = {}
    
    for slot in time_slots:
        patterns[slot] = data["by_time_slot"][slot]["pattern"]
    
    # Calculate dimensions
    n_cols = max(len(patterns[slot]) for slot in patterns)
    n_rows = len(patterns)
    
    # Create figure
    plt.figure(figsize=(max(8, n_cols/2), max(3, n_rows*1.2)))
    
    # Create a grid of cells
    for row, slot in enumerate(time_slots):
        pattern = patterns[slot]
        for col, bit in enumerate(pattern):
            color = 'black' if bit else 'white'
            plt.fill_between([col, col+1], [row, row], [row+1, row+1], color=color, edgecolor='gray')
            plt.text(col+0.5, row+0.5, str(bit), ha='center', va='center', 
                     color='white' if bit else 'black', fontweight='bold')
    
    # Customize plot
    plt.xlim(0, n_cols)
    plt.ylim(0, n_rows)
    plt.yticks(np.arange(n_rows) + 0.5, time_slots)
    plt.xticks(np.arange(0, n_cols, 1) + 0.5, np.arange(1, n_cols+1))
    plt.grid(True, alpha=0.2)
    plt.title("Binary Pattern by Time Slot")
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved pattern grid visualization to {output_file}")


def analyze_bit_distribution(pattern: list) -> None:
    """
    Analyze and print the distribution of bits in the pattern.
    
    Args:
        pattern: List of 0s and 1s
    """
    total_bits = len(pattern)
    ones = sum(pattern)
    zeros = total_bits - ones
    
    print(f"\nBit Distribution Analysis:")
    print(f"Total bits: {total_bits}")
    print(f"Ones: {ones} ({ones/total_bits:.1%})")
    print(f"Zeros: {zeros} ({zeros/total_bits:.1%})")
    
    # Check for patterns
    if total_bits % 8 == 0:
        print(f"Pattern length is divisible by 8 (could be bytes)")
    elif total_bits % 7 == 0:
        print(f"Pattern length is divisible by 7 (could be 7-bit ASCII)")
    
    # Check for divisibility by common factors
    for factor in [2, 3, 4, 5, 6, 9, 10, 12, 16]:
        if factor != 8 and factor != 7 and total_bits % factor == 0:
            print(f"Pattern length is divisible by {factor}")


def visualize_as_qr_code(pattern: list, output_file: Path) -> None:
    """
    Attempt to visualize the pattern as a QR code by arranging in a square.
    
    Args:
        pattern: List of 0s and 1s
        output_file: Path to save the visualization
    """
    # Check if we can form a reasonable square
    length = len(pattern)
    
    # Find the nearest perfect square
    side_length = int(np.sqrt(length))
    
    # If not enough bits, pad with zeros
    if side_length * side_length < length:
        side_length += 1
    
    # Create a square array
    square_size = side_length * side_length
    padded_pattern = pattern + [0] * (square_size - length)
    
    # Reshape into a square
    qr_array = np.array(padded_pattern).reshape(side_length, side_length)
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(qr_array, cmap='binary', interpolation='nearest')
    plt.title(f"Pattern as {side_length}x{side_length} QR-like code")
    plt.axis('off')
    
    plt.savefig(output_file)
    print(f"Saved QR-like visualization to {output_file}")


def main() -> None:
    """Main entry point for the pattern visualization script."""
    parser = argparse.ArgumentParser(description='Visualize binary pattern from steganography analysis')
    parser.add_argument('--pattern-file', type=str, default='results/stego_pattern/extracted_pattern.json',
                      help='Path to the pattern JSON file')
    parser.add_argument('--output-dir', type=str, default='results/stego_pattern/visualizations',
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
    
    # Analyze bit distribution
    analyze_bit_distribution(pattern)
    
    # Create visualizations
    visualize_as_image(pattern, output_dir / "pattern_image.png")
    visualize_as_time_slots(data, output_dir / "time_slots.png")
    visualize_pattern_grid(data, output_dir / "pattern_grid.png")
    visualize_as_qr_code(pattern, output_dir / "qr_code.png")
    
    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    main() 