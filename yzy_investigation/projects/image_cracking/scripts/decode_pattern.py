#!/usr/bin/env python3
"""
Script to decode the binary pattern from steganography analysis in various ways.

This script attempts to decode the pattern using various techniques:
1. Different bit encodings (ASCII, UTF-8, etc.)
2. Matrix rearrangements (reading by rows/columns)
3. Binary transformations (bit shifts, reversals)
4. Transposition ciphers
"""

import json
import argparse
import binascii
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple


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


def try_binary_decoding(pattern: List[int]) -> None:
    """
    Try various binary decoding methods on the pattern.
    
    Args:
        pattern: List of 0s and 1s
    """
    pattern_str = ''.join(str(b) for b in pattern)
    print(f"Raw binary: {pattern_str}")
    
    # Try as ASCII with different bit widths
    for bits_per_char in [7, 8]:
        if len(pattern) >= bits_per_char:
            chars = []
            for i in range(0, len(pattern), bits_per_char):
                if i + bits_per_char <= len(pattern):
                    byte = 0
                    for j in range(bits_per_char):
                        byte |= pattern[i + j] << j
                    chars.append(chr(byte) if 32 <= byte <= 126 else '.')
            
            text = ''.join(chars)
            print(f"{bits_per_char}-bit ASCII (LSB first): {text}")
    
    # Try as ASCII with MSB first
    for bits_per_char in [7, 8]:
        if len(pattern) >= bits_per_char:
            chars = []
            for i in range(0, len(pattern), bits_per_char):
                if i + bits_per_char <= len(pattern):
                    byte = 0
                    for j in range(bits_per_char):
                        byte |= pattern[i + j] << (bits_per_char - 1 - j)
                    chars.append(chr(byte) if 32 <= byte <= 126 else '.')
            
            text = ''.join(chars)
            print(f"{bits_per_char}-bit ASCII (MSB first): {text}")
    
    # Try as hexadecimal
    if len(pattern) >= 4:
        hex_chars = []
        for i in range(0, len(pattern), 4):
            if i + 4 <= len(pattern):
                nibble = 0
                for j in range(4):
                    nibble |= pattern[i + j] << j
                hex_chars.append(hex(nibble)[2:])
        
        hex_str = ''.join(hex_chars)
        print(f"Hexadecimal: {hex_str}")
        
        # Try to convert hex to ASCII
        if len(hex_str) % 2 == 0:
            try:
                hex_bytes = bytes.fromhex(hex_str)
                hex_ascii = hex_bytes.decode('ascii', errors='replace')
                print(f"Hex as ASCII: {hex_ascii}")
            except:
                pass


def try_matrix_arrangements(pattern: List[int], output_dir: Path) -> None:
    """
    Try arranging the pattern into different matrix shapes and read in different ways.
    
    Args:
        pattern: List of 0s and 1s
        output_dir: Output directory for visualizations
    """
    length = len(pattern)
    
    # Find all divisors of the pattern length
    divisors = [i for i in range(2, length + 1) if length % i == 0]
    
    results = []
    
    for width in divisors:
        height = length // width
        
        # Skip if too extreme dimensions
        if width > 10 * height or height > 10 * width:
            continue
            
        # Create matrix (filled by rows)
        matrix = np.array(pattern).reshape(height, width)
        
        # Read by rows
        row_bits = matrix.flatten()
        row_pattern = ''.join(str(int(b)) for b in row_bits)
        
        # Read by columns
        col_bits = matrix.T.flatten()
        col_pattern = ''.join(str(int(b)) for b in col_bits)
        
        # Try reading row-wise and interpret as ASCII (both LSB and MSB first)
        row_lsb_text = bits_to_ascii(row_bits, msb_first=False)
        row_msb_text = bits_to_ascii(row_bits, msb_first=True)
        
        # Try reading column-wise and interpret as ASCII (both LSB and MSB first)
        col_lsb_text = bits_to_ascii(col_bits, msb_first=False)
        col_msb_text = bits_to_ascii(col_bits, msb_first=True)
        
        # Store results
        results.append({
            "shape": f"{height}x{width}",
            "by_rows": {
                "binary": row_pattern,
                "lsb_ascii": row_lsb_text,
                "msb_ascii": row_msb_text
            },
            "by_columns": {
                "binary": col_pattern,
                "lsb_ascii": col_lsb_text,
                "msb_ascii": col_msb_text
            }
        })
        
        # Visualize the matrix
        plt.figure(figsize=(max(6, width/2), max(4, height/2)))
        plt.imshow(matrix, cmap='binary', interpolation='nearest')
        plt.title(f"{height}x{width} Matrix")
        
        # Add grid
        plt.grid(True, alpha=0.3, color='gray')
        
        # Add bit values as text if not too large
        if width * height <= 100:
            for i in range(height):
                for j in range(width):
                    plt.text(j, i, str(int(matrix[i, j])), ha='center', va='center',
                             color='red' if matrix[i, j] else 'blue', fontweight='bold')
        
        # Add row and column indices
        plt.xticks(np.arange(width), np.arange(1, width + 1))
        plt.yticks(np.arange(height), np.arange(1, height + 1))
        
        plt.tight_layout()
        plt.savefig(output_dir / f"matrix_{height}x{width}.png")
        plt.close()
    
    # Print results
    print("\nMatrix Arrangement Analysis:")
    for result in results:
        print(f"\nMatrix shape: {result['shape']}")
        print(f"  By rows (LSB first): {result['by_rows']['lsb_ascii']}")
        print(f"  By rows (MSB first): {result['by_rows']['msb_ascii']}")
        print(f"  By columns (LSB first): {result['by_columns']['lsb_ascii']}")
        print(f"  By columns (MSB first): {result['by_columns']['msb_ascii']}")
        
    return results


def bits_to_ascii(bits: np.ndarray, msb_first: bool = False) -> str:
    """
    Convert a bit array to ASCII text.
    
    Args:
        bits: Array of 0s and 1s
        msb_first: Whether the most significant bit comes first
        
    Returns:
        ASCII text representation
    """
    if len(bits) < 8:
        return ""
        
    chars = []
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for j in range(8):
            if msb_first:
                byte |= bits[i + j] << (7 - j)
            else:
                byte |= bits[i + j] << j
        chars.append(chr(byte) if 32 <= byte <= 126 else '.')
    
    return ''.join(chars)


def try_visual_messages(pattern: List[int], output_dir: Path) -> None:
    """
    Check if the pattern forms a visual message when arranged in specified ways.
    
    Args:
        pattern: List of 0s and 1s
        output_dir: Output directory for visualizations
    """
    length = len(pattern)
    
    # Specific arrangements that might form visual messages
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
            
        # Create matrices in different orientations
        matrix = np.array(pattern).reshape(rows, cols)
        
        # Visualize regular matrix
        plt.figure(figsize=(max(6, cols/1.5), max(4, rows/1.5)))
        plt.imshow(matrix, cmap='binary', interpolation='nearest')
        plt.title(f"Visual Message: {rows}x{cols}")
        plt.grid(False)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / f"visual_{rows}x{cols}.png")
        plt.close()
        
        # Visualize with inverted colors
        plt.figure(figsize=(max(6, cols/1.5), max(4, rows/1.5)))
        plt.imshow(1 - matrix, cmap='binary', interpolation='nearest')
        plt.title(f"Visual Message: {rows}x{cols} (Inverted)")
        plt.grid(False)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / f"visual_{rows}x{cols}_inverted.png")
        plt.close()


def try_time_slot_patterns(data: dict, output_dir: Path) -> None:
    """
    Analyze patterns in time slots.
    
    Args:
        data: Pattern data dictionary
        output_dir: Output directory for visualizations
    """
    # Get patterns by time slot
    time_slots = sorted(data["by_time_slot"].keys())
    patterns = {}
    
    for slot in time_slots:
        patterns[slot] = data["by_time_slot"][slot]["pattern"]
    
    # Try extracting a character from each time slot
    print("\nTime Slot Analysis:")
    
    # Each time slot as a separate byte
    for msb_first in [False, True]:
        message = []
        for slot in time_slots:
            pattern = patterns[slot]
            
            # Try to interpret as ASCII if long enough
            if len(pattern) >= 8:
                byte = 0
                for j in range(8):
                    if msb_first:
                        byte |= pattern[j] << (7 - j)
                    else:
                        byte |= pattern[j] << j
                message.append(chr(byte) if 32 <= byte <= 126 else '.')
        
        orientation = "MSB first" if msb_first else "LSB first"
        print(f"  First byte from each time slot ({orientation}): {''.join(message)}")
    
    # Try interleaving bits from time slots
    interleaved = []
    max_length = max(len(patterns[slot]) for slot in time_slots)
    
    for i in range(max_length):
        for slot in time_slots:
            if i < len(patterns[slot]):
                interleaved.append(patterns[slot][i])
    
    interleaved_text_lsb = bits_to_ascii(interleaved, msb_first=False)
    interleaved_text_msb = bits_to_ascii(interleaved, msb_first=True)
    
    print(f"  Interleaved bits (LSB first): {interleaved_text_lsb}")
    print(f"  Interleaved bits (MSB first): {interleaved_text_msb}")


def try_transposition(pattern: List[int]) -> None:
    """
    Try various transposition patterns on the binary string.
    
    Args:
        pattern: List of 0s and 1s
    """
    pattern_str = ''.join(str(b) for b in pattern)
    print("\nTransposition Analysis:")
    
    # Try reversing the pattern
    reversed_pattern = pattern_str[::-1]
    print(f"  Reversed: {reversed_pattern}")
    
    # Try common transposition key lengths
    for key_length in [2, 3, 4, 5, 6]:
        if len(pattern) % key_length == 0:
            # Column-wise reading
            columns = [pattern_str[i::key_length] for i in range(key_length)]
            transposed = ''.join(columns)
            print(f"  Transposed (key length {key_length}): {transposed}")
    
    # Try some known cryptographic methods
    # Rail fence cipher with different heights
    for rails in [2, 3, 4]:
        rail_fence = rail_fence_decode(pattern_str, rails)
        print(f"  Rail fence (rails={rails}): {rail_fence}")


def rail_fence_decode(ciphertext: str, rails: int) -> str:
    """
    Decode a rail fence cipher.
    
    Args:
        ciphertext: The encoded text
        rails: Number of rails
        
    Returns:
        Decoded text
    """
    # Create the fence pattern
    fence = [[] for _ in range(rails)]
    rail = 0
    direction = 1  # 1 for down, -1 for up
    
    # Populate the fence with indexes
    for i in range(len(ciphertext)):
        fence[rail].append(i)
        rail += direction
        if rail == rails - 1 or rail == 0:
            direction *= -1
    
    # Read off the fence
    result = [''] * len(ciphertext)
    i = 0
    for rail_content in fence:
        for pos in rail_content:
            if i < len(ciphertext):
                result[pos] = ciphertext[i]
                i += 1
    
    return ''.join(result)


def main() -> None:
    """Main entry point for the pattern decoding script."""
    parser = argparse.ArgumentParser(description='Decode binary pattern from steganography analysis')
    parser.add_argument('--pattern-file', type=str, default='results/stego_pattern/extracted_pattern.json',
                      help='Path to the pattern JSON file')
    parser.add_argument('--output-dir', type=str, default='results/stego_pattern/decoding',
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
    
    print("\n=== Basic Binary Decoding ===")
    try_binary_decoding(pattern)
    
    print("\n=== Matrix Arrangement Analysis ===")
    try_matrix_arrangements(pattern, output_dir)
    
    print("\n=== Time Slot Pattern Analysis ===")
    try_time_slot_patterns(data, output_dir)
    
    print("\n=== Transposition Analysis ===")
    try_transposition(pattern)
    
    print("\n=== Visual Message Analysis ===")
    try_visual_messages(pattern, output_dir)
    
    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    main() 