#!/usr/bin/env python3
"""
Decoder for the 60-bit pattern from YZY steganography analysis.

This script analyzes the 60-bit binary pattern (000000110000000000000011000000110011000000000000110011111100)
extracted from 60 news images, focusing on multiple theories:
1. Visualizing the pattern in different arrangements
2. Mapping bit positions to coordinate systems
3. Testing cryptographic connections to keywords
4. Analyzing bit distributions and relationships
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import base64
import hashlib
import binascii
import itertools
from typing import List, Dict, Any, Tuple, Optional

# For Blake hash functions
try:
    import pyblake2  # Try to import pyblake2 for Blake2
except ImportError:
    pyblake2 = None  # Flag that pyblake2 is not available

# The 60-bit pattern extracted from the 60 news images
PATTERN = [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]

PATTERN_STRING = "000000110000000000000011000000110011000000000000110011111100"

# Distribution by time slots (10AM: 0-19, 3PM: 20-39, 8PM: 40-59)
TIME_SLOTS = {
    "10AM": PATTERN[:20],
    "3PM": PATTERN[20:40],
    "8PM": PATTERN[40:60]
}

# Key terms to search for connections
KEY_TERMS = [
    "4NBT",
    "4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump",
    "YZY",
    "Silver",
    "333",
    "353"
]


def visualize_as_constellation(rows: int, cols: int, output_dir: Path) -> None:
    """
    Visualize the pattern as a constellation with 1s as stars in a grid.
    
    Args:
        rows: Number of rows for the arrangement
        cols: Number of columns for the arrangement
        output_dir: Directory to save the visualization
    """
    if rows * cols != len(PATTERN):
        print(f"Error: {rows}x{cols} matrix has {rows*cols} cells, but pattern has {len(PATTERN)} bits")
        return
        
    # Create matrix
    matrix = np.array(PATTERN).reshape(rows, cols)
    
    # Create constellation visualization
    plt.figure(figsize=(cols/1.5, rows/1.5))
    plt.grid(False)
    plt.axis('off')
    
    # Black background
    plt.gca().set_facecolor('black')
    
    # Plot stars
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] == 1:
                # Draw star (with random slight size variation for realism)
                size = 100 + np.random.randint(-20, 20)
                plt.scatter(j, i, s=size, color='white', alpha=0.9)
                
                # Add subtle glow
                plt.scatter(j, i, s=size*2, color='white', alpha=0.1)
                plt.scatter(j, i, s=size*3, color='white', alpha=0.05)
    
    # Invert y-axis for star map convention
    plt.gca().invert_yaxis()
    
    # Save the figure
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"constellation_{rows}x{cols}.png", bbox_inches='tight', facecolor='black')
    plt.close()
    
    print(f"Saved constellation visualization as {rows}x{cols} to {output_dir}")


def map_to_coordinates(rows: int, cols: int) -> Dict[str, List[Tuple[int, int]]]:
    """
    Map the pattern bits to coordinates in different arrangements.
    
    Args:
        rows: Number of rows for the arrangement
        cols: Number of columns for the arrangement
        
    Returns:
        Dictionary with coordinates of 1s and 0s
    """
    if rows * cols != len(PATTERN):
        return {"error": f"{rows}x{cols} matrix has {rows*cols} cells, but pattern has {len(PATTERN)} bits"}
    
    # Create matrix
    matrix = np.array(PATTERN).reshape(rows, cols)
    
    # Extract coordinates
    ones_coords = []
    zeros_coords = []
    
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] == 1:
                ones_coords.append((i+1, j+1))  # 1-indexed
            else:
                zeros_coords.append((i+1, j+1))  # 1-indexed
    
    return {
        "ones": ones_coords,
        "zeros": zeros_coords
    }


def try_ascii_decoding() -> Dict[str, str]:
    """
    Try decoding the pattern as ASCII in various ways.
    
    Returns:
        Dictionary with different ASCII interpretations
    """
    results = {}
    
    # Try as ASCII with LSB first
    for bits_per_char in [7, 8]:
        chars = []
        for i in range(0, len(PATTERN), bits_per_char):
            if i + bits_per_char <= len(PATTERN):
                byte = 0
                for j in range(bits_per_char):
                    byte |= PATTERN[i + j] << j
                chars.append(chr(byte) if 32 <= byte <= 126 else '.')
        
        text = ''.join(chars)
        results[f"{bits_per_char}-bit_lsb"] = text
    
    # Try as ASCII with MSB first
    for bits_per_char in [7, 8]:
        chars = []
        for i in range(0, len(PATTERN), bits_per_char):
            if i + bits_per_char <= len(PATTERN):
                byte = 0
                for j in range(bits_per_char):
                    byte |= PATTERN[i + j] << (bits_per_char - 1 - j)
                chars.append(chr(byte) if 32 <= byte <= 126 else '.')
        
        text = ''.join(chars)
        results[f"{bits_per_char}-bit_msb"] = text
    
    return results


def test_keyword_connections() -> Dict[str, Any]:
    """
    Test connections between the pattern and keywords.
    
    Returns:
        Dictionary with results of keyword tests
    """
    results = {}
    
    # Check if pattern contains bits representing ASCII values of keywords
    for term in KEY_TERMS:
        term_bits = []
        for char in term:
            for i in range(8):
                term_bits.append(1 if (ord(char) & (1 << i)) else 0)
        
        # Check if pattern contains this sequence
        term_str = ''.join(str(b) for b in term_bits)
        pattern_str = PATTERN_STRING
        
        if term_str in pattern_str:
            results[f"{term}_found"] = {
                "position": pattern_str.index(term_str),
                "bit_sequence": term_str
            }
    
    # Check if XORing the pattern with keywords produces interesting results
    for term in KEY_TERMS:
        term_bytes = term.encode('utf-8')
        xor_result = []
        
        for i in range(len(PATTERN)):
            char_idx = i % len(term)
            bit_idx = i % 8
            term_bit = 1 if (term_bytes[char_idx] & (1 << bit_idx)) else 0
            xor_result.append(PATTERN[i] ^ term_bit)
        
        # Count runs of 1s and 0s
        runs = []
        current_run = {"bit": xor_result[0], "length": 1}
        
        for i in range(1, len(xor_result)):
            if xor_result[i] == current_run["bit"]:
                current_run["length"] += 1
            else:
                runs.append(current_run)
                current_run = {"bit": xor_result[i], "length": 1}
        
        runs.append(current_run)
        
        # Save results if there are interesting patterns (long runs)
        if any(run["length"] >= 6 for run in runs):
            results[f"{term}_xor"] = {
                "runs": runs,
                "result": ''.join(str(b) for b in xor_result)
            }
    
    return results


def try_42_significance() -> Dict[str, Any]:
    """
    Explore the significance of the number 42 in the pattern.
    ASCII 42 = '*' (star), connecting to constellation theory.
    
    Returns:
        Dictionary with analysis of number 42
    """
    results = {}
    
    # Count 1s (16) and 0s (44) in the pattern
    ones_count = sum(PATTERN)
    zeros_count = len(PATTERN) - ones_count
    
    results["bit_counts"] = {
        "ones": ones_count,
        "zeros": zeros_count
    }
    
    # ASCII star '*' is 42
    if zeros_count == 42 or ones_count == 42:
        results["star_connection"] = True
    
    # Coordinates summing to 42
    for rows, cols in [(6, 10), (5, 12), (4, 15), (3, 20)]:
        coords = map_to_coordinates(rows, cols)
        
        sums_to_42 = []
        for i, (row, col) in enumerate(coords["ones"]):
            if row + col == 42:
                sums_to_42.append((i, row, col))
        
        if sums_to_42:
            if "coordinate_42_sums" not in results:
                results["coordinate_42_sums"] = {}
            results["coordinate_42_sums"][f"{rows}x{cols}"] = sums_to_42
    
    return results


def try_time_distribution_analysis() -> Dict[str, Any]:
    """
    Analyze the distribution of bits across time slots (10AM, 3PM, 8PM).
    
    Returns:
        Dictionary with analysis of time-based bit distribution
    """
    results = {}
    
    # Count 1s in each time slot
    for slot, bits in TIME_SLOTS.items():
        ones_count = sum(bits)
        zeros_count = len(bits) - ones_count
        results[slot] = {
            "ones": ones_count,
            "zeros": zeros_count,
            "ones_ratio": ones_count / len(bits),
            "pattern": ''.join(str(b) for b in bits)
        }
    
    # Visualize distribution
    plt.figure(figsize=(10, 6))
    slots = list(TIME_SLOTS.keys())
    ones_counts = [results[slot]["ones"] for slot in slots]
    
    plt.bar(slots, ones_counts, color='blue')
    plt.title('Distribution of 1s across Time Slots')
    plt.ylabel('Number of 1s')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, count in enumerate(ones_counts):
        plt.text(i, count + 0.1, str(count), ha='center')
    
    plt.savefig(Path("stego_results") / "time_distribution.png")
    plt.close()
    
    return results


def try_bit_as_indices() -> Dict[str, Any]:
    """
    Try using bit positions as indices into various sequences.
    
    Returns:
        Dictionary with results from bit position indexing
    """
    results = {}
    
    # Get positions of 1s (0-indexed)
    positions = [i for i, bit in enumerate(PATTERN) if bit == 1]
    results["positions_of_ones"] = positions
    
    # Try treating positions as indices into alphabet
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    alphabet_result = []
    
    for pos in positions:
        if pos < len(alphabet):
            alphabet_result.append(alphabet[pos])
    
    if alphabet_result:
        results["as_alphabet_indices"] = ''.join(alphabet_result)
    
    # Try treating positions as indices into ASCII range
    ascii_result = []
    for pos in positions:
        if 32 <= pos <= 126:  # Printable ASCII range
            ascii_result.append(chr(pos))
    
    if ascii_result:
        results["as_ascii_indices"] = ''.join(ascii_result)
    
    return results


def try_binary_transformations() -> Dict[str, Any]:
    """
    Try various transformations of the binary pattern.
    
    Returns:
        Dictionary with results from different transformations
    """
    results = {}
    
    # Reversal
    reversed_pattern = PATTERN[::-1]
    results["reversed"] = ''.join(str(b) for b in reversed_pattern)
    
    # Complement (flip bits)
    complemented = [1 - b for b in PATTERN]
    results["complemented"] = ''.join(str(b) for b in complemented)
    
    # Rotations (circular shifts)
    for shift in [1, 2, 3, 6, 12, 20, 30]:
        rotated = PATTERN[shift:] + PATTERN[:shift]
        results[f"rotated_{shift}"] = ''.join(str(b) for b in rotated)
    
    # Group into tuples and interpret as decimal
    for group_size in [2, 3, 4, 5, 6]:
        if 60 % group_size == 0:
            groups = []
            for i in range(0, len(PATTERN), group_size):
                value = 0
                for j in range(group_size):
                    value = (value << 1) | PATTERN[i + j]
                groups.append(value)
            
            results[f"grouped_{group_size}"] = groups
            
            # Try to interpret as ASCII
            ascii_result = []
            for val in groups:
                if 32 <= val <= 126:
                    ascii_result.append(chr(val))
                else:
                    ascii_result.append('.')
            
            results[f"grouped_{group_size}_as_ascii"] = ''.join(ascii_result)
    
    return results


def try_cryptographic_approaches() -> Dict[str, Any]:
    """
    Try various cryptographic approaches with the pattern.
    
    Returns:
        Dictionary with cryptographic analysis results
    """
    results = {}
    
    # Convert pattern to bytes
    pattern_bytes = bytearray()
    for i in range(0, len(PATTERN), 8):
        if i + 8 <= len(PATTERN):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | PATTERN[i + j]
            pattern_bytes.append(byte)
    
    results["as_bytes"] = list(pattern_bytes)
    results["as_hex"] = pattern_bytes.hex()
    
    # Try as hash input
    for algo in ['md5', 'sha1', 'sha256']:
        h = hashlib.new(algo)
        h.update(pattern_bytes)
        results[f"{algo}_hash"] = h.hexdigest()
        
        # Try to find connections between hash and keywords
        for term in KEY_TERMS:
            if term.lower() in h.hexdigest().lower():
                results[f"{algo}_hash_contains_{term}"] = True
    
    # Try Blake2 hash if available
    if pyblake2:
        b2b = pyblake2.blake2b(pattern_bytes)
        b2s = pyblake2.blake2s(pattern_bytes)
        results["blake2b_hash"] = b2b.hexdigest()
        results["blake2s_hash"] = b2s.hexdigest()
        
        # Check for keywords in Blake hashes
        for term in KEY_TERMS:
            if term.lower() in b2b.hexdigest().lower():
                results[f"blake2b_hash_contains_{term}"] = True
            if term.lower() in b2s.hexdigest().lower():
                results[f"blake2s_hash_contains_{term}"] = True
    else:
        # Try to use hashlib's blake implementation (Python 3.6+)
        try:
            # Blake2b
            blake2b = hashlib.blake2b()
            blake2b.update(pattern_bytes)
            results["blake2b_hash"] = blake2b.hexdigest()
            
            # Blake2s
            blake2s = hashlib.blake2s()
            blake2s.update(pattern_bytes)
            results["blake2s_hash"] = blake2s.hexdigest()
            
            # Check for keywords in Blake hashes
            for term in KEY_TERMS:
                if term.lower() in blake2b.hexdigest().lower():
                    results[f"blake2b_hash_contains_{term}"] = True
                if term.lower() in blake2s.hexdigest().lower():
                    results[f"blake2s_hash_contains_{term}"] = True
        except (AttributeError, ValueError):
            results["blake_hash_error"] = "Blake hash functions not available"
    
    # Treat as a key for decrypt test messages
    for term in KEY_TERMS:
        term_bytes = term.encode('utf-8')
        # XOR decrypt
        xor_result = bytearray()
        for i in range(len(term_bytes)):
            key_byte = pattern_bytes[i % len(pattern_bytes)]
            xor_result.append(term_bytes[i] ^ key_byte)
        
        try:
            decoded = xor_result.decode('utf-8', errors='replace')
            results[f"decrypt_{term}"] = decoded
        except:
            pass
    
    return results


def try_text_focused_decoding() -> Dict[str, Any]:
    """
    Try more text-focused approaches to decode the binary pattern.
    
    Returns:
        Dictionary with text-focused decoding results
    """
    results = {}
    
    # Try different bit groupings for text
    group_results = {}
    for bits_per_group in [5, 6, 7, 8]:
        if len(PATTERN) % bits_per_group == 0:
            # MSB first
            groups_msb = []
            for i in range(0, len(PATTERN), bits_per_group):
                value = 0
                for j in range(bits_per_group):
                    value = (value << 1) | PATTERN[i + j]
                groups_msb.append(value)
            
            group_results[f"{bits_per_group}bit_groups_msb"] = groups_msb
            
            # LSB first
            groups_lsb = []
            for i in range(0, len(PATTERN), bits_per_group):
                value = 0
                for j in range(bits_per_group):
                    value |= PATTERN[i + j] << j
                groups_lsb.append(value)
            
            group_results[f"{bits_per_group}bit_groups_lsb"] = groups_lsb
    
    # Store the grouped values in results
    results.update(group_results)
    
    # ASCII interpretation by different encodings
    for bits_per_char, groups in group_results.items():
        if "msb" in bits_per_char:
            # For 5-bit groups, try Baudot/ITA2 code
            if "5bit" in bits_per_char:
                # Simple Baudot/ITA2-inspired mapping (limited)
                baudot_letters = {
                    0: ' ', 1: 'E', 2: 'A', 3: 'S', 4: 'I', 5: 'U', 
                    6: 'D', 7: 'R', 8: 'N', 9: 'T', 10: 'O', 11: 'L',
                    12: 'H', 13: 'F', 14: 'P', 15: 'Z'
                }
                
                text = []
                for val in groups:
                    if val in baudot_letters:
                        text.append(baudot_letters[val])
                    else:
                        text.append('.')
                
                results[f"{bits_per_char}_as_baudot"] = ''.join(text)
            
            # For 6-bit groups, try Base64-like encoding
            if "6bit" in bits_per_char:
                # Simple 6-bit to ASCII mapping
                base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
                
                text = []
                for val in groups:
                    if 0 <= val < len(base64_chars):
                        text.append(base64_chars[val])
                    else:
                        text.append('.')
                
                results[f"{bits_per_char}_as_base64"] = ''.join(text)
            
            # For 7-bit and 8-bit groups, try ASCII
            if "7bit" in bits_per_char or "8bit" in bits_per_char:
                text = []
                for val in groups:
                    if 32 <= val <= 126:  # Printable ASCII
                        text.append(chr(val))
                    else:
                        text.append('.')
                
                results[f"{bits_per_char}_as_ascii"] = ''.join(text)
    
    # Try morse code interpretation (1 = dash, 0 = dot)
    morse_map = {
        ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
        "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
        "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
        ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
        "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
        "--..": "Z", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
        ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
        "-----": "0", ".-.-.-": ".", "--..--": ",", "..--..": "?",
        ".----.": "'", "-.-.--": "!", "-..-.": "/", "-.--.": "(", 
        "-.--.-": ")", ".-...": "&", "---...": ":", "-.-.-.": ";",
        "-...-": "=", ".-.-.": "+", "-....-": "-", "..--.-": "_",
        ".-..-.": "\"", "...-..-": "$", ".--.-.": "@"
    }
    
    # Convert 1s to dashes, 0s to dots
    morse_dots_dashes = ''.join(['-' if b == 1 else '.' for b in PATTERN])
    results["as_morse_dots_dashes"] = morse_dots_dashes
    
    # Try to decode as morse with different word breaks
    # Simplistic approach just trying a few split patterns
    for split_size in [3, 4, 5, 6]:
        morse_chars = []
        for i in range(0, len(morse_dots_dashes), split_size):
            if i + split_size <= len(morse_dots_dashes):
                morse_char = morse_dots_dashes[i:i+split_size]
                if morse_char in morse_map:
                    morse_chars.append(morse_map[morse_char])
                else:
                    morse_chars.append('?')
        
        if morse_chars:
            results[f"morse_decode_split{split_size}"] = ''.join(morse_chars)
    
    # Try common substitution ciphers on pattern string
    # Convert binary to A=0, B=1 and try ROT13/Caesar
    ab_text = ''.join(['B' if b == 1 else 'A' for b in PATTERN])
    results["as_ab_text"] = ab_text
    
    # ROT13 on AB text
    rot13_ab = ''.join([chr((ord(c) - ord('A') + 13) % 26 + ord('A')) for c in ab_text])
    results["rot13_on_ab"] = rot13_ab
    
    # Try as a simple columnar transposition cipher key
    # Using the bit pattern to determine column order
    columnar_results = []
    message = "THISISATESTMESSAGE"  # Test with a known message
    
    # Use just first 20 bits for columns (arbitrary)
    col_pattern = PATTERN[:min(20, len(PATTERN))]
    col_order = sorted(range(len(col_pattern)), key=lambda i: (col_pattern[i], i))
    
    # Arrange message in columns, then read by new order
    cols = [''] * len(col_pattern)
    for i, char in enumerate(message):
        cols[i % len(col_pattern)] += char
    
    transposed = ''.join(cols[i] for i in col_order)
    results["columnar_transposition_test"] = transposed
    
    return results


def try_coordinates_in_other_systems() -> Dict[str, Any]:
    """
    Try interpreting bit positions as coordinates in alternative systems.
    
    Returns:
        Dictionary with coordinate system interpretations
    """
    results = {}
    
    # Get positions of 1s
    positions = [i for i, bit in enumerate(PATTERN) if bit == 1]
    
    # Interpret as (lat, long) pairs
    if len(positions) % 2 == 0:
        coord_pairs = []
        for i in range(0, len(positions), 2):
            # Scale to reasonable lat/long values
            lat = (positions[i] / 60) * 180 - 90  # -90 to 90
            lon = (positions[i+1] / 60) * 360 - 180  # -180 to 180
            coord_pairs.append((lat, lon))
        
        results["as_lat_long"] = coord_pairs
    
    # Interpret as bit positions in a 8x8 grid (like chess)
    chess_positions = []
    for pos in positions:
        if pos < 64:  # 8x8 grid
            row = pos // 8
            col = pos % 8
            # Convert to chess notation
            chess_pos = f"{chr(97 + col)}{8 - row}"
            chess_positions.append(chess_pos)
    
    results["as_chess_positions"] = chess_positions
    
    return results


def try_time_sequence_patterns() -> Dict[str, Any]:
    """
    Analyze the pattern as a time sequence across the news cycle.
    
    Returns:
        Dictionary with time sequence analysis
    """
    results = {}
    
    # Combine time slots in different order
    all_orders = list(itertools.permutations(TIME_SLOTS.keys()))
    
    for order in all_orders:
        combined = []
        for slot in order:
            combined.extend(TIME_SLOTS[slot])
        
        results[f"order_{'_'.join(order)}"] = ''.join(str(b) for b in combined)
    
    # Look for patterns in transitions between time slots
    transition_patterns = {}
    time_order = ["10AM", "3PM", "8PM"]
    
    for i in range(len(time_order) - 1):
        current = TIME_SLOTS[time_order[i]]
        next_slot = TIME_SLOTS[time_order[i+1]]
        
        # Count transition patterns (00->00, 00->01, 01->00, etc)
        for j in range(len(current) - 1):
            transition = f"{current[j]}{current[j+1]}->{next_slot[j]}{next_slot[j+1]}"
            transition_patterns[transition] = transition_patterns.get(transition, 0) + 1
    
    results["transition_patterns"] = transition_patterns
    
    return results


def main() -> None:
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description="Decode the 60-bit YZY steganography pattern")
    parser.add_argument("--output-dir", type=Path, default=Path("stego_results"),
                       help="Directory to save results")
    parser.add_argument("--focus", type=str, default="all",
                        choices=["all", "text", "crypto", "visual"],
                        help="Focus the analysis on a specific approach")
    args = parser.parse_args()
    
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("YZY Steganography Pattern Decoder")
    print("=================================")
    print(f"Pattern: {PATTERN_STRING}")
    print(f"Length: {len(PATTERN)} bits")
    print(f"1s: {sum(PATTERN)}, 0s: {len(PATTERN) - sum(PATTERN)}")
    print()
    
    # Initialize result dictionaries
    ascii_results = {}
    keyword_results = {}
    significance_42 = {}
    time_results = {}
    position_results = {}
    transform_results = {}
    crypto_results = {}
    coord_results = {}
    time_sequence_results = {}
    text_focused_results = {}
    
    # Run analyses based on focus
    if args.focus in ["all", "visual"]:
        # Try key arrangements
        for rows, cols in [(6, 10), (5, 12), (10, 6), (12, 5), (4, 15), (15, 4), (3, 20), (20, 3)]:
            print(f"Analyzing {rows}x{cols} arrangement...")
            visualize_as_constellation(rows, cols, output_dir)
            
            coords = map_to_coordinates(rows, cols)
            print(f"Star coordinates (1s) in {rows}x{cols} arrangement:")
            for i, (row, col) in enumerate(coords["ones"]):
                print(f"  Star {i+1}: ({row}, {col})")
            print()
        
        # Explore 42 significance
        significance_42 = try_42_significance()
        print("Analysis of number 42 significance:")
        for key, result in significance_42.items():
            print(f"  {key}: {result}")
        print()
    
    if args.focus in ["all", "text", "crypto"]:
        # Try ASCII decoding
        ascii_results = try_ascii_decoding()
        print("ASCII decoding attempts:")
        for method, result in ascii_results.items():
            print(f"  {method}: {result}")
        print()
        
        # New text-focused decoding approaches
        text_focused_results = try_text_focused_decoding()
        print("Text-focused decoding approaches:")
        for key, value in text_focused_results.items():
            if isinstance(value, list) and len(value) > 10:
                print(f"  {key}: {value[:10]}... ({len(value)} items)")
            elif isinstance(value, str) and len(value) > 30:
                print(f"  {key}: {value[:30]}... ({len(value)} chars)")
            else:
                print(f"  {key}: {value}")
        print()
    
    if args.focus in ["all", "crypto"]:
        # Check keyword connections
        keyword_results = test_keyword_connections()
        print("Keyword connection tests:")
        for key, result in keyword_results.items():
            print(f"  {key}: {result}")
        print()
        
        # Cryptographic approaches
        crypto_results = try_cryptographic_approaches()
        print("Cryptographic approaches:")
        for key, value in crypto_results.items():
            if isinstance(value, list) and len(value) > 10:
                print(f"  {key}: {value[:10]}... ({len(value)} items)")
            elif isinstance(value, str) and len(value) > 30:
                print(f"  {key}: {value[:30]}... ({len(value)} chars)")
            else:
                print(f"  {key}: {value}")
        print()
    
    if args.focus in ["all"]:
        # Time distribution analysis
        time_results = try_time_distribution_analysis()
        print("Time distribution analysis:")
        for slot, data in time_results.items():
            print(f"  {slot}: {data['ones']} ones, {data['zeros']} zeros")
        print()
        
        # Bit positions as indices
        position_results = try_bit_as_indices()
        print("Bit positions as indices:")
        for key, value in position_results.items():
            if isinstance(value, list) and len(value) > 10:
                print(f"  {key}: {value[:10]}... ({len(value)} items)")
            else:
                print(f"  {key}: {value}")
        print()
        
        # Binary transformations
        transform_results = try_binary_transformations()
        print("Binary transformations:")
        for key, value in transform_results.items():
            if isinstance(value, list) and len(value) > 10:
                print(f"  {key}: {value[:10]}... ({len(value)} items)")
            elif isinstance(value, str) and len(value) > 30:
                print(f"  {key}: {value[:30]}... ({len(value)} chars)")
            else:
                print(f"  {key}: {value}")
        print()
        
        # Alternative coordinate systems
        coord_results = try_coordinates_in_other_systems()
        print("Alternative coordinate systems:")
        for key, value in coord_results.items():
            if isinstance(value, list) and len(value) > 10:
                print(f"  {key}: {value[:10]}... ({len(value)} items)")
            else:
                print(f"  {key}: {value}")
        print()
        
        # Time sequence patterns
        time_sequence_results = try_time_sequence_patterns()
        print("Time sequence patterns:")
        for key, value in time_sequence_results.items():
            if isinstance(value, dict) and len(value) > 10:
                items = list(value.items())
                print(f"  {key}: {items[:5]}... ({len(value)} items)")
            elif isinstance(value, str) and len(value) > 30:
                print(f"  {key}: {value[:30]}... ({len(value)} chars)")
            else:
                print(f"  {key}: {value}")
        print()
    
    # Save all results to JSON
    import json
    all_results = {
        "pattern": PATTERN,
        "pattern_string": PATTERN_STRING,
        "ascii_decodings": ascii_results,
        "text_focused_decodings": text_focused_results,
        "keyword_connections": keyword_results,
        "significance_42": significance_42,
        "time_distribution": time_results,
        "bit_positions": position_results,
        "binary_transformations": transform_results,
        "cryptographic_analysis": crypto_results,
        "alternative_coordinates": coord_results,
        "time_sequences": time_sequence_results,
        "coordinates": {
            f"{rows}x{cols}": map_to_coordinates(rows, cols)
            for rows, cols in [(6, 10), (5, 12), (4, 15), (3, 20)]
        }
    }
    
    with open(output_dir / "decoded_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll results saved to {output_dir}/decoded_results.json")


if __name__ == "__main__":
    main() 