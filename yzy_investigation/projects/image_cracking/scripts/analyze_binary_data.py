#!/usr/bin/env python3
"""
Script to analyze binary data for patterns and hidden information.

This script examines binary data files for:
1. Common file signatures
2. Byte frequency distribution
3. Repeating patterns
4. Possible encodings
"""

import sys
import argparse
import binascii
import collections
import hashlib
import math
import struct
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
import json
import re


def check_file_signatures(data: bytes) -> List[Dict[str, Any]]:
    """
    Check for common file signatures in binary data.
    
    Args:
        data: Binary data to analyze
        
    Returns:
        List of dictionaries with information about found signatures
    """
    signatures = {
        b"\x50\x4B\x03\x04": {"type": "ZIP", "description": "ZIP archive"},
        b"\x50\x4B\x05\x06": {"type": "ZIP", "description": "Empty ZIP archive"},
        b"\x50\x4B\x07\x08": {"type": "ZIP", "description": "Spanned ZIP archive"},
        b"\x25\x50\x44\x46": {"type": "PDF", "description": "PDF document"},
        b"\xFF\xD8\xFF": {"type": "JPEG", "description": "JPEG image"},
        b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": {"type": "PNG", "description": "PNG image"},
        b"\x47\x49\x46\x38\x37\x61": {"type": "GIF", "description": "GIF87a image"},
        b"\x47\x49\x46\x38\x39\x61": {"type": "GIF", "description": "GIF89a image"},
        b"\x42\x4D": {"type": "BMP", "description": "BMP image"},
        b"\x49\x49\x2A\x00": {"type": "TIFF", "description": "TIFF image (little-endian)"},
        b"\x4D\x4D\x00\x2A": {"type": "TIFF", "description": "TIFF image (big-endian)"},
        b"\x52\x61\x72\x21\x1A\x07": {"type": "RAR", "description": "RAR archive"},
        b"\x1F\x8B\x08": {"type": "GZIP", "description": "GZIP compressed file"},
        b"\x42\x5A\x68": {"type": "BZIP2", "description": "BZIP2 compressed file"},
        b"\x37\x7A\xBC\xAF\x27\x1C": {"type": "7Z", "description": "7-Zip archive"},
        b"\x75\x73\x74\x61\x72": {"type": "TAR", "description": "TAR archive"},
        b"\x7F\x45\x4C\x46": {"type": "ELF", "description": "ELF executable"},
        b"\x4D\x5A": {"type": "EXE", "description": "DOS/Windows executable"},
        b"\xCA\xFE\xBA\xBE": {"type": "CLASS", "description": "Java class file"},
        b"\xFF\xFE": {"type": "UTF-16LE", "description": "UTF-16 little-endian text"},
        b"\xFE\xFF": {"type": "UTF-16BE", "description": "UTF-16 big-endian text"},
        b"\xEF\xBB\xBF": {"type": "UTF-8", "description": "UTF-8 text with BOM"},
        b"\x00\x61\x73\x6D": {"type": "WASM", "description": "WebAssembly binary format"},
        b"\x4F\x67\x67\x53": {"type": "OGG", "description": "OGG container format"},
        b"\x49\x44\x33": {"type": "MP3", "description": "MP3 audio (ID3 tag)"},
        b"\xFF\xFB": {"type": "MP3", "description": "MP3 audio (no ID3 tag)"},
        b"\x66\x74\x79\x70": {"type": "MP4", "description": "MP4 video/container"},
    }
    
    found = []
    
    # Check for signatures at the start of the file
    for sig, info in signatures.items():
        if data.startswith(sig):
            found.append({
                "position": 0,
                "signature": binascii.hexlify(sig).decode('ascii'),
                "type": info["type"],
                "description": info["description"],
                "location": "start"
            })
    
    # Check for signatures anywhere in the file
    for sig, info in signatures.items():
        pos = 0
        while True:
            pos = data.find(sig, pos)
            if pos == -1 or pos == 0:  # Already found at start
                break
                
            # Only consider signatures aligned to 16/32 byte boundaries
            # This reduces false positives
            if pos % 16 == 0 or pos % 32 == 0:
                found.append({
                    "position": pos,
                    "signature": binascii.hexlify(sig).decode('ascii'),
                    "type": info["type"],
                    "description": info["description"],
                    "location": "embedded"
                })
            
            pos += 1
    
    return found


def analyze_byte_frequency(data: bytes) -> Dict[str, Any]:
    """
    Analyze the frequency distribution of bytes in the data.
    
    Args:
        data: Binary data to analyze
        
    Returns:
        Dictionary with frequency analysis results
    """
    # Count occurrences of each byte
    counter = collections.Counter(data)
    total_bytes = len(data)
    
    # Calculate entropy
    entropy = 0
    for count in counter.values():
        probability = count / total_bytes
        entropy -= probability * math.log2(probability)
    
    # Determine most and least common bytes
    most_common = counter.most_common(10)
    least_common = counter.most_common()[:-11:-1]
    
    # Check for unusual patterns
    zero_bytes = counter.get(0, 0)
    zero_percentage = (zero_bytes / total_bytes) * 100
    
    printable_bytes = sum(counter.get(b, 0) for b in range(32, 127))
    printable_percentage = (printable_bytes / total_bytes) * 100
    
    return {
        "entropy": entropy,
        "max_entropy": 8.0,  # Maximum entropy for 8-bit data
        "entropy_percentage": (entropy / 8.0) * 100,
        "most_common_bytes": [(b, c, c/total_bytes*100) for b, c in most_common],
        "least_common_bytes": [(b, c, c/total_bytes*100) for b, c in least_common],
        "zero_bytes_percentage": zero_percentage,
        "printable_ascii_percentage": printable_percentage,
        "data_type_guess": guess_data_type(entropy, printable_percentage)
    }


def guess_data_type(entropy: float, printable_percentage: float) -> str:
    """
    Make an educated guess about the type of data based on entropy and printable character percentage.
    
    Args:
        entropy: Shannon entropy of the data
        printable_percentage: Percentage of printable ASCII characters
        
    Returns:
        String describing the likely data type
    """
    if entropy < 1.0:
        return "Highly structured/repetitive data"
    elif entropy < 3.0:
        return "Low entropy data - possibly simple encoded text or sparse binary"
    elif entropy > 7.8:
        return "High entropy data - likely compressed, encrypted, or random"
    elif printable_percentage > 90:
        return "Text data - ASCII or other text encoding"
    elif printable_percentage > 60:
        return "Mostly text with some binary data"
    elif entropy > 6.0:
        return "Binary data - possibly compressed or encoded"
    else:
        return "Structured binary data"


def find_repeating_patterns(data: bytes, min_length: int = 4, max_length: int = 20) -> List[Dict[str, Any]]:
    """
    Find repeating byte patterns in the data.
    
    Args:
        data: Binary data to analyze
        min_length: Minimum pattern length to consider
        max_length: Maximum pattern length to consider
        
    Returns:
        List of dictionaries with information about repeating patterns
    """
    patterns = []
    
    # Limit the search range for performance reasons
    search_data = data[:min(len(data), 50000)]
    
    for pattern_len in range(min_length, min(max_length + 1, len(search_data) // 2)):
        pattern_counts = collections.defaultdict(list)
        
        for i in range(len(search_data) - pattern_len + 1):
            pattern = search_data[i:i+pattern_len]
            pattern_counts[pattern].append(i)
        
        # Filter patterns that appear multiple times
        for pattern, positions in pattern_counts.items():
            if len(positions) > 1:
                # Avoid reporting subpatterns of larger patterns
                is_subpattern = False
                for p in patterns:
                    if p["pattern_length"] > pattern_len and pattern in p["pattern"]:
                        is_subpattern = True
                        break
                
                if not is_subpattern:
                    patterns.append({
                        "pattern": pattern,
                        "pattern_hex": binascii.hexlify(pattern).decode('ascii'),
                        "pattern_length": pattern_len,
                        "occurrences": len(positions),
                        "positions": positions[:10]  # Limit positions for brevity
                    })
    
    # Sort by number of occurrences (most frequent first)
    patterns.sort(key=lambda x: x["occurrences"], reverse=True)
    
    # Limit to top patterns
    return patterns[:20]


def try_decode_text(data: bytes) -> Dict[str, str]:
    """
    Try to decode the data using various text encodings.
    
    Args:
        data: Binary data to analyze
        
    Returns:
        Dictionary mapping encoding names to decoded text samples
    """
    encodings = ['utf-8', 'ascii', 'latin1', 'utf-16', 'utf-16le', 'utf-16be']
    samples = {}
    
    for encoding in encodings:
        try:
            decoded = data.decode(encoding, errors='replace')
            # Take first 200 characters as sample
            sample = decoded[:200]
            # Only include if it has a reasonable amount of printable characters
            printable = sum(1 for c in sample if 32 <= ord(c) <= 126)
            if printable / len(sample) > 0.5:
                samples[encoding] = sample
        except Exception:
            pass
    
    return samples


def try_common_transformations(data: bytes) -> Dict[str, Any]:
    """
    Try common data transformations to see if they reveal anything.
    
    Args:
        data: Binary data to analyze
        
    Returns:
        Dictionary with results of various transformations
    """
    results = {}
    
    # Try bit manipulation
    if len(data) > 0:
        # Invert bits
        inverted = bytes(~b & 0xFF for b in data)
        results["inverted_bits_sample"] = binascii.hexlify(inverted[:20]).decode('ascii')
        
        # XOR with common keys
        xor_keys = [0xFF, 0x55, 0xAA, 0x00]
        xor_results = {}
        for key in xor_keys:
            xored = bytes(b ^ key for b in data)
            xor_results[f"xor_0x{key:02x}_sample"] = binascii.hexlify(xored[:20]).decode('ascii')
            
            # Check if XORed data has any recognizable file signatures
            signatures = check_file_signatures(xored)
            if signatures:
                xor_results[f"xor_0x{key:02x}_signatures"] = signatures
                
        results["xor_operations"] = xor_results
        
        # Bit rotation
        rotated = {}
        for bits in [1, 2, 4]:
            left_rotated = bytes(((b << bits) | (b >> (8 - bits))) & 0xFF for b in data)
            right_rotated = bytes(((b >> bits) | (b << (8 - bits))) & 0xFF for b in data)
            rotated[f"rotate_left_{bits}_sample"] = binascii.hexlify(left_rotated[:20]).decode('ascii')
            rotated[f"rotate_right_{bits}_sample"] = binascii.hexlify(right_rotated[:20]).decode('ascii')
            
        results["bit_rotation"] = rotated
    
    return results


def check_for_steganography(data: bytes) -> Dict[str, Any]:
    """
    Check for potential steganographic techniques in the data.
    
    Args:
        data: Binary data to analyze
        
    Returns:
        Dictionary with steganography analysis results
    """
    results = {}
    
    # Check LSB distribution
    lsb_counts = collections.Counter(b & 1 for b in data)
    total_bytes = len(data)
    
    lsb_0_percentage = (lsb_counts[0] / total_bytes) * 100
    lsb_1_percentage = (lsb_counts[1] / total_bytes) * 100
    
    # In natural data, LSBs are typically close to 50/50
    # Significant deviation might indicate steganography
    lsb_bias = abs(50 - lsb_0_percentage)
    
    results["lsb_analysis"] = {
        "lsb_0_percentage": lsb_0_percentage,
        "lsb_1_percentage": lsb_1_percentage,
        "lsb_bias": lsb_bias,
        "potential_lsb_stego": lsb_bias < 5  # Low bias suggests possible LSB steganography
    }
    
    # Extract LSBs as a bit stream
    lsb_bits = ''.join(str(b & 1) for b in data[:min(len(data), 1000)])
    results["lsb_bit_stream_sample"] = lsb_bits[:100]
    
    # Try to decode LSBs as ASCII
    lsb_bytes = bytearray()
    for i in range(0, min(len(data), 1000) - 7, 8):
        byte = 0
        for j in range(8):
            if i + j < len(data):
                byte |= ((data[i + j] & 1) << j)
        lsb_bytes.append(byte)
    
    try:
        lsb_text = lsb_bytes.decode('ascii', errors='replace')
        printable = sum(1 for c in lsb_text if 32 <= ord(c) <= 126)
        if printable / len(lsb_text) > 0.5:
            results["lsb_as_ascii_sample"] = lsb_text[:100]
    except:
        pass
    
    return results


def analyze_binary_file(file_path: Path, output_dir: Path) -> None:
    """
    Analyze a binary file and save results.
    
    Args:
        file_path: Path to the binary file
        output_dir: Directory to save analysis results
    """
    print(f"Analyzing binary file: {file_path}")
    output_dir.mkdir(exist_ok=True)
    
    # Read the file
    with open(file_path, 'rb') as f:
        data = f.read()
    
    file_size = len(data)
    print(f"File size: {file_size} bytes")
    
    # Create results dictionary
    results = {
        "file_name": file_path.name,
        "file_size": file_size,
        "file_hash": {
            "md5": hashlib.md5(data).hexdigest()
        },
        "analysis_timestamp": time.time()
    }
    
    # Check for file signatures
    print("\n=== Checking for File Signatures ===")
    signatures = check_file_signatures(data)
    results["file_signatures"] = signatures
    
    if signatures:
        print(f"Found {len(signatures)} potential file signatures:")
        for sig in signatures:
            print(f"  {sig['type']} ({sig['description']}) at position {sig['position']}")
    else:
        print("No common file signatures found")
    
    # Analyze byte frequency
    print("\n=== Analyzing Byte Frequency ===")
    frequency_analysis = analyze_byte_frequency(data)
    results["byte_frequency"] = frequency_analysis
    
    print(f"Entropy: {frequency_analysis['entropy']:.2f} bits/byte ({frequency_analysis['entropy_percentage']:.1f}% of maximum)")
    print(f"Data type guess: {frequency_analysis['data_type_guess']}")
    print(f"Printable ASCII: {frequency_analysis['printable_ascii_percentage']:.1f}%")
    
    # Find repeating patterns
    print("\n=== Finding Repeating Patterns ===")
    patterns = find_repeating_patterns(data)
    results["repeating_patterns"] = patterns
    
    if patterns:
        print(f"Found {len(patterns)} repeating patterns:")
        for i, pattern in enumerate(patterns[:5]):
            print(f"  Pattern {i+1}: {pattern['pattern_hex']} (length: {pattern['pattern_length']}, occurrences: {pattern['occurrences']})")
    else:
        print("No significant repeating patterns found")
    
    # Try to decode as text
    print("\n=== Trying Text Decodings ===")
    text_samples = try_decode_text(data)
    results["text_decodings"] = text_samples
    
    if text_samples:
        print(f"Possible text encodings detected:")
        for encoding, sample in text_samples.items():
            print(f"  {encoding}: {sample[:50]}...")
    else:
        print("No valid text encodings detected")
    
    # Try common transformations
    print("\n=== Trying Common Transformations ===")
    transformations = try_common_transformations(data)
    results["transformations"] = transformations
    
    # Check for potential steganography
    print("\n=== Checking for Steganography ===")
    stego_analysis = check_for_steganography(data)
    results["steganography_analysis"] = stego_analysis
    
    print(f"LSB distribution: 0s: {stego_analysis['lsb_analysis']['lsb_0_percentage']:.1f}%, 1s: {stego_analysis['lsb_analysis']['lsb_1_percentage']:.1f}%")
    if stego_analysis['lsb_analysis']['potential_lsb_stego']:
        print("LSB distribution is close to 50/50, which could indicate LSB steganography")
    if "lsb_as_ascii_sample" in stego_analysis:
        print(f"LSB as ASCII sample: {stego_analysis['lsb_as_ascii_sample']}")
    
    # Save results to file
    output_file = output_dir / f"{file_path.stem}_analysis.json"
    with open(output_file, 'w') as f:
        # Remove binary data before saving
        for pattern in results.get("repeating_patterns", []):
            if "pattern" in pattern:
                del pattern["pattern"]
                
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis complete. Results saved to {output_file}")


def main() -> None:
    """Main entry point for the binary data analyzer."""
    parser = argparse.ArgumentParser(description='Analyze binary data for patterns and hidden information')
    parser.add_argument('--file', type=str, required=True,
                      help='Path to the binary file to analyze')
    parser.add_argument('--output-dir', type=str, default='results/binary_analysis',
                      help='Directory to store analysis output')
    
    args = parser.parse_args()
    
    file_path = Path(args.file)
    output_dir = Path(args.output_dir)
    
    analyze_binary_file(file_path, output_dir)


if __name__ == "__main__":
    main() 