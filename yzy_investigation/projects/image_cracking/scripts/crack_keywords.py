#!/usr/bin/env python3
"""
Script to search for specific keywords in extracted steganography data 
and try advanced decoding methods including Base58 and XOR operations.

This script focuses on finding potential connections to:
- "4NBT", "4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump"
- "YZY", "Silver"
- 333, 353
"""

import os
import sys
import json
import base64
import binascii
import re
import hashlib
from pathlib import Path
import numpy as np
import argparse
from typing import Dict, List, Tuple, Any, Optional, Set
import matplotlib.pyplot as plt


# Add project root to Python path if running script directly
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(project_root))


# Key terms to search for
KEY_TERMS = [
    "4NBT",
    "4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump",
    "YZY",
    "Silver",
    "333",
    "353"
]


def load_binary_data(file_path: Path) -> bytes:
    """
    Load binary data from a file.
    
    Args:
        file_path: Path to the binary file
        
    Returns:
        Binary data as bytes
    """
    with open(file_path, 'rb') as f:
        return f.read()


def search_binary_for_terms(data: bytes, output_file: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Search binary data for key terms.
    
    Args:
        data: Binary data to search
        output_file: Optional file to save results
        
    Returns:
        List of matches found
    """
    matches = []
    
    # Convert to various formats for searching
    hex_data = binascii.hexlify(data).decode('ascii')
    try:
        ascii_data = data.decode('ascii', errors='ignore')
    except:
        ascii_data = ""
    
    try:
        utf8_data = data.decode('utf-8', errors='ignore')
    except:
        utf8_data = ""
        
    # Search in different formats
    for term in KEY_TERMS:
        # Direct binary search
        term_positions = []
        pos = 0
        term_bytes = term.encode('utf-8')
        while True:
            pos = data.find(term_bytes, pos)
            if pos == -1:
                break
            term_positions.append(pos)
            pos += 1
            
        if term_positions:
            matches.append({
                "term": term,
                "encoding": "binary",
                "positions": term_positions,
                "count": len(term_positions)
            })
            
        # ASCII search
        term_positions = []
        for m in re.finditer(re.escape(term), ascii_data):
            term_positions.append(m.start())
            
        if term_positions:
            matches.append({
                "term": term,
                "encoding": "ascii",
                "positions": term_positions,
                "count": len(term_positions)
            })
            
        # UTF-8 search
        if utf8_data != ascii_data:
            term_positions = []
            for m in re.finditer(re.escape(term), utf8_data):
                term_positions.append(m.start())
                
            if term_positions:
                matches.append({
                    "term": term,
                    "encoding": "utf-8",
                    "positions": term_positions,
                    "count": len(term_positions)
                })
                
        # Hex search
        term_hex = binascii.hexlify(term.encode('utf-8')).decode('ascii')
        term_positions = []
        for m in re.finditer(re.escape(term_hex), hex_data):
            term_positions.append(m.start() // 2)  # Convert hex position to byte position
            
        if term_positions:
            matches.append({
                "term": term,
                "encoding": "hex",
                "positions": term_positions,
                "count": len(term_positions)
            })
    
    # Save results if output file specified
    if output_file and matches:
        with open(output_file, 'w') as f:
            json.dump(matches, f, indent=2)
    
    return matches


def try_base58_decoding(data: bytes) -> Optional[bytes]:
    """
    Try to decode data as Base58.
    
    Args:
        data: Binary data to decode
        
    Returns:
        Decoded data or None if decoding fails
    """
    # Base58 alphabet (Bitcoin style)
    alphabet = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    
    try:
        # Convert data to ASCII if it's not already a string
        if isinstance(data, bytes):
            data_str = data.decode('ascii', errors='ignore')
        else:
            data_str = data
            
        # Check if the string uses only Base58 alphabet characters
        if all(c in alphabet for c in data_str):
            # Decode Base58
            result = 0
            for c in data_str:
                result = result * 58 + alphabet.index(c)
                
            # Convert to bytes
            result_bytes = result.to_bytes((result.bit_length() + 7) // 8, byteorder='big')
            return result_bytes
    except Exception as e:
        print(f"Base58 decoding error: {e}")
    
    return None


def try_key_xor(data: bytes, key_terms: List[str]) -> List[Dict[str, Any]]:
    """
    Try XORing data with key terms.
    
    Args:
        data: Binary data to XOR
        key_terms: List of key terms to use as XOR keys
        
    Returns:
        List of interesting XOR results
    """
    results = []
    
    for term in key_terms:
        # Convert term to bytes
        key_bytes = term.encode('utf-8')
        
        # XOR with key repeating as needed
        result = bytearray()
        for i in range(len(data)):
            result.append(data[i] ^ key_bytes[i % len(key_bytes)])
        
        # Check if result contains interesting patterns
        printable_ratio = sum(32 <= b <= 126 for b in result) / len(result)
        if printable_ratio > 0.7:  # If mostly printable ASCII
            sample = result[:100].decode('ascii', errors='replace')
            results.append({
                "key": term,
                "printable_ratio": printable_ratio,
                "sample": sample
            })
            
            # Look for known terms in the result
            match_found = False
            result_str = result.decode('ascii', errors='ignore')
            for search_term in KEY_TERMS:
                if search_term in result_str:
                    match_found = True
                    results[-1]["contains_term"] = search_term
                    results[-1]["term_position"] = result_str.index(search_term)
                    break
                    
            if match_found:
                # Save the full result if it contains a key term
                results[-1]["full_result"] = result_str
    
    return results


def try_number_conversions(data: bytes, numbers: List[int]) -> List[Dict[str, Any]]:
    """
    Try various conversions and operations with specific numbers.
    
    Args:
        data: Binary data to process
        numbers: List of significant numbers to try
        
    Returns:
        List of interesting results
    """
    results = []
    
    # Try using numbers as offsets or lengths
    for num in numbers:
        if num < len(data):
            # Try extracting a chunk of data
            chunk = data[num:num+100]
            results.append({
                "operation": f"Extract at offset {num}",
                "sample": chunk[:50].hex(),
                "ascii_sample": chunk.decode('ascii', errors='replace')[:50]
            })
            
            # Check if this chunk contains key terms
            for term in KEY_TERMS:
                term_bytes = term.encode('utf-8')
                if term_bytes in chunk:
                    results[-1]["contains_term"] = term
                    break
                
    # Try XOR with each number (as a single byte)
    for num in numbers:
        result = bytearray(b ^ (num & 0xFF) for b in data)
        
        # Check if result contains interesting patterns
        printable_ratio = sum(32 <= b <= 126 for b in result) / len(result)
        if printable_ratio > 0.5:  # If reasonably printable
            sample = result[:100].decode('ascii', errors='replace')
            results.append({
                "operation": f"XOR with {num}",
                "printable_ratio": printable_ratio,
                "sample": sample
            })
            
            # Look for known terms in the result
            result_str = result.decode('ascii', errors='ignore')
            for term in KEY_TERMS:
                if term in result_str:
                    results[-1]["contains_term"] = term
                    results[-1]["term_position"] = result_str.index(term)
                    break
    
    return results


def check_bit_pattern_relations(pattern: List[int], numbers: List[int]) -> List[Dict[str, Any]]:
    """
    Check for relations between the bit pattern and significant numbers.
    
    Args:
        pattern: Binary pattern (list of 0s and 1s)
        numbers: List of significant numbers
        
    Returns:
        List of potential relations found
    """
    results = []
    
    # Check if any number matches the count of 1s or 0s
    ones_count = sum(pattern)
    zeros_count = len(pattern) - ones_count
    
    for num in numbers:
        if num == ones_count:
            results.append({
                "relation": "count_match",
                "description": f"Number of 1s in pattern ({ones_count}) matches key number {num}"
            })
        if num == zeros_count:
            results.append({
                "relation": "count_match",
                "description": f"Number of 0s in pattern ({zeros_count}) matches key number {num}"
            })
            
    # Check for patterns at specific positions
    for num in numbers:
        if num < len(pattern):
            results.append({
                "relation": "position_value",
                "description": f"Value at position {num} is {pattern[num]}"
            })
            
    # Try arranging the pattern in matrices of dimensions related to the numbers
    for num in numbers:
        if len(pattern) % num == 0:
            # Can arrange in a matrix with num columns
            rows = len(pattern) // num
            results.append({
                "relation": "matrix_dimension",
                "description": f"Pattern can be arranged in a {rows}x{num} matrix"
            })
        
        if num < len(pattern) and len(pattern) % num == 0:
            # Can arrange in a matrix with num rows
            cols = len(pattern) // num
            results.append({
                "relation": "matrix_dimension",
                "description": f"Pattern can be arranged in a {num}x{cols} matrix"
            })
    
    return results


def analyze_permutations(data: bytes, pattern: List[int], numbers: List[int]) -> Dict[str, Any]:
    """
    Analyze data using various permutations based on the significant numbers.
    
    Args:
        data: Binary data to analyze
        pattern: Binary pattern
        numbers: Significant numbers
        
    Returns:
        Dictionary with analysis results
    """
    results = {}
    
    # Try cyclic permutations
    for num in numbers:
        # Shift pattern by num positions
        shifted_pattern = pattern[num % len(pattern):] + pattern[:num % len(pattern)]
        results[f"shift_pattern_{num}"] = {
            "pattern": shifted_pattern,
            "pattern_str": ''.join(str(b) for b in shifted_pattern)
        }
        
        # Shift data by num positions
        if len(data) > num:
            shifted_data = data[num:] + data[:num]
            ascii_result = shifted_data[:100].decode('ascii', errors='replace')
            printable_ratio = sum(32 <= b <= 126 for b in shifted_data[:100]) / min(100, len(shifted_data))
            
            results[f"shift_data_{num}"] = {
                "sample": shifted_data[:50].hex(),
                "ascii_sample": ascii_result,
                "printable_ratio": printable_ratio
            }
            
            # Check for key terms in shifted data
            for term in KEY_TERMS:
                if term in ascii_result:
                    results[f"shift_data_{num}"]["contains_term"] = term
                    break
    
    # Try special arrangements based on numbers
    for num in numbers:
        if len(pattern) % num == 0:
            # Reshape pattern into matrix and try various readings
            cols = num
            rows = len(pattern) // num
            
            matrix = np.array(pattern).reshape(rows, cols)
            
            # Read by columns
            col_pattern = matrix.T.flatten()
            col_pattern_str = ''.join(str(int(b)) for b in col_pattern)
            
            results[f"matrix_{rows}x{cols}_col_read"] = {
                "pattern": col_pattern.tolist(),
                "pattern_str": col_pattern_str
            }
            
            # Diagonal reading (if square)
            if rows == cols:
                diag = np.diag(matrix)
                diag_pattern_str = ''.join(str(int(b)) for b in diag)
                
                results[f"matrix_{rows}x{cols}_diag"] = {
                    "pattern": diag.tolist(),
                    "pattern_str": diag_pattern_str
                }
    
    return results


def try_decoding_as_coordinates(pattern: List[int], numbers: List[int]) -> Dict[str, Any]:
    """
    Try interpreting the pattern as coordinates based on the significant numbers.
    
    Args:
        pattern: Binary pattern
        numbers: Significant numbers
        
    Returns:
        Dictionary with coordinate analysis results
    """
    results = {}
    
    # Get indices of 1s (these might represent coordinates)
    one_indices = [i for i, bit in enumerate(pattern) if bit == 1]
    
    results["one_indices"] = one_indices
    
    # Check if indices have any relation to the key numbers
    for num in numbers:
        if num in one_indices:
            results[f"found_number_{num}"] = "Number appears as index of a 1 in the pattern"
            
        # Check for modulo relationships
        mod_indices = [i % num for i in one_indices]
        if len(set(mod_indices)) < len(mod_indices) // 2:  # If there are repeats
            results[f"modulo_{num}_pattern"] = {
                "mod_indices": mod_indices,
                "unique_remainders": list(set(mod_indices))
            }
    
    # Try to interpret as (x,y) coordinates
    # For example, 6x10 grid would use positions % 10 as x and positions // 10 as y
    for num in numbers:
        if num > 1 and len(pattern) % num == 0:
            width = num
            height = len(pattern) // num
            
            # Convert 1 indices to (x,y) coordinates
            coords = [(i % width, i // width) for i in one_indices]
            
            results[f"coordinates_grid_{height}x{width}"] = coords
            
    return results


def explore_special_string(string: str, data: bytes, pattern: List[int]) -> Dict[str, Any]:
    """
    Specifically explore the special string "4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump".
    
    Args:
        string: The special string to analyze
        data: Binary data to check against
        pattern: Binary pattern to check against
        
    Returns:
        Dictionary with analysis results
    """
    results = {}
    
    # Check if string is present in data
    string_bytes = string.encode('utf-8')
    if string_bytes in data:
        results["found_in_data"] = True
        results["position"] = data.index(string_bytes)
    else:
        results["found_in_data"] = False
        
    # Try Base58 decoding (common in cryptocurrency keys)
    try:
        # This is approximate Base58 decoding, may need adjustment
        alphabet = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
        decoded = 0
        for char in string:
            decoded = decoded * 58 + alphabet.index(char)
        
        # Convert to bytes and hex
        hex_result = hex(decoded)[2:]
        if len(hex_result) % 2 == 1:
            hex_result = '0' + hex_result
            
        byte_result = bytes.fromhex(hex_result)
        
        results["base58_decoding"] = {
            "decimal": decoded,
            "hex": hex_result,
            "bytes": byte_result.hex(),
            "ascii": byte_result.decode('ascii', errors='replace')
        }
    except Exception as e:
        results["base58_decoding_error"] = str(e)
        
    # Try as hash input
    for algorithm in ['md5', 'sha1', 'sha256']:
        h = hashlib.new(algorithm)
        h.update(string_bytes)
        digest = h.hexdigest()
        
        # Check if hash appears in binary data
        hash_bytes = bytes.fromhex(digest)
        if hash_bytes in data:
            results[f"{algorithm}_hash_found"] = True
            results[f"{algorithm}_hash_position"] = data.index(hash_bytes)
        
        results[f"{algorithm}_hash"] = digest
        
    # Try mapping to binary pattern
    if len(string) >= len(pattern):
        # Map chars to bit positions
        for i, bit in enumerate(pattern):
            if bit == 1 and i < len(string):
                if "string_chars_at_bit_positions" not in results:
                    results["string_chars_at_bit_positions"] = ""
                results["string_chars_at_bit_positions"] += string[i]
                
    # XOR with pattern
    if len(pattern) > 0:
        xor_result = ""
        for i, char in enumerate(string):
            bit_pos = i % len(pattern)
            if pattern[bit_pos] == 1:
                xor_result += char
                
        results["xor_with_pattern"] = xor_result
    
    return results


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Search for keywords in steganography data')
    parser.add_argument('--data-dir', type=str, default='results/stego_explored/decoded_data',
                      help='Directory containing extracted binary data')
    parser.add_argument('--pattern-file', type=str, default='results/stego_pattern/extracted_pattern.json',
                      help='File containing extracted binary pattern')
    parser.add_argument('--output-dir', type=str, default='results/keyword_analysis',
                      help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    pattern_file = Path(args.pattern_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load pattern
    pattern_data = None
    binary_pattern = []
    if pattern_file.exists():
        with open(pattern_file, 'r') as f:
            pattern_data = json.load(f)
            binary_pattern = pattern_data["binary_pattern"]
            
    print(f"Loaded pattern with {len(binary_pattern)} bits")
    
    # Process all binary files in data directory
    binary_files = list(data_dir.glob("*.bin"))
    print(f"Found {len(binary_files)} binary files to analyze")
    
    # Numbers of significance
    significant_numbers = [333, 353, 42, 66]
    
    all_results = {}
    
    for file_path in binary_files:
        print(f"\nAnalyzing file: {file_path}")
        
        # Load binary data
        data = load_binary_data(file_path)
        print(f"File size: {len(data)} bytes")
        
        # Search for key terms
        matches = search_binary_for_terms(data, output_dir / f"{file_path.stem}_term_matches.json")
        
        if matches:
            print("Found key terms in data:")
            for match in matches:
                print(f"  {match['term']} ({match['encoding']}): {match['count']} occurrences")
                
        # Try Base58 decoding
        base58_result = try_base58_decoding(data)
        if base58_result:
            print("Base58 decoding succeeded")
            with open(output_dir / f"{file_path.stem}_base58_decoded.bin", 'wb') as f:
                f.write(base58_result)
                
        # Try XOR with key terms
        xor_results = try_key_xor(data, KEY_TERMS)
        if xor_results:
            print("Found interesting XOR results:")
            for result in xor_results:
                print(f"  XOR with '{result['key']}': {result['printable_ratio']:.2f} printable ratio")
                print(f"    Sample: {result['sample'][:50]}")
                if "contains_term" in result:
                    print(f"    Contains term: {result['contains_term']}")
                    
            # Save XOR results
            with open(output_dir / f"{file_path.stem}_xor_results.json", 'w') as f:
                json.dump(xor_results, f, indent=2)
                
        # Try number conversions
        number_results = try_number_conversions(data, significant_numbers)
        if number_results:
            print("Found interesting number-based results:")
            for result in number_results:
                print(f"  {result['operation']}")
                if "contains_term" in result:
                    print(f"    Contains term: {result['contains_term']}")
                    
            # Save number results
            with open(output_dir / f"{file_path.stem}_number_results.json", 'w') as f:
                json.dump(number_results, f, indent=2)
        
        # Save all results for this file
        file_results = {
            "term_matches": matches,
            "xor_results": xor_results,
            "number_results": number_results
        }
        
        all_results[file_path.name] = file_results
    
    # Analyze pattern relations to significant numbers
    if binary_pattern:
        print("\nAnalyzing pattern relations to significant numbers:")
        pattern_relations = check_bit_pattern_relations(binary_pattern, significant_numbers)
        
        for relation in pattern_relations:
            print(f"  {relation['description']}")
            
        # Try pattern permutations
        permutation_results = {}
        
        # Use first binary file for permutation analysis
        if binary_files:
            permutation_results = analyze_permutations(load_binary_data(binary_files[0]), binary_pattern, significant_numbers)
            
        # Save pattern analysis
        with open(output_dir / "pattern_analysis.json", 'w') as f:
            json.dump({
                "relations": pattern_relations,
                "permutations": permutation_results
            }, f, indent=2)
            
        # Try coordinate interpretation
        coordinate_results = try_decoding_as_coordinates(binary_pattern, significant_numbers)
        with open(output_dir / "coordinate_analysis.json", 'w') as f:
            json.dump(coordinate_results, f, indent=2)
            
    # Special analysis of the long string
    special_string = "4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump"
    print(f"\nSpecial analysis of string: {special_string}")
    
    # Check each binary file
    for file_path in binary_files:
        data = load_binary_data(file_path)
        special_results = explore_special_string(special_string, data, binary_pattern)
        
        if special_results.get("found_in_data", False):
            print(f"  String found in {file_path.name} at position {special_results['position']}")
            
        # Hash checks
        for algo in ['md5', 'sha1', 'sha256']:
            if f"{algo}_hash_found" in special_results and special_results[f"{algo}_hash_found"]:
                print(f"  {algo.upper()} hash of string found in {file_path.name}")
                
        # Save special analysis
        with open(output_dir / f"{file_path.stem}_special_string_analysis.json", 'w') as f:
            json.dump(special_results, f, indent=2)
    
    # Save overall results
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\nAnalysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main() 