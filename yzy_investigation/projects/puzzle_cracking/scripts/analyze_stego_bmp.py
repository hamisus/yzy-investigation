#!/usr/bin/env python3
"""
Script to analyze the BMP image extracted from steganography data.

This script examines the BMP file for:
1. Hidden text or metadata
2. Pixel patterns that might encode information
3. LSB steganography within the BMP itself (nested steganography)
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any


def analyze_bmp_metadata(bmp_path: Path) -> Dict[str, Any]:
    """
    Analyze metadata from a BMP file.
    
    Args:
        bmp_path: Path to the BMP file
        
    Returns:
        Dictionary with metadata information
    """
    results = {}
    
    try:
        with open(bmp_path, "rb") as f:
            # Read BMP header (14 bytes)
            header = f.read(14)
            
            # Verify BMP signature
            if header[0:2] != b'BM':
                print("Not a valid BMP file")
                return {"valid": False}
                
            # Get file size
            file_size = int.from_bytes(header[2:6], byteorder='little')
            results["file_size"] = file_size
            
            # Get data offset
            data_offset = int.from_bytes(header[10:14], byteorder='little')
            results["data_offset"] = data_offset
            
            # Read DIB header
            dib_size = int.from_bytes(f.read(4), byteorder='little')
            results["dib_header_size"] = dib_size
            
            # Return to start of DIB header
            f.seek(14)
            dib_header = f.read(dib_size)
            
            # Image dimensions
            if dib_size >= 16:  # At least BITMAPCOREHEADER
                width = int.from_bytes(dib_header[4:8], byteorder='little')
                height = int.from_bytes(dib_header[8:12], byteorder='little')
                results["width"] = width
                results["height"] = height
            
            # Bits per pixel
            if dib_size >= 14:
                bpp = int.from_bytes(dib_header[14:16], byteorder='little')
                results["bits_per_pixel"] = bpp
            
            # Compression method
            if dib_size >= 20:
                compression = int.from_bytes(dib_header[16:20], byteorder='little')
                compression_methods = {
                    0: "BI_RGB (no compression)",
                    1: "BI_RLE8 (RLE 8-bit)",
                    2: "BI_RLE4 (RLE 4-bit)",
                    3: "BI_BITFIELDS",
                    4: "BI_JPEG",
                    5: "BI_PNG"
                }
                results["compression"] = compression_methods.get(compression, f"Unknown ({compression})")
            
            # Check for extra data between headers and pixel data
            if data_offset > 14 + dib_size:
                extra_bytes = data_offset - (14 + dib_size)
                results["extra_data_size"] = extra_bytes
                
                # Read extra data
                f.seek(14 + dib_size)
                extra_data = f.read(extra_bytes)
                
                # Try to interpret as text
                try:
                    extra_text = extra_data.decode('ascii', errors='ignore')
                    printable = ''.join(c for c in extra_text if 32 <= ord(c) <= 126 or c in '\n\r\t')
                    if len(printable) / len(extra_text) > 0.7:  # If mostly printable
                        results["extra_data_text"] = printable
                except Exception as e:
                    results["extra_data_error"] = str(e)
            
            results["valid"] = True
            return results
                
    except Exception as e:
        print(f"Error analyzing BMP: {e}")
        return {"valid": False, "error": str(e)}


def extract_lsb_from_image(image_path: Path, output_dir: Path, bits: int = 1) -> Optional[Path]:
    """
    Extract least significant bits from an image.
    
    Args:
        image_path: Path to the image file
        output_dir: Directory to save extracted data
        bits: Number of least significant bits to extract (1-4)
        
    Returns:
        Path to extracted data file or None if failed
    """
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Open the image
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Extract LSBs
        if len(img_array.shape) == 3:  # Color image
            height, width, channels = img_array.shape
            # Create a bit array
            bit_array = np.zeros(height * width * channels, dtype=np.uint8)
            
            index = 0
            for h in range(height):
                for w in range(width):
                    for c in range(channels):
                        bit_array[index] = img_array[h, w, c] & ((1 << bits) - 1)
                        index += 1
        else:  # Grayscale image
            height, width = img_array.shape
            bit_array = np.zeros(height * width, dtype=np.uint8)
            
            index = 0
            for h in range(height):
                for w in range(width):
                    bit_array[index] = img_array[h, w] & ((1 << bits) - 1)
                    index += 1
        
        # Convert bits to bytes
        byte_array = bytearray()
        for i in range(0, len(bit_array) - 7, 8):
            byte_val = 0
            for j in range(8):
                if i + j < len(bit_array):
                    byte_val |= bit_array[i + j] << j
            byte_array.append(byte_val)
        
        # Save the extracted data
        output_file = output_dir / f"{image_path.stem}_extracted_lsb_{bits}bit.bin"
        with open(output_file, "wb") as f:
            f.write(byte_array)
            
        print(f"Extracted {len(byte_array)} bytes of LSB data to {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Error extracting LSBs: {e}")
        return None


def visualize_bit_planes(image_path: Path, output_dir: Path) -> None:
    """
    Visualize the bit planes of the image to detect patterns.
    
    Args:
        image_path: Path to the image file
        output_dir: Directory to save visualizations
    """
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Open the image
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Check if color or grayscale
        is_color = len(img_array.shape) == 3 and img_array.shape[2] >= 3
        
        if is_color:
            # Process each color channel
            channels = ["Red", "Green", "Blue"]
            for c, channel_name in enumerate(channels[:3]):  # Max 3 channels (RGB)
                channel = img_array[:, :, c]
                
                plt.figure(figsize=(15, 8))
                plt.suptitle(f"{channel_name} Channel Bit Planes", fontsize=16)
                
                for bit in range(8):
                    # Extract the bit plane
                    bit_plane = (channel >> bit) & 1
                    
                    # Plot
                    plt.subplot(2, 4, bit + 1)
                    plt.imshow(bit_plane, cmap='gray')
                    plt.title(f"Bit {bit} (LSB)" if bit == 0 else f"Bit {bit}")
                    plt.axis('off')
                
                # Save the figure
                output_file = output_dir / f"{image_path.stem}_{channel_name.lower()}_bit_planes.png"
                plt.savefig(output_file)
                plt.close()
                print(f"Saved {channel_name} bit planes to {output_file}")
        else:
            # Grayscale image
            plt.figure(figsize=(15, 8))
            plt.suptitle("Grayscale Bit Planes", fontsize=16)
            
            for bit in range(8):
                # Extract the bit plane
                bit_plane = (img_array >> bit) & 1
                
                # Plot
                plt.subplot(2, 4, bit + 1)
                plt.imshow(bit_plane, cmap='gray')
                plt.title(f"Bit {bit} (LSB)" if bit == 0 else f"Bit {bit}")
                plt.axis('off')
            
            # Save the figure
            output_file = output_dir / f"{image_path.stem}_grayscale_bit_planes.png"
            plt.savefig(output_file)
            plt.close()
            print(f"Saved grayscale bit planes to {output_file}")
            
    except Exception as e:
        print(f"Error visualizing bit planes: {e}")


def check_hidden_text(image_path: Path) -> None:
    """
    Check for hidden text in an image using various techniques.
    
    Args:
        image_path: Path to the image file
    """
    try:
        # Open the image
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Check LSB of each channel for text patterns
        if len(img_array.shape) == 3:  # Color image
            height, width, channels = img_array.shape
            for c in range(min(3, channels)):  # Check RGB channels
                channel = img_array[:, :, c]
                lsb = channel & 1  # Extract LSB
                
                # Convert to string of bits
                bit_string = ''.join(str(b) for row in lsb for b in row)
                
                # Try to decode text (8 bits per character)
                try_decode_text(bit_string, f"Channel {c} LSB")
        else:
            # Grayscale image
            lsb = img_array & 1
            bit_string = ''.join(str(b) for row in lsb for b in row)
            try_decode_text(bit_string, "Grayscale LSB")
            
    except Exception as e:
        print(f"Error checking for hidden text: {e}")


def try_decode_text(bit_string: str, source: str) -> None:
    """
    Try to decode a bit string as ASCII text.
    
    Args:
        bit_string: String of '0' and '1' characters
        source: Description of the source for reporting
    """
    chars = []
    for i in range(0, len(bit_string) - 7, 8):
        byte_str = bit_string[i:i+8]
        byte = int(byte_str, 2)
        chars.append(chr(byte) if 32 <= byte <= 126 else '.')
    
    # Print the first 100 characters
    text = ''.join(chars[:100])
    printable_ratio = sum(1 for c in text if c != '.') / len(text)
    
    if printable_ratio > 0.5:  # If more than 50% printable
        print(f"\nPossible hidden text found in {source}:")
        print(f"First 100 chars: {text}")
        print(f"Printable ratio: {printable_ratio:.2f}")


def main() -> None:
    """Main entry point for the BMP analyzer."""
    parser = argparse.ArgumentParser(description='Analyze a BMP file extracted from steganography data')
    parser.add_argument('--bmp-file', type=str, required=True,
                      help='Path to the BMP file to analyze')
    parser.add_argument('--output-dir', type=str, default='results/stego_bmp_analysis',
                      help='Directory to store analysis output')
    
    args = parser.parse_args()
    
    bmp_path = Path(args.bmp_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Analyzing BMP file: {bmp_path}")
    
    # Analyze BMP metadata
    metadata = analyze_bmp_metadata(bmp_path)
    print("\n=== BMP Metadata ===")
    for key, value in metadata.items():
        print(f"{key}: {value}")
    
    # Save metadata to file
    with open(output_dir / "bmp_metadata.json", "w") as f:
        import json
        json.dump(metadata, f, indent=2)
    
    # Check for hidden text
    print("\n=== Checking for Hidden Text ===")
    check_hidden_text(bmp_path)
    
    # Extract LSB data (try different bit depths)
    print("\n=== Extracting LSB Data ===")
    for bits in [1, 2]:
        extract_lsb_from_image(bmp_path, output_dir, bits)
    
    # Visualize bit planes
    print("\n=== Visualizing Bit Planes ===")
    visualize_bit_planes(bmp_path, output_dir)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main() 