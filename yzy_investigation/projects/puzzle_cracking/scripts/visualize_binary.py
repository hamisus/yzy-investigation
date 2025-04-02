#!/usr/bin/env python3
"""
Script to visualize binary data in different formats.

This script converts binary data to different image formats
to help identify hidden patterns or images.
"""

import argparse
import math
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional


def visualize_as_bitmap(data: bytes, output_file: Path, width: Optional[int] = None) -> None:
    """
    Visualize binary data as a 1-bit bitmap.
    
    Args:
        data: Binary data to visualize
        output_file: Path to save the output image
        width: Optional width of the image (auto-calculated if None)
    """
    # Calculate dimensions
    total_pixels = len(data) * 8  # Each byte has 8 bits
    if width is None:
        width = int(math.sqrt(total_pixels))
    
    height = total_pixels // width
    if total_pixels % width != 0:
        height += 1
    
    # Create a 1-bit image
    img = Image.new('1', (width, height), 0)
    pixels = img.load()
    
    # Fill in pixels based on bit values
    for byte_idx, byte in enumerate(data):
        for bit_idx in range(8):
            bit_value = (byte >> bit_idx) & 1
            pixel_idx = byte_idx * 8 + bit_idx
            x = pixel_idx % width
            y = pixel_idx // width
            
            if y < height:
                pixels[x, y] = bit_value
    
    # Save the image
    img.save(output_file)
    print(f"Saved 1-bit bitmap to {output_file}")


def visualize_as_grayscale(data: bytes, output_file: Path, width: Optional[int] = None) -> None:
    """
    Visualize binary data as a grayscale image.
    
    Args:
        data: Binary data to visualize
        output_file: Path to save the output image
        width: Optional width of the image (auto-calculated if None)
    """
    # Calculate dimensions
    total_pixels = len(data)
    if width is None:
        width = int(math.sqrt(total_pixels))
    
    height = total_pixels // width
    if total_pixels % width != 0:
        height += 1
    
    # Create a grayscale image
    img = Image.new('L', (width, height), 0)
    pixels = img.load()
    
    # Fill in pixels based on byte values
    for idx, byte in enumerate(data):
        x = idx % width
        y = idx // width
        
        if y < height:
            pixels[x, y] = byte
    
    # Save the image
    img.save(output_file)
    print(f"Saved grayscale image to {output_file}")


def visualize_as_rgb(data: bytes, output_file: Path, width: Optional[int] = None) -> None:
    """
    Visualize binary data as an RGB image.
    
    Args:
        data: Binary data to visualize
        output_file: Path to save the output image
        width: Optional width of the image (auto-calculated if None)
    """
    # Calculate dimensions
    total_pixels = len(data) // 3  # Each pixel needs 3 bytes (RGB)
    if width is None:
        width = int(math.sqrt(total_pixels))
    
    height = total_pixels // width
    if total_pixels % width != 0:
        height += 1
    
    # Create an RGB image
    img = Image.new('RGB', (width, height), (0, 0, 0))
    pixels = img.load()
    
    # Fill in pixels based on byte values
    for i in range(0, len(data) - 2, 3):
        pixel_idx = i // 3
        x = pixel_idx % width
        y = pixel_idx // width
        
        if y < height:
            r = data[i] if i < len(data) else 0
            g = data[i+1] if i+1 < len(data) else 0
            b = data[i+2] if i+2 < len(data) else 0
            pixels[x, y] = (r, g, b)
    
    # Save the image
    img.save(output_file)
    print(f"Saved RGB image to {output_file}")


def try_common_dimensions(data: bytes, output_dir: Path) -> None:
    """
    Try visualizing binary data with several common image dimensions.
    
    Args:
        data: Binary data to visualize
        output_dir: Directory to save output images
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Define common image dimensions to try
    common_widths = [
        16, 32, 64, 128, 256, 512, 1024,  # Power of 2 widths
        20, 40, 60, 80, 100, 320, 640, 800,  # Common screen widths
        24, 36, 48, 72, 96,  # Multiples of 12
        25, 50, 75, 100, 125, 150, 175, 200,  # Multiples of 25
        240, 320, 360, 480, 960, 1280  # Common video widths
    ]
    
    # Try each width for different image types
    for width in common_widths:
        # 1-bit bitmap
        bitmap_file = output_dir / f"binary_bitmap_w{width}.png"
        visualize_as_bitmap(data, bitmap_file, width)
        
        # Grayscale
        gray_file = output_dir / f"grayscale_w{width}.png"
        visualize_as_grayscale(data, gray_file, width)
        
        # RGB
        rgb_file = output_dir / f"rgb_w{width}.png"
        visualize_as_rgb(data, rgb_file, width)


def try_divisor_dimensions(data: bytes, output_dir: Path, mode: str = 'grayscale') -> None:
    """
    Try visualizing binary data with dimensions that divide evenly into the data length.
    
    Args:
        data: Binary data to visualize
        output_dir: Directory to save output images
        mode: Image mode ('bitmap', 'grayscale', or 'rgb')
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Calculate reasonable data size based on mode
    if mode == 'bitmap':
        data_size = len(data) * 8
    elif mode == 'rgb':
        data_size = len(data) // 3
    else:  # grayscale
        data_size = len(data)
    
    # Find divisors of data size for potential widths
    divisors = []
    for i in range(1, int(math.sqrt(data_size)) + 1):
        if data_size % i == 0:
            divisors.append(i)
            if i != data_size // i:
                divisors.append(data_size // i)
    
    # Sort divisors
    divisors.sort()
    
    # Keep only reasonable divisors (not too small or too large)
    reasonable_divisors = [d for d in divisors if 10 <= d <= 2000]
    
    # Try each width for the selected image type
    for width in reasonable_divisors:
        if mode == 'bitmap':
            output_file = output_dir / f"binary_bitmap_exact_w{width}.png"
            visualize_as_bitmap(data, output_file, width)
        elif mode == 'rgb':
            output_file = output_dir / f"rgb_exact_w{width}.png"
            visualize_as_rgb(data, output_file, width)
        else:  # grayscale
            output_file = output_dir / f"grayscale_exact_w{width}.png"
            visualize_as_grayscale(data, output_file, width)


def visualize_as_lsb_bitmap(data: bytes, output_file: Path, width: Optional[int] = None) -> None:
    """
    Visualize just the LSB (least significant bit) of each byte as a bitmap.
    
    Args:
        data: Binary data to visualize
        output_file: Path to save the output image
        width: Optional width of the image (auto-calculated if None)
    """
    # Calculate dimensions
    total_pixels = len(data)  # One pixel per byte (just the LSB)
    if width is None:
        width = int(math.sqrt(total_pixels))
    
    height = total_pixels // width
    if total_pixels % width != 0:
        height += 1
    
    # Create a 1-bit image
    img = Image.new('1', (width, height), 0)
    pixels = img.load()
    
    # Fill in pixels based on LSB values
    for idx, byte in enumerate(data):
        x = idx % width
        y = idx // width
        
        if y < height:
            pixels[x, y] = byte & 1
    
    # Save the image
    img.save(output_file)
    print(f"Saved LSB bitmap to {output_file}")


def visualize_as_multi_bit_planes(data: bytes, output_dir: Path, width: Optional[int] = None) -> None:
    """
    Visualize each bit plane (0-7) of the data as separate bitmaps.
    
    Args:
        data: Binary data to visualize
        output_dir: Directory to save output images
        width: Optional width of the image (auto-calculated if None)
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Calculate dimensions
    total_pixels = len(data)
    if width is None:
        width = int(math.sqrt(total_pixels))
    
    height = total_pixels // width
    if total_pixels % width != 0:
        height += 1
    
    # Create a bitmap for each bit plane
    for bit in range(8):
        img = Image.new('1', (width, height), 0)
        pixels = img.load()
        
        # Fill in pixels based on the value of the selected bit
        for idx, byte in enumerate(data):
            x = idx % width
            y = idx // width
            
            if y < height:
                pixels[x, y] = (byte >> bit) & 1
        
        # Save the image
        output_file = output_dir / f"bit_plane_{bit}.png"
        img.save(output_file)
        print(f"Saved bit plane {bit} to {output_file}")


def main() -> None:
    """Main entry point for the binary data visualizer."""
    parser = argparse.ArgumentParser(description='Visualize binary data as images')
    parser.add_argument('--file', type=str, required=True,
                      help='Path to the binary file to visualize')
    parser.add_argument('--output-dir', type=str, default='results/binary_visualized',
                      help='Directory to store visualizations')
    parser.add_argument('--width', type=int, default=None,
                      help='Width for the output images (auto-calculated if not specified)')
    parser.add_argument('--mode', type=str, choices=['all', 'bitmap', 'grayscale', 'rgb', 'lsb', 'bitplanes', 'divisors'],
                      default='all',
                      help='Visualization mode')
    
    args = parser.parse_args()
    
    # Parse arguments
    file_path = Path(args.file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Read the file
    with open(file_path, 'rb') as f:
        data = f.read()
    
    print(f"Visualizing binary file: {file_path} ({len(data)} bytes)")
    
    # Perform visualizations based on mode
    if args.mode == 'bitmap' or args.mode == 'all':
        bitmap_file = output_dir / f"{file_path.stem}_bitmap.png"
        visualize_as_bitmap(data, bitmap_file, args.width)
    
    if args.mode == 'grayscale' or args.mode == 'all':
        grayscale_file = output_dir / f"{file_path.stem}_grayscale.png"
        visualize_as_grayscale(data, grayscale_file, args.width)
    
    if args.mode == 'rgb' or args.mode == 'all':
        rgb_file = output_dir / f"{file_path.stem}_rgb.png"
        visualize_as_rgb(data, rgb_file, args.width)
    
    if args.mode == 'lsb' or args.mode == 'all':
        lsb_file = output_dir / f"{file_path.stem}_lsb.png"
        visualize_as_lsb_bitmap(data, lsb_file, args.width)
    
    if args.mode == 'bitplanes' or args.mode == 'all':
        bitplane_dir = output_dir / f"{file_path.stem}_bitplanes"
        visualize_as_multi_bit_planes(data, bitplane_dir, args.width)
    
    if args.mode == 'divisors':
        divisors_dir = output_dir / f"{file_path.stem}_divisors"
        try_divisor_dimensions(data, divisors_dir, 'grayscale')
    
    if args.mode == 'all':
        # Try common dimensions as a bonus
        common_dir = output_dir / f"{file_path.stem}_common_dims"
        try_common_dimensions(data, common_dir)
    
    print(f"Visualization complete. Images saved to {output_dir}")


if __name__ == "__main__":
    main() 