"""
Command-line interface for steganography analysis.

This module provides a simple CLI for analyzing images for steganography.
"""
import argparse
import json
import sys
from typing import List, Optional
from pathlib import Path
import logging

from .analyzer import detect_steganography
from .image_integration import analyze_with_keywords


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Args:
        args: Command line arguments (defaults to sys.argv[1:])
        
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Steganography analysis tools")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Analyze single image
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a single image")
    analyze_parser.add_argument("image_path", help="Path to the image file")
    analyze_parser.add_argument(
        "--output", "-o", 
        help="Output file for results (default: stdout)"
    )
    analyze_parser.add_argument(
        "--use-keywords",
        action="store_true",
        help="Use keywords from image_cracking project"
    )
    
    # Batch analyze directory
    batch_parser = subparsers.add_parser("batch", help="Analyze multiple images in a directory")
    batch_parser.add_argument("directory", help="Directory containing images")
    batch_parser.add_argument(
        "--output", "-o", 
        help="Output file for results (default: stdout)"
    )
    batch_parser.add_argument(
        "--extensions", "-e",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".gif"],
        help="File extensions to include (default: .jpg .jpeg .png .gif)"
    )
    
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.
    
    Args:
        args: Command line arguments (defaults to sys.argv[1:])
        
    Returns:
        Exit code
    """
    parsed_args = parse_args(args)
    
    if parsed_args.command == "analyze":
        if not Path(parsed_args.image_path).exists():
            print(f"Error: Image file not found: {parsed_args.image_path}", file=sys.stderr)
            return 1
        
        if parsed_args.use_keywords:
            results = analyze_with_keywords(parsed_args.image_path)
        else:
            results = detect_steganography(parsed_args.image_path)
            
        # Format results as JSON
        output = json.dumps(results, indent=2)
        
        # Write to file or stdout
        if parsed_args.output:
            with open(parsed_args.output, 'w') as f:
                f.write(output)
        else:
            print(output)
            
    elif parsed_args.command == "batch":
        if not Path(parsed_args.directory).exists() or not Path(parsed_args.directory).is_dir():
            print(f"Error: Directory not found: {parsed_args.directory}", file=sys.stderr)
            return 1
        
        results = batch_analyze_directory(
            parsed_args.directory,
            [ext if ext.startswith('.') else f'.{ext}' for ext in parsed_args.extensions]
        )
        
        # Format results as JSON
        output = json.dumps(results, indent=2)
        
        # Write to file or stdout
        if parsed_args.output:
            with open(parsed_args.output, 'w') as f:
                f.write(output)
        else:
            print(output)
            
    else:
        print("Error: No command specified. Use 'analyze' or 'batch'.", file=sys.stderr)
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main()) 