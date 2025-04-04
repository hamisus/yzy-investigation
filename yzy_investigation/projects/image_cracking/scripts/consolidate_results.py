"""
Script to consolidate all results from the results directory into a single text file.

This script walks through the results directory, reading all files (JSON, text, logs)
and combines them into a single, well-formatted text file for easy review.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import base64

def setup_logging() -> logging.Logger:
    """
    Set up logging configuration.
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger("consolidate_results")
    logger.setLevel(logging.INFO)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

def format_json_for_text(data: Any, indent: int = 0) -> str:
    """
    Format JSON data as readable text with proper indentation.
    
    Args:
        data: JSON data to format
        indent: Current indentation level
        
    Returns:
        Formatted text representation
    """
    indent_str = "    " * indent
    
    if isinstance(data, dict):
        lines = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{indent_str}{key}:")
                lines.append(format_json_for_text(value, indent + 1))
            else:
                lines.append(f"{indent_str}{key}: {value}")
        return "\n".join(lines)
    elif isinstance(data, list):
        lines = []
        for item in data:
            if isinstance(item, (dict, list)):
                lines.append(format_json_for_text(item, indent))
            else:
                lines.append(f"{indent_str}- {item}")
        return "\n".join(lines)
    else:
        return f"{indent_str}{data}"

def process_json_file(file_path: Path) -> str:
    """
    Process a JSON file and return its contents as formatted text.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Formatted text representation of JSON contents
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return format_json_for_text(data)
    except json.JSONDecodeError:
        return f"ERROR: Could not parse JSON file {file_path}"

def process_binary_file(file_path: Path) -> str:
    """
    Process a binary file and return a summary.
    
    Args:
        file_path: Path to binary file
        
    Returns:
        Summary of binary file contents
    """
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        size = len(data)
        preview = base64.b64encode(data[:100]).decode('utf-8')
        return f"Binary file size: {size} bytes\nFirst 100 bytes (base64): {preview}..."
    except Exception as e:
        return f"ERROR: Could not read binary file {file_path}: {e}"

def consolidate_results(results_dir: Path, output_file: Path) -> None:
    """
    Consolidate all results into a single text file.
    
    Args:
        results_dir: Path to results directory
        output_file: Path to output text file
    """
    logger = setup_logging()
    logger.info(f"Starting consolidation of results from {results_dir}")
    
    with open(output_file, 'w') as out:
        # Write header
        out.write("=" * 80 + "\n")
        out.write("CONSOLIDATED RESULTS REPORT\n")
        out.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write("=" * 80 + "\n\n")
        
        # Process all files recursively
        for file_path in sorted(results_dir.rglob("*")):
            if file_path.is_file():
                # Write file header
                rel_path = file_path.relative_to(results_dir)
                out.write("-" * 80 + "\n")
                out.write(f"FILE: {rel_path}\n")
                out.write(f"Size: {file_path.stat().st_size} bytes\n")
                out.write(f"Last modified: {datetime.fromtimestamp(file_path.stat().st_mtime)}\n")
                out.write("-" * 80 + "\n\n")
                
                # Process file based on type
                try:
                    if file_path.suffix.lower() == '.json':
                        content = process_json_file(file_path)
                    elif file_path.suffix.lower() in ['.txt', '.log']:
                        with open(file_path, 'r') as f:
                            content = f.read()
                    elif file_path.suffix.lower() in ['.bin', '.dat']:
                        content = process_binary_file(file_path)
                    else:
                        content = f"Skipped: Unsupported file type {file_path.suffix}"
                    
                    out.write(content + "\n\n")
                except Exception as e:
                    out.write(f"ERROR processing file: {e}\n\n")
                    logger.error(f"Error processing {file_path}: {e}")
        
        # Write footer
        out.write("=" * 80 + "\n")
        out.write("END OF REPORT\n")
        out.write("=" * 80 + "\n")
    
    logger.info(f"Results consolidated into {output_file}")

def main() -> None:
    """Main entry point for the script."""
    # Set up paths
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent.parent.parent.parent / "results"
    
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"consolidated_results_{timestamp}.txt"
    
    # Run consolidation
    consolidate_results(results_dir, output_file)

if __name__ == "__main__":
    main() 