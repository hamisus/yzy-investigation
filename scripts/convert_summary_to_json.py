#!/usr/bin/env python
"""
Script to convert markdown Discord summaries to JSON format.

This script takes a markdown summary file and converts it to the JSON format
expected by the Discord Summary Publisher.
"""

import re
import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

def parse_markdown_summary(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse a markdown summary file and extract sections into JSON format.
    
    Args:
        file_path: Path to the markdown summary file
        
    Returns:
        List of topic dictionaries in JSON format
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all sections that start with "- **" and end with the next "- **" or end of file
    topics = []
    
    # Use regex to find all section headers with bold titles
    section_pattern = r'- \*\*(.*?)\*\*:\s*((?:.|\n)*?)(?=\n-\s+\*\*|\Z)'
    matches = re.findall(section_pattern, content, re.DOTALL)
    
    for title, body in matches:
        # Extract sources from the body
        sources = []
        source_pattern = r'\[Source\]\((https://discord\.com/channels/\d+/\d+/\d+)\)'
        source_matches = re.findall(source_pattern, body)
        sources.extend(source_matches)
        
        # Check if the body contains bullet points
        bullet_pattern = r'^\s*-\s+(.*?)(?:\n|$)'
        bullet_matches = re.findall(bullet_pattern, body, re.MULTILINE)
        
        topic = {
            "topic": title.strip(),
            "sources": sources
        }
        
        if bullet_matches:
            # If bullet points are found, use them as details
            details = []
            for bullet in bullet_matches:
                # Clean bullet text and remove any source links
                clean_bullet = re.sub(r'\[Source\]\(.*?\)', '', bullet)
                clean_bullet = clean_bullet.strip()
                if clean_bullet:  # Only add non-empty bullets
                    details.append(clean_bullet)
            
            topic["details"] = details
        else:
            # Even with no bullet points, convert the whole body to a single detail
            # Remove the sources from the text
            description = re.sub(r'\[Source\]\(.*?\)', '', body)
            # Clean up extra whitespace and newlines
            description = re.sub(r'\n\s+', ' ', description)
            description = re.sub(r'\s+', ' ', description)
            description = description.strip()
            
            # Add as a single item in the details array
            topic["details"] = [description]
        
        topics.append(topic)
    
    return topics

def save_json(topics: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save topics as JSON to the specified path.
    
    Args:
        topics: List of topic dictionaries
        output_path: Path to save the JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(topics, f, indent=2)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Convert markdown Discord summaries to JSON format")
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input markdown summary file"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        help="Path to save the output JSON file. If not specified, will use the input filename with .json extension"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    
    # Verify input file exists
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        # Replace .md with .json, or append .json if no extension
        if input_path.suffix.lower() == '.md':
            output_path = input_path.with_suffix('.json')
        else:
            output_path = Path(str(input_path) + '.json')
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert the file
    try:
        print(f"Converting {input_path} to JSON format...")
        topics = parse_markdown_summary(str(input_path))
        
        if not topics:
            print("Warning: No topics found in the input file.")
        else:
            print(f"Found {len(topics)} topics.")
        
        save_json(topics, str(output_path))
        print(f"Saved JSON to {output_path}")
    except Exception as e:
        print(f"Error converting file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 