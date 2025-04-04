"""
Integration module for connecting stego_analysis with image_cracking project.

This module provides utilities to use keywords from the image_cracking project
within the stego_analysis tools.
"""
import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from .analyzer import analyze_with_keywords, detect_steganography


def load_keywords() -> List[str]:
    """
    Load keywords from the image_cracking project's keywords file.
    
    Returns:
        List of keywords from the image_cracking project
    """
    try:
        # Find the image_cracking keywords.json file
        project_root = Path(__file__).resolve().parents[4]
        keywords_path = project_root / "yzy_investigation" / "projects" / \
            "image_cracking" / "config" / "keywords.json"
            
        if not keywords_path.exists():
            return []
            
        with open(keywords_path, 'r') as f:
            data = json.load(f)
            
        if isinstance(data, dict) and "keywords" in data:
            return data["keywords"]
        return []
    except Exception as e:
        print(f"Error loading image_cracking keywords: {e}")
        return []


def get_key_numbers() -> List[int]:
    """
    Get key numbers used in the image_cracking project.
    
    Returns:
        List of key numbers from the image_cracking project
    """
    try:
        # Find the image_cracking keywords.json file
        project_root = Path(__file__).resolve().parents[4]
        keywords_path = project_root / "yzy_investigation" / "projects" / \
            "image_cracking" / "config" / "keywords.json"
            
        if not keywords_path.exists():
            return [4, 333, 353]  # Default key numbers
            
        with open(keywords_path, 'r') as f:
            data = json.load(f)
            
        if isinstance(data, dict) and "key_numbers" in data:
            return data["key_numbers"]
        return [4, 333, 353]  # Default key numbers
    except Exception as e:
        print(f"Error loading image_cracking key numbers: {e}")
        return [4, 333, 353]  # Default key numbers
        

def analyze_with_keywords(image_path: Path) -> dict:
    """
    Analyze an image using keywords from the image_cracking project.
    
    Args:
        image_path: Path to the image to analyze
        
    Returns:
        Dictionary with analysis results
    """
    # This is a placeholder - the actual implementation would use the
    # stego_analysis module with the keywords from image_cracking
    keywords = load_keywords()
    key_numbers = get_key_numbers()
    
    # Here we would run the actual analysis
    # For now, just return a placeholder result
    return {
        "image": str(image_path),
        "using_keywords": keywords,
        "using_key_numbers": key_numbers,
        "analysis_complete": True
    }


def analyze_image_with_puzzle_keywords(image_path: str) -> Dict[str, Any]:
    """
    Analyze an image using keywords from the puzzle_cracking project.
    
    Args:
        image_path: Path to the image to analyze
        
    Returns:
        Dictionary with analysis results
    """
    keywords = load_keywords()
    return analyze_with_keywords(image_path, keywords)


def batch_analyze_directory(directory_path: str, extensions: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Analyze all images in a directory using puzzle keywords.
    
    Args:
        directory_path: Path to directory containing images
        extensions: List of file extensions to include (default: ['.jpg', '.jpeg', '.png', '.gif'])
        
    Returns:
        Dictionary mapping filenames to analysis results
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.gif']
    
    results = {}
    dir_path = Path(directory_path)
    
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"Invalid directory path: {directory_path}")
    
    for file_path in dir_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            try:
                results[file_path.name] = analyze_image_with_puzzle_keywords(str(file_path))
            except Exception as e:
                results[file_path.name] = {"error": str(e)}
    
    return results 