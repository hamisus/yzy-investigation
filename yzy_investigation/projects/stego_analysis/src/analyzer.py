"""
Steganography analyzer module for detecting hidden data in images.

This module uses Aletheia to analyze images and detect steganographic content.
"""
import os
import subprocess
from typing import Dict, List, Union, Optional, Tuple
from pathlib import Path


def run_aletheia_command(command: str, image_path: str, *args) -> Tuple[str, int]:
    """
    Run an Aletheia command on the specified image.
    
    Args:
        command: The Aletheia command to run
        image_path: Path to the image to analyze
        args: Additional arguments for the command
        
    Returns:
        Tuple containing command output and exit code
    """
    full_command = ["aletheia.py", command, image_path]
    if args:
        full_command.extend(args)
    
    try:
        result = subprocess.run(
            full_command, 
            check=False,
            capture_output=True,
            text=True
        )
        return result.stdout, result.returncode
    except Exception as e:
        return f"Error executing command: {str(e)}", 1


def detect_steganography(image_path: str) -> Dict[str, Dict[str, Union[str, bool]]]:
    """
    Run multiple steganography detection methods on an image.
    
    Args:
        image_path: Path to the image to analyze
        
    Returns:
        Dictionary with results from different detection methods
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    results = {}
    
    # Run automated analysis
    auto_output, _ = run_aletheia_command("auto", image_path)
    results["auto"] = {"output": auto_output, "detected": "detected" in auto_output.lower()}
    
    # Run LSB detection methods
    for method in ["spa", "rs", "ws", "triples"]:
        output, _ = run_aletheia_command(method, image_path)
        results[method] = {
            "output": output,
            "detected": "detected" in output.lower() or "positive" in output.lower()
        }
    
    # Check for JPEG steganography if it's a JPEG image
    if image_path.lower().endswith((".jpg", ".jpeg")):
        output, _ = run_aletheia_command("calibration", image_path)
        results["calibration"] = {"output": output, "detected": "detected" in output.lower()}
    
    return results


def analyze_with_keywords(image_path: str, keywords: List[str]) -> Dict[str, Dict[str, Union[str, bool]]]:
    """
    Analyze an image with Aletheia and check for keyword-related steganography.
    
    Args:
        image_path: Path to the image to analyze
        keywords: List of keywords to use for password cracking
        
    Returns:
        Dictionary with analysis results including any extracted data
    """
    # Run basic detection first
    results = detect_steganography(image_path)
    
    # If steganography is detected, try to extract content using keywords as passwords
    if any(method["detected"] for method in results.values()):
        for method in ["brute-force-steghide", "brute-force-outguess", "brute-force-f5"]:
            # Create a temporary password file
            temp_password_file = Path("temp_passwords.txt")
            with open(temp_password_file, "w") as f:
                f.write("\n".join(keywords))
            
            # Run the brute force attack
            output, _ = run_aletheia_command(
                method, 
                image_path, 
                str(temp_password_file)
            )
            
            # Add results
            results[method] = {
                "output": output,
                "success": "password found" in output.lower() or "extracted" in output.lower()
            }
            
            # Clean up
            if temp_password_file.exists():
                temp_password_file.unlink()
    
    return results 