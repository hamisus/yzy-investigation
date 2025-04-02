"""
Process and analyze steganography results from our puzzle cracker.

This module provides functionality to:
1. Process extracted data from steganography analysis
2. Reconstruct potential hidden files
3. Combine fragments across multiple images
4. Identify patterns and sequences in the hidden data
"""

import json
import base64
import shutil
import re
import struct
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

class StegoResultProcessor:
    """
    Process and organize extracted steganography data from multiple images.
    """
    
    def __init__(self, results_dir: Path, output_dir: Path) -> None:
        """
        Initialize the data processor.
        
        Args:
            results_dir: Directory containing the stego analysis results
            output_dir: Directory to store processed output
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
        # Track all discovered files
        self.discovered_files: List[Dict[str, Any]] = []
        # Track binary data from each image
        self.binary_data_map: Dict[str, bytes] = {}
        
    def process_all_images(self) -> Dict[str, Any]:
        """
        Process all image results from the stego analysis.
        
        Returns:
            Dictionary with processing summary
        """
        self.logger.info(f"Processing steganography results from {self.results_dir}")
        
        # Read summary first
        summary_path = self.results_dir / "analysis_summary.json"
        with open(summary_path, "r") as f:
            summary = json.load(f)
            
        # Process extracted data directory
        extracted_data_dir = self.results_dir / "extracted_data"
        image_dirs = [d for d in extracted_data_dir.iterdir() if d.is_dir()]
        
        self.logger.info(f"Found {len(image_dirs)} image directories to process")
        
        for image_dir in sorted(image_dirs, key=lambda x: self._get_image_number(x.name)):
            self._process_image_data(image_dir)
            
        # After processing all images, attempt to reconstruct files
        self._reconstruct_files()
        
        # Analyze for sequences and patterns
        sequence_analysis = self._analyze_sequences()
        
        # Compile and return results
        results = {
            "total_images_processed": len(image_dirs),
            "files_discovered": len(self.discovered_files),
            "file_types_found": self._get_file_type_summary(),
            "sequence_analysis": sequence_analysis
        }
        
        # Save summary to file
        with open(self.output_dir / "processing_summary.json", "w") as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"Processing complete. Found {len(self.discovered_files)} potential files across {len(image_dirs)} images")
        
        return results
    
    def _get_image_number(self, image_name: str) -> int:
        """
        Extract image number for sorting.
        
        Args:
            image_name: Name of the image directory
            
        Returns:
            Integer representing the image number
        """
        match = re.search(r'image_(\d+)', image_name)
        if match:
            return int(match.group(1))
        return 0
        
    def _process_image_data(self, image_dir: Path) -> None:
        """
        Process data from a single image.
        
        Args:
            image_dir: Directory containing the image's extracted data
        """
        image_name = image_dir.name
        self.logger.debug(f"Processing data from {image_name}")
        
        # Process LSB data
        lsb_bin_path = image_dir / "lsb_strategy_data.bin"
        if lsb_bin_path.exists():
            with open(lsb_bin_path, "rb") as f:
                lsb_data = f.read()
                self.binary_data_map[f"{image_name}_lsb"] = lsb_data
                self.logger.debug(f"Loaded {len(lsb_data)} bytes of LSB data from {image_name}")
                
        # Process file signature data
        sig_bin_path = image_dir / "file_signature_strategy_data.bin"
        if sig_bin_path.exists():
            with open(sig_bin_path, "rb") as f:
                sig_data = f.read()
                self.binary_data_map[f"{image_name}_signature"] = sig_data
                
                # Detect file signatures
                file_info = self._detect_file_type(sig_data)
                if file_info:
                    file_info["source_image"] = image_name
                    self.discovered_files.append(file_info)
                    self.logger.info(f"Discovered {file_info['type']} file ({file_info['size']} bytes) in {image_name}")
    
    def _detect_file_type(self, data: bytes) -> Optional[Dict[str, Any]]:
        """
        Detect file type based on file signatures.
        
        Args:
            data: Binary data to analyze
            
        Returns:
            Dictionary with file type information or None
        """
        # Common file signatures
        signatures = {
            b"\x50\x4B\x03\x04": {"type": "zip", "ext": ".zip"},
            b"\x50\x4B\x05\x06": {"type": "zip (empty)", "ext": ".zip"},
            b"\x50\x4B\x07\x08": {"type": "zip (spanned)", "ext": ".zip"},
            b"\x25\x50\x44\x46": {"type": "pdf", "ext": ".pdf"},
            b"\xFF\xD8\xFF": {"type": "jpeg", "ext": ".jpg"},
            b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": {"type": "png", "ext": ".png"},
            b"\x47\x49\x46\x38": {"type": "gif", "ext": ".gif"},
            b"\x52\x61\x72\x21": {"type": "rar", "ext": ".rar"},
            b"\x1F\x8B\x08": {"type": "gzip", "ext": ".gz"},
            b"\x42\x5A\x68": {"type": "bzip2", "ext": ".bz2"},
            b"\x37\x7A\xBC\xAF\x27\x1C": {"type": "7zip", "ext": ".7z"},
            b"\x75\x73\x74\x61\x72": {"type": "tar", "ext": ".tar"},
            b"\xFF\xFB": {"type": "mp3", "ext": ".mp3"},
            b"\xFF\xF3": {"type": "mp3", "ext": ".mp3"},
            b"\xFF\xF2": {"type": "mp3", "ext": ".mp3"},
            b"\x49\x44\x33": {"type": "mp3", "ext": ".mp3"},
            b"\x00\x01\x00\x00\x00": {"type": "ttf", "ext": ".ttf"},
        }
        
        # Look for signatures
        for sig, info in signatures.items():
            if data.startswith(sig):
                return {
                    "type": info["type"],
                    "extension": info["ext"],
                    "size": len(data),
                    "data": data
                }
                
        # If no common signature is found, check for text files
        try:
            text = data.decode('utf-8', errors='strict')
            # Only consider it text if it has a high proportion of printable chars
            printable = sum(1 for c in text if 32 <= ord(c) <= 126 or c in '\n\r\t')
            if printable / len(text) > 0.9:
                return {
                    "type": "text",
                    "extension": ".txt",
                    "size": len(data),
                    "data": data
                }
        except UnicodeDecodeError:
            pass
            
        return None
        
    def _reconstruct_files(self) -> None:
        """Attempt to reconstruct complete files from fragments."""
        file_output_dir = self.output_dir / "reconstructed_files"
        file_output_dir.mkdir(exist_ok=True)
        
        self.logger.info("Attempting to reconstruct files from fragments")
        
        # Group files by type
        files_by_type = {}
        for file_info in self.discovered_files:
            file_type = file_info["type"]
            if file_type not in files_by_type:
                files_by_type[file_type] = []
            files_by_type[file_type].append(file_info)
            
        # Save each file
        for file_type, files in files_by_type.items():
            # Sort by image number for sequential files
            files.sort(key=lambda x: self._get_image_number(x["source_image"]))
            
            self.logger.info(f"Processing {len(files)} files of type {file_type}")
            
            # Create individual files
            for i, file_info in enumerate(files):
                filename = f"{file_type}_{i+1}{file_info['extension']}"
                filepath = file_output_dir / filename
                with open(filepath, "wb") as f:
                    f.write(file_info["data"])
                self.logger.debug(f"Saved individual file: {filename}")
                    
            # Try to concatenate files if there are multiple of the same type
            if len(files) > 1:
                combined_data = b"".join(file_info["data"] for file_info in files)
                combined_path = file_output_dir / f"{file_type}_combined{files[0]['extension']}"
                with open(combined_path, "wb") as f:
                    f.write(combined_data)
                self.logger.info(f"Created combined file of type {file_type}: {combined_path.name} ({len(combined_data)} bytes)")

        # Also try to combine LSB data across images in different ways
        self._combine_lsb_data(file_output_dir)
                    
    def _combine_lsb_data(self, output_dir: Path) -> None:
        """
        Combine LSB data from multiple images in different ways.
        
        Args:
            output_dir: Directory to store combined outputs
        """
        self.logger.info("Combining LSB data across images")
        
        # Get all LSB data sorted by image number
        lsb_items = [(key, data) for key, data in self.binary_data_map.items() if "_lsb" in key]
        lsb_items.sort(key=lambda x: self._get_image_number(x[0]))
        
        # Simple concatenation
        lsb_data = b"".join(data for _, data in lsb_items)
        lsb_combined_path = output_dir / "combined_lsb_data.bin"
        with open(lsb_combined_path, "wb") as f:
            f.write(lsb_data)
        self.logger.info(f"Created combined LSB data file: {lsb_combined_path.name} ({len(lsb_data)} bytes)")
            
        # Try to detect file type in combined LSB data
        file_info = self._detect_file_type(lsb_data)
        if file_info:
            # Rename with proper extension
            new_path = output_dir / f"combined_lsb_data{file_info['extension']}"
            shutil.move(lsb_combined_path, new_path)
            self.logger.info(f"Detected {file_info['type']} in combined LSB data, renamed to {new_path.name}")
            
        # Try interleaving bytes
        if len(lsb_items) > 1:
            interleaved = bytearray()
            max_length = max(len(data) for _, data in lsb_items)
            
            for i in range(max_length):
                for _, data in lsb_items:
                    if i < len(data):
                        interleaved.append(data[i])
                        
            interleaved_path = output_dir / "interleaved_lsb_data.bin"
            with open(interleaved_path, "wb") as f:
                f.write(interleaved)
            self.logger.info(f"Created interleaved LSB data file: {interleaved_path.name} ({len(interleaved)} bytes)")
            
            # Check file type
            file_info = self._detect_file_type(interleaved)
            if file_info:
                new_path = output_dir / f"interleaved_lsb_data{file_info['extension']}"
                shutil.move(interleaved_path, new_path)
                self.logger.info(f"Detected {file_info['type']} in interleaved LSB data, renamed to {new_path.name}")
    
    def _analyze_sequences(self) -> Dict[str, Any]:
        """
        Analyze sequences and patterns in the extracted data.
        
        Returns:
            Dictionary with sequence analysis results
        """
        self.logger.info("Analyzing sequences and patterns in the data")
        
        results = {
            "color_histogram_pattern": None,
            "image_sequence_analysis": None
        }
        
        # The 42/60 images with color histogram pattern might be significant
        # Could be binary data (0/1) across the sequence of images
        
        # Get list of all images with data
        image_numbers = set()
        for key in self.binary_data_map.keys():
            match = re.search(r'image_(\d+)', key)
            if match:
                image_numbers.add(int(match.group(1)))
        
        images_with_data = sorted(list(image_numbers))
        results["images_with_data"] = images_with_data
        self.logger.info(f"Found data in {len(images_with_data)} unique images")
        
        return results
                
    def _get_file_type_summary(self) -> Dict[str, int]:
        """
        Get summary of file types found.
        
        Returns:
            Dictionary mapping file types to counts
        """
        summary = {}
        for file_info in self.discovered_files:
            file_type = file_info["type"]
            if file_type not in summary:
                summary[file_type] = 0
            summary[file_type] += 1
        return summary


def main() -> None:
    """Main entry point for the steganography results processor."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process steganography analysis results')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory containing the steganography analysis results')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to store processed output')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, 
                         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Process the results
    processor = StegoResultProcessor(Path(args.results_dir), Path(args.output_dir))
    results = processor.process_all_images()
    
    print("\nSteganography Results Processing Complete")
    print(f"Processed {results['total_images_processed']} images")
    print(f"Discovered {results['files_discovered']} potential files")
    print("\nFile types found:")
    for file_type, count in results['file_types_found'].items():
        print(f"  - {file_type}: {count}")
    
    print(f"\nDetailed results saved to {args.output_dir}/processing_summary.json")
    print(f"Reconstructed files saved to {args.output_dir}/reconstructed_files/")


if __name__ == "__main__":
    main() 