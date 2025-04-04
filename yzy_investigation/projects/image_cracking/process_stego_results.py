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
import time

class ResultSignificanceChecker:
    """
    Specialized checker to evaluate the significance of stego analysis results.
    
    This class focuses on identifying high-value patterns or strings in the analysis
    results, particularly targeting specific keywords and patterns of interest.
    """
    
    # Key strings to look for (in order of significance)
    TARGET_STRING = "4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump"
    HIGH_VALUE_TERMS = ["4NBT", "silver"]
    
    # Numbers of interest for pattern analysis
    KEY_NUMBERS = [4, 333, 353]
    
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the significance checker.
        
        Args:
            logger: Optional logger to use; creates one if not provided
        """
        self.logger = logger or logging.getLogger(__name__)
        self.found_target = False
        self.high_significance_results: List[Dict[str, Any]] = []
        self.medium_significance_results: List[Dict[str, Any]] = []
        self.low_significance_results: List[Dict[str, Any]] = []
    
    def check_text_data(self, data: str, source: str) -> Dict[str, Any]:
        """
        Check text data for significant content.
        
        Args:
            data: Text data to analyze
            source: Description of where the data came from
            
        Returns:
            Dictionary with significance findings
        """
        # Handle empty data
        if not data:
            return {
                "source": source,
                "significance": "low",
                "findings": []
            }
            
        result = {
            "source": source,
            "significance": "low",
            "findings": []
        }
        
        # Check if this is a file generated by our analysis
        is_analysis_artifact = self._is_analysis_artifact(source, data)
        
        # Check for target string (highest significance)
        if self.TARGET_STRING in data:
            # Don't consider the finding significant if it's in a header or created by our own analysis
            if not self._is_in_file_header(data, self.TARGET_STRING) and not is_analysis_artifact:
                finding = {
                    "type": "target_string",
                    "term": self.TARGET_STRING,
                    "context": self._get_string_context(data, self.TARGET_STRING),
                    "significance": "high"
                }
                result["findings"].append(finding)
                result["significance"] = "high"
                self.found_target = True
                self.logger.warning(f"FOUND TARGET STRING in {source}!")
            else:
                # Log it as a low significance when it's from our own analysis
                finding = {
                    "type": "analysis_artifact",
                    "term": self.TARGET_STRING,
                    "context": self._get_string_context(data, self.TARGET_STRING),
                    "significance": "low",
                    "note": "Target string found in analysis metadata or generated file header"
                }
                result["findings"].append(finding)
                self.logger.info(f"TARGET STRING found in analysis artifact: {source}")
            
        # Check for high-value terms
        for term in self.HIGH_VALUE_TERMS:
            if term in data:
                # Skip if it's just in our generated file header
                if not self._is_in_file_header(data, term) and not is_analysis_artifact:
                    finding = {
                        "type": "high_value_term",
                        "term": term,
                        "context": self._get_string_context(data, term),
                        "significance": "medium"
                    }
                    result["findings"].append(finding)
                    if result["significance"] != "high":
                        result["significance"] = "medium"
                    self.logger.info(f"Found high-value term '{term}' in {source}")
                
        # Check for key numbers - with more careful criteria to avoid false positives
        for num in self.KEY_NUMBERS:
            num_str = str(num)
            
            # Skip reporting single digits unless they appear in specific contexts
            # to avoid excessive false positives (e.g., "4" is common)
            if len(num_str) == 1:
                # Look for the number in meaningful contexts (with delimiters)
                patterns = [f" {num_str} ", f"[{num_str}]", f"({num_str})", f":{num_str}:", f".{num_str}."]
                if not any(pattern in data for pattern in patterns):
                    continue
            else:
                # For multi-digit key numbers, ensure they appear as distinct numbers
                # by checking for word boundaries or special characters around them
                if not re.search(fr'\b{num_str}\b', data):
                    continue
                    
            finding = {
                "type": "key_number",
                "number": num,
                "context": self._get_string_context(data, num_str),
                "significance": "low"
            }
            result["findings"].append(finding)
        
        # Check for readable text
        if self._is_readable_text(data):
            finding = {
                "type": "readable_text",
                "sample": data[:100] + ("..." if len(data) > 100 else ""),
                "significance": "low"
            }
            result["findings"].append(finding)
        
        # Store the result based on significance
        if result["significance"] == "high":
            self.high_significance_results.append(result)
        elif result["significance"] == "medium":
            self.medium_significance_results.append(result)
        elif result["findings"]:  # Only store low significance if we have findings
            self.low_significance_results.append(result)
            
        return result
    
    def _is_analysis_artifact(self, source: str, data: str) -> bool:
        """
        Check if the source or data appears to be an artifact of our own analysis.
        
        Args:
            source: The source description
            data: The data content
            
        Returns:
            True if this is likely an artifact of our analysis
        """
        # Check filename patterns that we generate
        artifact_patterns = [
            r'decoded_with_.*\.txt',
            r'.*_highlighted\.txt',
            r'.*_analysis\.txt',
            r'using.*xor.*key',
            r'decoded.*content'
        ]
        
        # Check if source matches any artifact pattern
        if any(re.search(pattern, source, re.IGNORECASE) for pattern in artifact_patterns):
            return True
            
        # Check content for analysis headers
        header_patterns = [
            r'=== DECODED CONTENT FROM.*USING XOR KEY',
            r'=== .* WITH TARGET STRING HIGHLIGHTED',
            r'TARGET STRING: 4NBTf.*mp\nFOUND AT POSITION:'
        ]
        
        # Check first 500 chars for header patterns
        first_portion = data[:500] if len(data) > 500 else data
        if any(re.search(pattern, first_portion, re.IGNORECASE) for pattern in header_patterns):
            return True
            
        return False
    
    def _is_in_file_header(self, data: str, target: str) -> bool:
        """
        Check if the target string appears in what looks like a file header.
        
        Args:
            data: The data content
            target: The target string to check
            
        Returns:
            True if the target appears in what looks like a file header
        """
        # Get position of the target
        pos = data.find(target)
        if pos == -1:
            return False
            
        # Extract content around the target
        # Look at a bigger window for headers (up to 200 chars before)
        start = max(0, pos - 200)
        end = min(len(data), pos + len(target) + 100)
        surrounding = data[start:end]
        
        # Check if this appears to be in a header section
        header_indicators = [
            "=== DECODED CONTENT FROM",
            "WITH TARGET STRING HIGHLIGHTED",
            "USING XOR KEY:",
            "TARGET STRING:",
            "FOUND AT POSITION:",
            "CONTEXT AROUND"
        ]
        
        return any(indicator in surrounding for indicator in header_indicators)
    
    def check_binary_data(self, data: bytes, source: str) -> Dict[str, Any]:
        """
        Check binary data for significant content.
        
        Args:
            data: Binary data to analyze
            source: Description of where the data came from
            
        Returns:
            Dictionary with significance findings
        """
        # Handle empty data
        if not data or len(data) == 0:
            return {
                "source": source,
                "significance": "low",
                "findings": []
            }
            
        result = {
            "source": source,
            "significance": "low",
            "findings": []
        }
        
        # Try to interpret as text first
        try:
            text_data = data.decode('ascii', errors='ignore')
            if text_data:  # Ensure text data isn't empty
                text_result = self.check_text_data(text_data, f"{source} (as text)")
                
                # Copy findings from text analysis
                result["findings"].extend(text_result["findings"])
                
                # Update significance based on text findings
                if text_result["significance"] == "high":
                    result["significance"] = "high"
                elif text_result["significance"] == "medium" and result["significance"] != "high":
                    result["significance"] = "medium"
        except:
            pass
            
        # Check for patterns related to key numbers
        for num in self.KEY_NUMBERS:
            num_byte = num % 256  # Ensure valid byte value
            
            # For single digit numbers, we need a clear pattern to avoid false positives
            if num < 10:
                # Look for sequences that are unlikely to occur naturally
                pattern = bytes([num_byte]) * 5  # Repeated 5 times
            else:
                # For multi-digit numbers, at least 3 repetitions
                pattern = bytes([num_byte]) * 3  # Repeated 3 times
                
            if pattern in data:
                finding = {
                    "type": "key_number_pattern",
                    "number": num,
                    "pattern": base64.b64encode(pattern).decode('ascii'),
                    "significance": "low"
                }
                result["findings"].append(finding)
                
        # Store the result based on significance
        if result["significance"] == "high":
            self.high_significance_results.append(result)
        elif result["significance"] == "medium":
            self.medium_significance_results.append(result)
        elif result["findings"]:  # Only store low significance if we have findings
            self.low_significance_results.append(result)
            
        return result
    
    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all significance findings.
        
        Returns:
            Dictionary summarizing the significance findings
        """
        return {
            "found_target": self.found_target,
            "high_significance_count": len(self.high_significance_results),
            "medium_significance_count": len(self.medium_significance_results),
            "low_significance_count": len(self.low_significance_results),
            "high_significance_results": self.high_significance_results,
            "medium_significance_results": self.medium_significance_results,
            "low_significance_results": self.low_significance_results[:10]  # Limit to avoid massive output
        }
    
    def _get_string_context(self, text: str, target: str) -> str:
        """
        Get the context around a target string.
        
        Args:
            text: The full text to search in
            target: The target string to find
            
        Returns:
            String context (40 chars before and after)
        """
        pos = text.find(target)
        if pos == -1:
            return ""
            
        start = max(0, pos - 40)
        end = min(len(text), pos + len(target) + 40)
        
        return text[start:end]
    
    def _is_readable_text(self, text: str) -> bool:
        """
        Check if a string appears to be readable text.
        
        Args:
            text: Text to check
            
        Returns:
            True if the text appears readable
        """
        # Require a minimum length to avoid false positives
        if len(text) < 20:
            return False
            
        # Count printable ASCII characters
        printable_chars = sum(1 for c in text if 32 <= ord(c) <= 126)
        ratio = printable_chars / len(text)
        
        # Check for common word patterns
        has_space_patterns = len(re.findall(r'\w+\s+\w+', text)) > 3
        
        return ratio > 0.8 and has_space_patterns


class StegoResultProcessor:
    """
    Process and organize extracted steganography data from multiple images.
    """
    
    def __init__(self, results_dir: Path, output_dir: Path, timestamp: Optional[str] = None) -> None:
        """
        Initialize the data processor.
        
        Args:
            results_dir: Directory containing the stego analysis results
            output_dir: Directory to store processed output
            timestamp: Optional timestamp for directory naming (uses current time if None)
        """
        self.results_dir = Path(results_dir)
        self.base_output_dir = Path(output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped directory
        if timestamp is None:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        self.timestamp = timestamp
        self.output_dir = self.base_output_dir / f"run_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
        # Set up log directory
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file handler
        file_handler = logging.FileHandler(log_dir / "processing.log")
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        
        # Track all discovered files
        self.discovered_files: List[Dict[str, Any]] = []
        # Track binary data from each image
        self.binary_data_map: Dict[str, bytes] = {}
        
        # Initialize the significance checker
        self.significance_checker = ResultSignificanceChecker(self.logger)
        
        # Track start time for processing
        import time
        self.start_time = time.time()
        
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
        
        # Create directories that will hold our results
        reconstructed_files_dir = self.output_dir / "reconstructed_files"
        combined_lsb_dir = self.output_dir / "combined_lsb_data"
        sequence_dir = self.output_dir / "sequence_analysis"
        
        for directory in [reconstructed_files_dir, combined_lsb_dir, sequence_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        for image_dir in sorted(image_dirs, key=lambda x: self._get_image_number(x.name)):
            self._process_image_data(image_dir)
            
        # After processing all images, attempt to reconstruct files
        self._reconstruct_files()
        
        # Combine LSB data
        self._combine_lsb_data(self.output_dir)
        
        # Analyze for sequences and patterns
        sequence_analysis = self._analyze_sequences()
        
        # Get significance summary
        significance_summary = self.significance_checker.get_results_summary()
        
        # Count genuine findings vs analysis artifacts
        genuine_high_findings = 0
        artifact_high_findings = 0
        
        for result in significance_summary["high_significance_results"]:
            is_artifact = False
            for finding in result.get("findings", []):
                if finding.get("type") == "analysis_artifact":
                    is_artifact = True
                    break
            
            if is_artifact:
                artifact_high_findings += 1
            else:
                genuine_high_findings += 1
        
        # Add counts to the summary
        significance_summary["genuine_high_findings"] = genuine_high_findings
        significance_summary["artifact_high_findings"] = artifact_high_findings
        
        # Update the found_target flag to be true only if there are genuine findings
        significance_summary["found_target"] = genuine_high_findings > 0
        
        # Filter high_significance_results to separate genuine from artifacts
        genuine_findings = []
        artifact_findings = []
        
        for result in significance_summary["high_significance_results"]:
            is_artifact = False
            for finding in result.get("findings", []):
                if finding.get("type") == "analysis_artifact":
                    is_artifact = True
                    break
                    
            if is_artifact:
                artifact_findings.append(result)
            else:
                genuine_findings.append(result)
        
        # Replace high_significance_results with genuine findings only
        significance_summary["high_significance_results"] = genuine_findings
        significance_summary["artifact_findings"] = artifact_findings
        
        # Compile and return results
        results = {
            "total_images_processed": len(image_dirs),
            "files_discovered": len(self.discovered_files),
            "file_types_found": self._get_file_type_summary(),
            "sequence_analysis": sequence_analysis,
            "significance_analysis": significance_summary,
            "timestamp": self.timestamp,
            "output_dir": str(self.output_dir)
        }
        
        # Save summary to file
        with open(self.output_dir / "processing_summary.json", "w") as f:
            json.dump(results, f, indent=2)
            
        # Also save a copy to the base output directory for quick reference
        with open(self.base_output_dir / f"latest_run_summary_{self.timestamp}.txt", "w") as f:
            f.write(f"Latest processing run: {self.timestamp}\n\n")
            f.write(f"Input directory: {str(self.results_dir.parent)}\n")
            f.write(f"Full results in: {str(self.output_dir)}\n\n")
            import time
            f.write(f"Analysis completed in {time.time() - self.start_time:.2f} seconds\n")
            f.write(f"Total images: {summary.get('total_images', 0)}\n")
            f.write(f"Images with hidden data: {summary.get('images_with_hidden_data', 0)}\n")
            
            # Add strategy success counts
            if 'strategy_success_counts' in summary:
                f.write("Strategy success counts:\n")
                for strategy, count in summary['strategy_success_counts'].items():
                    f.write(f"  - {strategy}: {count}\n")
            
            # Add significance findings with distinction between genuine and artifacts
            f.write(f"\nHigh significance findings: {genuine_high_findings} genuine, {artifact_high_findings} analysis artifacts\n")
            f.write(f"Medium significance findings: {significance_summary['medium_significance_count']}\n")
            f.write(f"Low significance findings: {significance_summary['low_significance_count']}\n")
            f.write(f"Files discovered: {len(self.discovered_files)}\n")
            
            if genuine_high_findings > 0:
                f.write("\n!!! GENUINE TARGET STRING FOUND !!!\n")
            
        # If we found genuine target findings, create a special file to highlight this
        if genuine_high_findings > 0:
            with open(self.output_dir / "TARGET_FOUND.txt", "w") as f:
                f.write("!!! GENUINE TARGET STRING FOUND !!!\n\n")
                for result in genuine_findings:
                    f.write(f"Source: {result['source']}\n")
                    for finding in result["findings"]:
                        if finding["type"] == "target_string":
                            f.write(f"Context: {finding['context']}\n\n")
            
            # Also save a copy to the base directory for visibility
            with open(self.base_output_dir / "TARGET_FOUND.txt", "w") as f:
                f.write(f"!!! GENUINE TARGET STRING FOUND in run {self.timestamp} !!!\n\n")
                for result in genuine_findings:
                    f.write(f"Source: {result['source']}\n")
                    for finding in result["findings"]:
                        if finding["type"] == "target_string":
                            f.write(f"Context: {finding['context']}\n\n")
                f.write(f"Full details in: {self.output_dir}/TARGET_FOUND.txt\n")
        # If we only have artifact findings, mention that as well
        elif artifact_high_findings > 0:
            with open(self.output_dir / "ANALYSIS_ARTIFACTS_ONLY.txt", "w") as f:
                f.write("NOTE: All high significance findings were determined to be analysis artifacts\n\n")
                f.write("These findings are from our own analysis process and not from the original data.\n")
                f.write("Examples include finding target strings in headers we generated or in analysis metadata.\n\n")
                f.write("Review the processing_summary.json for details.\n")
            
        self.logger.info(f"Processing complete. Found {len(self.discovered_files)} potential files across {len(image_dirs)} images")
        if genuine_high_findings > 0:
            self.logger.warning(f"Found {genuine_high_findings} genuine high significance findings")
        elif artifact_high_findings > 0:
            self.logger.info(f"Found {artifact_high_findings} high significance findings, but all were analysis artifacts")
        
        return results
    
    def _get_image_number(self, image_name: str) -> int:
        """
        Extract image number for sorting.
        
        Args:
            image_name: Name of the image directory (may include parent directory name)
            
        Returns:
            Integer representing the image number
        """
        # Handle new format which may include parent directory: "parent_image_01"
        # First split by '_' and extract the number at the end
        parts = image_name.split('_')
        
        # Look for digit pattern in the parts
        for part in reversed(parts):  # Start from the end
            match = re.search(r'(\d+)', part)
            if match:
                return int(match.group(1))
        
        # Try to extract a timestamp if present in the name (e.g. "03_10AM_...")
        timestamp_match = re.search(r'(\d{1,2})_(\d{1,2})(AM|PM)', image_name)
        if timestamp_match:
            hour = int(timestamp_match.group(1))
            minute = int(timestamp_match.group(2))
            ampm = timestamp_match.group(3)
            
            # Convert to 24-hour time for better sorting
            if ampm.upper() == 'PM' and hour < 12:
                hour += 12
            elif ampm.upper() == 'AM' and hour == 12:
                hour = 0
                
            # Return a value that guarantees proper sorting
            return hour * 60 + minute
        
        # Fallback if no number found, use the hash of the name
        # This ensures consistent sorting even if the name has no numbers
        return hash(image_name) % 100000
        
    def _process_image_data(self, image_dir: Path) -> None:
        """
        Process data from a single image.
        
        Args:
            image_dir: Directory containing the image's extracted data
        """
        # Extract the original image ID, which might include parent directory
        image_name = image_dir.name
        self.logger.debug(f"Processing data from {image_name}")
        
        # Process LSB data
        lsb_bin_path = image_dir / "lsb_strategy_data.bin"
        if lsb_bin_path.exists():
            with open(lsb_bin_path, "rb") as f:
                lsb_data = f.read()
                self.binary_data_map[f"{image_name}_lsb"] = lsb_data
                self.logger.debug(f"Loaded {len(lsb_data)} bytes of LSB data from {image_name}")
                
                # Check significance of LSB data
                self.significance_checker.check_binary_data(lsb_data, f"{image_name} LSB data")
                
        # Process file signature data
        sig_bin_path = image_dir / "file_signature_strategy_data.bin"
        if sig_bin_path.exists():
            with open(sig_bin_path, "rb") as f:
                sig_data = f.read()
                self.binary_data_map[f"{image_name}_signature"] = sig_data
                
                # Check significance of signature data
                self.significance_checker.check_binary_data(sig_data, f"{image_name} signature data")
                
                # Detect file signatures
                file_info = self._detect_file_type(sig_data)
                if file_info:
                    file_info["source_image"] = image_name
                    self.discovered_files.append(file_info)
                    self.logger.info(f"Discovered {file_info['type']} file ({file_info['size']} bytes) in {image_name}")
        
        # Process any text files
        for txt_file in image_dir.glob("*.txt"):
            with open(txt_file, "r", errors='ignore') as f:
                text_data = f.read()
                
                # Check significance of text data
                self.significance_checker.check_text_data(text_data, f"{image_name} {txt_file.name}")
    
    def _detect_file_type(self, data: bytes) -> Optional[Dict[str, Any]]:
        """
        Detect file type based on file signatures.
        
        Args:
            data: Binary data to analyze
            
        Returns:
            Dictionary with file type information or None
        """
        # Handle empty data
        if not data or len(data) == 0:
            return None
            
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
            # and the text isn't empty
            if len(text) > 0:
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
        Attempt to combine LSB data from multiple images.
        
        Args:
            output_dir: Directory to save combined data
        """
        self.logger.info("Attempting to combine LSB data from multiple images")
        
        # Only proceed if we have LSB data
        lsb_data_items = [(k, v) for k, v in self.binary_data_map.items() if k.endswith("_lsb")]
        if not lsb_data_items:
            self.logger.info("No LSB data to combine")
            return
            
        # Create a directory for combined data
        combined_dir = self.output_dir / "combined_lsb_data"
        combined_dir.mkdir(exist_ok=True)
        
        # Group images by number
        lsb_data_by_number = {}
        for key, data in lsb_data_items:
            image_name = key.replace("_lsb", "")
            image_num = self._get_image_number(image_name)
            lsb_data_by_number[image_num] = data
            
        # Combine in order
        sorted_numbers = sorted(lsb_data_by_number.keys())
        combined_data = b"".join(lsb_data_by_number[num] for num in sorted_numbers)
        
        # Save combined data
        combined_bin_path = combined_dir / "all_lsb_combined.bin"
        with open(combined_bin_path, "wb") as f:
            f.write(combined_data)
        self.logger.info(f"Saved combined LSB data ({len(combined_data)} bytes) to {combined_bin_path}")
        
        # Try to interpret as text
        try:
            text_data = combined_data.decode('utf-8', errors='ignore')
            combined_txt_path = combined_dir / "all_lsb_combined.txt"
            with open(combined_txt_path, "w", errors='ignore') as f:
                f.write(text_data)
            self.logger.info(f"Saved LSB data as text to {combined_txt_path}")
            
            # Check significance
            self.significance_checker.check_text_data(text_data, "Combined LSB data")
        except:
            self.logger.warning("Failed to save LSB data as text")
            
        # Also try to save as various file formats
        for ext in [".jpg", ".png", ".gif", ".pdf", ".zip"]:
            try:
                file_path = combined_dir / f"all_lsb_combined{ext}"
                with open(file_path, "wb") as f:
                    f.write(combined_data)
                self.logger.debug(f"Saved LSB data as {ext}")
            except:
                self.logger.debug(f"Failed to save LSB data as {ext}")
                
        # Try alternative orderings
        try:
            # Reversed order
            reversed_data = b"".join(lsb_data_by_number[num] for num in reversed(sorted_numbers))
            reversed_path = combined_dir / "reverse_lsb_combined.bin"
            with open(reversed_path, "wb") as f:
                f.write(reversed_data)
            self.logger.info(f"Saved reverse-ordered LSB data to {reversed_path}")
            
            # Try to interpret as text
            reversed_text = reversed_data.decode('utf-8', errors='ignore')
            with open(combined_dir / "reverse_lsb_combined.txt", "w", errors='ignore') as f:
                f.write(reversed_text)
            
            # Check significance
            self.significance_checker.check_text_data(reversed_text, "Reverse-ordered LSB data")
        except:
            self.logger.warning("Failed to save reverse-ordered LSB data")
    
    def _analyze_sequences(self) -> Dict[str, Any]:
        """
        Analyze sequences and patterns across binary data.
        
        Returns:
            Dictionary with sequence analysis results
        """
        self.logger.info("Analyzing sequences across binary data")
        
        sequence_dir = self.output_dir / "sequence_analysis"
        sequence_dir.mkdir(exist_ok=True)
        
        results = {}
        
        # Look for repeating sequences
        all_data = b"".join(self.binary_data_map.values())
        if all_data:
            # Look for common sequences
            results["total_data_size"] = len(all_data)
            
            try:
                # Check for key strings in binary data
                key_strings = [self.significance_checker.TARGET_STRING.encode('utf-8')]
                key_strings.extend(term.encode('utf-8') for term in self.significance_checker.HIGH_VALUE_TERMS)
                
                string_positions = {}
                for key_string in key_strings:
                    positions = []
                    pos = 0
                    while True:
                        pos = all_data.find(key_string, pos)
                        if pos == -1:
                            break
                        positions.append(pos)
                        pos += len(key_string)
                    
                    if positions:
                        string_positions[key_string.decode('utf-8')] = positions
                
                if string_positions:
                    results["key_string_positions"] = string_positions
            except:
                pass
                
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
    
    # Get timestamp for this run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Process the results
    processor = StegoResultProcessor(Path(args.results_dir), Path(args.output_dir), timestamp)
    results = processor.process_all_images()
    
    print("\nSteganography Results Processing Complete")
    print(f"Processed {results['total_images_processed']} images")
    print(f"Discovered {results['files_discovered']} potential files")
    print("\nFile types found:")
    for file_type, count in results['file_types_found'].items():
        print(f"  - {file_type}: {count}")
    
    print(f"\nResults saved to: {results['output_dir']}")
    print(f"Detailed summary in: {results['output_dir']}/processing_summary.json")
    print(f"Reconstructed files in: {results['output_dir']}/reconstructed_files/")


if __name__ == "__main__":
    main() 