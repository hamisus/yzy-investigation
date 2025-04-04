"""
Implementation of various steganography analysis strategies.
"""
from pathlib import Path
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import struct
import base64
from PIL.TiffImagePlugin import IFDRational

import numpy as np
from PIL import Image
import hashlib
import itertools
import binascii
import re

from yzy_investigation.projects.image_cracking.stego_analysis import StegStrategy


class LsbStrategy(StegStrategy):
    """
    Strategy to detect and extract hidden data using LSB steganography.
    
    Least Significant Bit (LSB) steganography is a common technique where
    data is hidden in the least significant bits of pixel values.
    """
    
    name: str = "lsb_strategy"
    description: str = "Least Significant Bit steganography detection and extraction"
    
    def __init__(self, bits_to_check: int = 1) -> None:
        """
        Initialize the LSB strategy.
        
        Args:
            bits_to_check: Number of least significant bits to analyze (1-4)
        """
        super().__init__()
        self.bits_to_check = min(max(1, bits_to_check), 4)  # Limit to 1-4 bits
    
    def analyze(self, image_path: Path) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Analyze an image for LSB steganography.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple containing:
                - Boolean indicating if hidden data was detected
                - Optional dictionary with extracted data and analysis metadata
        """
        try:
            img = self.load_image(image_path)
            img_array = np.array(img)
            
            # Flatten the array and extract the LSBs
            if len(img_array.shape) == 3:  # Color image
                flat_array = img_array.reshape(-1, img_array.shape[2])
                # Extract LSBs from each color channel
                lsb_data = []
                for channel in range(flat_array.shape[1]):
                    channel_lsbs = flat_array[:, channel] & ((1 << self.bits_to_check) - 1)
                    lsb_data.extend(channel_lsbs)
            else:  # Grayscale image
                flat_array = img_array.reshape(-1)
                lsb_data = flat_array & ((1 << self.bits_to_check) - 1)
            
            # Analyze for non-random patterns
            is_random = self._check_randomness(lsb_data)
            
            # Try to extract meaningful data
            extracted_data = None
            if not is_random:
                extracted_data = self._extract_data(img_array)
                
            result_data = {
                "is_random": is_random,
                "bits_checked": self.bits_to_check,
                "randomness_score": float(self._calculate_randomness_score(lsb_data))  # Convert numpy float to Python float
            }
            
            if extracted_data is not None:
                if isinstance(extracted_data, str):
                    result_data["extracted_data"] = extracted_data
                else:
                    # Binary data
                    result_data["extracted_data"] = {
                        "type": "binary",
                        "encoding": "base64",
                        "data": base64.b64encode(extracted_data).decode('ascii')
                    }
            
            return (not is_random, result_data)
            
        except Exception as e:
            self.logger.error(f"Error in LSB analysis: {e}")
            return (False, {"error": str(e)})
    
    def _check_randomness(self, lsb_data: Union[List[int], np.ndarray]) -> bool:
        """
        Check if the LSB data appears random or not.
        
        Args:
            lsb_data: Array of LSB values
            
        Returns:
            Boolean indicating if data appears random (True) or not (False)
        """
        # Convert to numpy array if not already
        if not isinstance(lsb_data, np.ndarray):
            lsb_data = np.array(lsb_data)
        
        # Check frequency distribution
        for bit in range(self.bits_to_check):
            bit_values = (lsb_data >> bit) & 1
            ones_ratio = np.sum(bit_values) / len(bit_values)
            
            # For truly random data, we expect ones_ratio to be close to 0.5
            if abs(ones_ratio - 0.5) > 0.05:  # 5% threshold
                return False
                
        # Check for repeating patterns
        if len(lsb_data) > 1000:
            # Sample sizes to check for patterns
            for window in [8, 16, 32, 64]:
                if self._has_repeating_patterns(lsb_data, window):
                    return False
        
        return True
    
    def _has_repeating_patterns(self, data: np.ndarray, window_size: int) -> bool:
        """
        Check for repeating patterns in the data.
        
        Args:
            data: The data to check
            window_size: Size of the pattern window to check
            
        Returns:
            Boolean indicating if repeating patterns were found
        """
        if len(data) < window_size * 2:
            return False
            
        # Sample the data to avoid excessive computation
        max_samples = 10000
        step = max(1, len(data) // max_samples)
        sampled_data = data[::step]
        
        # Check for repeating patterns
        for i in range(len(sampled_data) - window_size * 2):
            window = sampled_data[i:i+window_size]
            next_window = sampled_data[i+window_size:i+window_size*2]
            if np.array_equal(window, next_window):
                return True
                
        return False
    
    def _calculate_randomness_score(self, lsb_data: Union[List[int], np.ndarray]) -> float:
        """
        Calculate a randomness score for the LSB data.
        
        Args:
            lsb_data: Array of LSB values
            
        Returns:
            Score between 0.0 (not random) and 1.0 (random)
        """
        # Convert to numpy array if not already
        if not isinstance(lsb_data, np.ndarray):
            lsb_data = np.array(lsb_data)
            
        if len(lsb_data) == 0:
            return 0.5
            
        # Check bit distribution for each LSB
        bit_scores = []
        for bit in range(self.bits_to_check):
            bit_values = (lsb_data >> bit) & 1
            ones_ratio = np.sum(bit_values) / len(bit_values)
            # Score based on how close ones_ratio is to 0.5
            bit_scores.append(1.0 - abs(ones_ratio - 0.5) * 2)
        
        return sum(bit_scores) / len(bit_scores)
    
    def _extract_data(self, img_array: np.ndarray) -> Optional[Union[str, bytes]]:
        """
        Attempt to extract meaningful data from the image.
        
        Args:
            img_array: Numpy array representation of the image
            
        Returns:
            Extracted data as string or bytes, or None if no valid data found
        """
        try:
            # Extract all LSBs as bits
            if len(img_array.shape) == 3:  # Color image
                height, width, channels = img_array.shape
                bit_array = np.zeros(height * width * channels, dtype=np.uint8)
                
                index = 0
                for h in range(height):
                    for w in range(width):
                        for c in range(channels):
                            bit_array[index] = img_array[h, w, c] & 1
                            index += 1
            else:  # Grayscale image
                height, width = img_array.shape
                bit_array = np.zeros(height * width, dtype=np.uint8)
                
                index = 0
                for h in range(height):
                    for w in range(width):
                        bit_array[index] = img_array[h, w] & 1
                        index += 1
            
            # Convert bits to bytes
            byte_array = bytearray()
            for i in range(0, len(bit_array) - 7, 8):
                byte_val = 0
                for j in range(8):
                    if i + j < len(bit_array):
                        byte_val |= bit_array[i + j] << j
                byte_array.append(byte_val)
            
            # Try to interpret as ASCII text
            try:
                # Look for printable ASCII characters
                text = byte_array.decode('ascii', errors='ignore')
                
                # Check if the decoded text contains a reasonable proportion of printable chars
                printable_chars = sum(1 for c in text if 32 <= ord(c) <= 126)
                if printable_chars / len(text) > 0.8:  # 80% threshold
                    return text
            except:
                pass
                
            # Return raw bytes if we can't decode as text
            return bytes(byte_array)
            
        except Exception as e:
            self.logger.error(f"Error extracting data: {e}")
            return None


class ColorHistogramStrategy(StegStrategy):
    """
    Strategy to detect steganography by analyzing color histograms.
    
    Many steganography techniques subtly alter the color distribution of an image.
    This strategy looks for unusual patterns in the color histogram.
    """
    
    name: str = "color_histogram_strategy"
    description: str = "Color histogram analysis for detecting abnormal patterns"
    
    def analyze(self, image_path: Path) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Analyze an image's color histogram for signs of steganography.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple containing:
                - Boolean indicating if hidden data was detected
                - Optional dictionary with analysis metadata
        """
        try:
            img = self.load_image(image_path)
            img_array = np.array(img)
            
            # Calculate histograms for each channel
            histograms = {}
            anomalies = {}
            
            if len(img_array.shape) == 3:  # Color image
                channels = ['red', 'green', 'blue']
                for i, channel in enumerate(channels):
                    channel_data = img_array[:, :, i].flatten()
                    hist, _ = np.histogram(channel_data, bins=256, range=(0, 256))
                    histograms[channel] = hist.tolist()  # Convert numpy array to list
                    anomalies[channel] = self._find_histogram_anomalies(hist)
            else:  # Grayscale image
                hist, _ = np.histogram(img_array.flatten(), bins=256, range=(0, 256))
                histograms['gray'] = hist.tolist()  # Convert numpy array to list
                anomalies['gray'] = self._find_histogram_anomalies(hist)
            
            # Detect if there are significant anomalies
            has_anomalies = any(len(anom) > 0 for anom in anomalies.values())
            
            # Calculate comb pattern score (stego often creates "comb" patterns)
            comb_scores = {}
            for channel, hist in histograms.items():
                comb_scores[channel] = float(self._calculate_comb_pattern_score(np.array(hist)))  # Convert score to float
            
            # Check for unusual peaks or patterns
            suspicious = has_anomalies or any(score > 0.7 for score in comb_scores.values())
            
            # Ensure all values are JSON serializable
            result = {
                "histograms": histograms,  # Already converted to lists
                "anomalies": self._make_serializable(anomalies),
                "comb_scores": comb_scores,  # Already converted to floats
                "suspicious": suspicious
            }
            
            return (suspicious, result)
            
        except Exception as e:
            self.logger.error(f"Error in color histogram analysis: {e}")
            return (False, {"error": str(e)})
    
    def _make_serializable(self, data: Any) -> Any:
        """
        Convert data to JSON serializable format.
        
        Args:
            data: Data to convert
            
        Returns:
            JSON serializable data
        """
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.int32, np.int64)):
            return int(data)
        elif isinstance(data, (np.float32, np.float64)):
            return float(data)
        elif isinstance(data, IFDRational):
            return float(data)
        else:
            return data
    
    def _find_histogram_anomalies(self, histogram: np.ndarray) -> List[Dict[str, Any]]:
        """
        Find anomalies in a color histogram.
        
        Args:
            histogram: Numpy array of the histogram
            
        Returns:
            List of anomalies with their type and location
        """
        anomalies = []
        
        # Check for sudden spikes
        for i in range(1, len(histogram) - 1):
            if (histogram[i] > histogram[i-1] * 3 and histogram[i] > histogram[i+1] * 3 and
                histogram[i] > np.mean(histogram) * 2):
                anomalies.append({
                    "type": "spike",
                    "position": int(i),  # Convert numpy int to Python int
                    "value": int(histogram[i])  # Convert numpy int to Python int
                })
        
        # Check for regular patterns
        pattern_size = self._detect_regular_pattern(histogram)
        if pattern_size > 0:
            anomalies.append({
                "type": "regular_pattern",
                "pattern_size": int(pattern_size)  # Convert numpy int to Python int
            })
        
        # Check for unusual gaps
        zero_runs = self._find_zero_runs(histogram)
        for run in zero_runs:
            if run["length"] > 5:  # Only report significant gaps
                anomalies.append({
                    "type": "gap",
                    "start": int(run["start"]),  # Convert numpy int to Python int
                    "length": int(run["length"])  # Convert numpy int to Python int
                })
        
        return anomalies
    
    def _detect_regular_pattern(self, histogram: np.ndarray) -> int:
        """
        Detect if there's a regular repeating pattern in the histogram.
        
        Args:
            histogram: Numpy array of the histogram
            
        Returns:
            The size of the pattern if detected, 0 otherwise
        """
        # Try different pattern sizes
        for size in range(2, 17):  # Check patterns of size 2 to 16
            is_pattern = True
            for i in range(size, len(histogram) - size, size):
                # Compare histogram segments
                segment1 = histogram[i-size:i]
                segment2 = histogram[i:i+size]
                
                # Calculate correlation between segments
                correlation = np.corrcoef(segment1, segment2)[0, 1]
                if np.isnan(correlation) or correlation < 0.7:  # Threshold for similarity
                    is_pattern = False
                    break
            
            if is_pattern:
                return size
                
        return 0
    
    def _find_zero_runs(self, histogram: np.ndarray) -> List[Dict[str, int]]:
        """
        Find runs of zeros or very low values in the histogram.
        
        Args:
            histogram: Numpy array of the histogram
            
        Returns:
            List of dictionaries with start position and length of each run
        """
        threshold = np.mean(histogram) * 0.1  # 10% of mean as threshold
        zero_runs = []
        run_start = -1
        
        for i in range(len(histogram)):
            if histogram[i] <= threshold:
                if run_start == -1:
                    run_start = i
            elif run_start != -1:
                zero_runs.append({"start": run_start, "length": i - run_start})
                run_start = -1
                
        # Check if there's a run at the end
        if run_start != -1:
            zero_runs.append({"start": run_start, "length": len(histogram) - run_start})
            
        return zero_runs
    
    def _calculate_comb_pattern_score(self, histogram: np.ndarray) -> float:
        """
        Calculate a 'comb pattern' score for the histogram.
        
        LSB steganography often creates a comb-like pattern where
        even values are more common than odd values or vice versa.
        
        Args:
            histogram: Numpy array of the histogram
            
        Returns:
            Score between 0.0 (no comb pattern) and 1.0 (strong comb pattern)
        """
        even_sum = np.sum(histogram[::2])
        odd_sum = np.sum(histogram[1::2])
        total_sum = even_sum + odd_sum
        
        if total_sum == 0:
            return 0.0
            
        # Calculate the dominance of even or odd values
        ratio = max(even_sum, odd_sum) / total_sum
        
        # Normalize to a score between 0 and 1
        # 0.5 would be a balanced distribution (no comb pattern)
        # 1.0 would be all values in either even or odd positions
        return (ratio - 0.5) * 2 if ratio > 0.5 else 0.0


class FileSignatureStrategy(StegStrategy):
    """
    Strategy to detect hidden files embedded within images.
    
    This strategy searches for file signatures (magic numbers) that
    might indicate a file has been embedded in the image.
    """
    
    name: str = "file_signature_strategy"
    description: str = "Detection of hidden files based on file signatures"
    
    # Common file signatures (magic numbers)
    FILE_SIGNATURES = {
        b'PK\x03\x04': 'zip',
        b'PK\x05\x06': 'zip (empty)',
        b'PK\x07\x08': 'zip (spanned)',
        b'\x50\x4B\x03\x04\x14\x00\x00\x00\x00\x00': 'zip',
        b'\x89PNG\r\n\x1a\n': 'png',
        b'GIF87a': 'gif',
        b'GIF89a': 'gif',
        b'\xff\xd8\xff': 'jpg',
        b'BM': 'bmp',
        b'\x49\x49\x2a\x00': 'tif (little endian)',
        b'\x4d\x4d\x00\x2a': 'tif (big endian)',
        b'%PDF': 'pdf',
        b'\x7fELF': 'elf',
        b'\x25\x50\x44\x46': 'pdf',
        b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1': 'ms_office',
        b'<!DOCTYPE html': 'html',
        b'<html': 'html',
        b'fLaC': 'flac',
        b'ID3': 'mp3',
        b'\xFF\xFB': 'mp3',
        b'OggS': 'ogg',
        b'\x1a\x45\xdf\xa3': 'matroska',
        b'RIFF': 'wav or avi',
        b'\x00\x00\x01\xba': 'mpeg',
        b'\x00\x00\x01\xb3': 'mpeg',
        b'free': 'mp4',
        b'mdat': 'mp4',
        b'moov': 'mp4',
        b'ftypM4A': 'm4a',
    }
    
    def analyze(self, image_path: Path) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Analyze an image for embedded file signatures.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple containing:
                - Boolean indicating if a hidden file was detected
                - Optional dictionary with analysis results
        """
        try:
            # Read the image as binary data
            with open(image_path, 'rb') as f:
                data = f.read()
            
            # Find all file signatures in the data
            found_signatures = []
            
            for signature, file_type in self.FILE_SIGNATURES.items():
                # Skip the beginning of the file to avoid detecting the image's own signature
                start_pos = len(signature) + 10
                
                # Look for the signature in the rest of the file
                pos = start_pos
                while True:
                    pos = data.find(signature, pos)
                    if pos == -1:
                        break
                    
                    found_signatures.append({
                        "position": pos,
                        "signature": signature.hex(),
                        "file_type": file_type
                    })
                    pos += len(signature)
            
            # Extract potential hidden file content
            extracted_files = []
            for sig_info in found_signatures:
                # Try to extract content after the signature
                start = sig_info["position"]
                # Limit extraction to 1MB to avoid memory issues
                extracted = data[start:start+1024*1024]
                
                extracted_files.append({
                    "start_position": start,
                    "file_type": sig_info["file_type"],
                    "size": len(extracted),
                    "data": base64.b64encode(extracted).decode('ascii')  # Base64 encode binary data
                })
            
            return (len(found_signatures) > 0, {
                "signatures_found": found_signatures,
                "extracted_data": {
                    "type": "binary",
                    "encoding": "base64",
                    "data": extracted_files[0]["data"] if extracted_files else None,
                    "file_type": found_signatures[0]["file_type"] if found_signatures else None
                }
            })
            
        except Exception as e:
            self.logger.error(f"Error in file signature analysis: {e}")
            return (False, {"error": str(e)})


class MetadataAnalysisStrategy(StegStrategy):
    """
    Strategy to analyze image metadata for hidden data.
    
    This strategy looks for unusual or suspicious metadata that could
    contain hidden information.
    """
    
    name: str = "metadata_analysis_strategy"
    description: str = "Analysis of image metadata for hidden information"
    
    def analyze(self, image_path: Path) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Analyze an image's metadata for hidden information.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple containing:
                - Boolean indicating if suspicious metadata was detected
                - Optional dictionary with analysis results
        """
        try:
            img = self.load_image(image_path)
            
            # Extract metadata
            metadata = {}
            suspicious_fields = []
            
            if hasattr(img, 'info'):
                metadata = dict(img.info)
                
                # Analyze metadata for suspicious content
                for key, value in metadata.items():
                    # Check for unusually large metadata fields
                    if isinstance(value, (bytes, str)) and len(str(value)) > 100:
                        suspicious_fields.append({
                            "field": key,
                            "reason": "unusually_large",
                            "size": len(str(value))
                        })
                    
                    # Check for base64-encoded data
                    if isinstance(value, str) and self._looks_like_base64(value):
                        suspicious_fields.append({
                            "field": key,
                            "reason": "possible_base64"
                        })
                        
                    # Check for hidden text in metadata
                    if isinstance(value, str) and len(value) > 10:
                        hidden_text = self._extract_hidden_text(value)
                        if hidden_text:
                            suspicious_fields.append({
                                "field": key,
                                "reason": "contains_text",
                                "text": hidden_text
                            })
            
            # Save any exif data specifically
            exif_data = self._extract_exif(img)
            if exif_data:
                metadata["EXIF"] = exif_data
            
            return (len(suspicious_fields) > 0, {
                "metadata": metadata,
                "suspicious_fields": suspicious_fields,
                "extracted_data": suspicious_fields[0].get("text") if suspicious_fields else None
            })
            
        except Exception as e:
            self.logger.error(f"Error in metadata analysis: {e}")
            return (False, {"error": str(e)})
    
    def _extract_exif(self, img: Image.Image) -> Dict[str, Any]:
        """
        Extract EXIF data from an image.
        
        Args:
            img: PIL Image object
            
        Returns:
            Dictionary containing EXIF data
        """
        exif_data = {}
        
        try:
            if hasattr(img, '_getexif') and callable(img._getexif):
                exif = img._getexif()
                if exif:
                    for tag, value in exif.items():
                        # Convert tag to string if possible
                        tag_name = str(tag)
                        
                        # Handle different EXIF value types
                        if isinstance(value, bytes):
                            try:
                                # Try to decode as ASCII
                                value = value.decode('ascii', errors='replace')
                            except:
                                value = base64.b64encode(value).decode('ascii')
                        elif isinstance(value, IFDRational):
                            # Convert IFDRational to a string representation
                            value = f"{float(value):.3f}"
                        elif isinstance(value, tuple):
                            # Handle tuple values (like GPS coordinates)
                            value = [
                                float(v) if isinstance(v, IFDRational)
                                else str(v)
                                for v in value
                            ]
                        elif isinstance(value, dict):
                            # Handle nested dictionaries
                            value = {
                                str(k): (
                                    float(v) if isinstance(v, IFDRational)
                                    else str(v) if isinstance(v, bytes)
                                    else v
                                )
                                for k, v in value.items()
                            }
                        else:
                            # Convert other types to string
                            value = str(value)
                        
                        exif_data[tag_name] = value
        except Exception as e:
            self.logger.warning(f"Error extracting EXIF: {e}")
            
        return exif_data
    
    def _looks_like_base64(self, s: str) -> bool:
        """
        Check if a string looks like Base64-encoded data.
        
        Args:
            s: String to check
            
        Returns:
            Boolean indicating if string looks like Base64
        """
        import re
        
        # Base64 typically consists of alphanumeric characters, +, /, and possibly = at the end
        pattern = r'^[A-Za-z0-9+/]+={0,2}$'
        
        # String should be reasonably long and match the pattern
        return len(s) > 20 and bool(re.match(pattern, s))
    
    def _extract_hidden_text(self, text: str) -> Optional[str]:
        """
        Extract hidden readable text from a string.
        
        Args:
            text: String to analyze
            
        Returns:
            Extracted text if found, None otherwise
        """
        import re
        
        # Look for sequences of readable ASCII characters
        readable_parts = re.findall(r'[A-Za-z0-9\s.,!?:;(){}\[\]\'\"]{5,}', text)
        
        if readable_parts:
            # Join and return parts
            return ' '.join(readable_parts)
            
        return None


class KeywordXorStrategy(StegStrategy):
    """
    Strategy that applies XOR operations with specific keywords 
    to detect hidden data.
    
    This strategy tries XORing image data with keywords from our
    investigation to see if any meaningful data emerges.
    """
    
    name: str = "keyword_xor_strategy"
    description: str = "XOR operation with key investigation terms"
    
    # Key terms to try as XOR keys
    KEY_TERMS = [
        "4NBT",
        "silver",
        "YZY",
        "4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump",
    ]
    
    # Key numbers to try as shifts or in other operations
    KEY_NUMBERS = [4, 333, 353]
    
    def __init__(self) -> None:
        """Initialize the strategy."""
        super().__init__()
        # Don't hardcode the output directory, use the one provided by the parent class
        # The output_dir will be set properly by StegAnalysisPipeline when the strategy is used
    
    def analyze(self, image_path: Path) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Analyze an image by applying XOR with key terms.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple containing:
                - Boolean indicating if hidden data was detected
                - Optional dictionary with extracted data and analysis metadata
        """
        try:
            img = self.load_image(image_path)
            img_array = np.array(img)
            
            # Flatten image data for processing
            if len(img_array.shape) == 3:  # Color image
                flat_data = img_array.reshape(-1)
            else:  # Grayscale image
                flat_data = img_array.reshape(-1)
                
            results = {}
            found_something = False
            genuine_findings = False  # Track if we found any genuine (non-artifact) findings
            
            # Try each key term
            for term in self.KEY_TERMS:
                self.logger.debug(f"Trying XOR with key: {term}")
                term_bytes = term.encode('utf-8')
                
                # Apply XOR with term
                xor_result = bytearray()
                for i in range(0, len(flat_data), len(term_bytes)):
                    for j in range(len(term_bytes)):
                        if i + j < len(flat_data):
                            xor_result.append(flat_data[i + j] ^ term_bytes[j % len(term_bytes)])
                
                # Try to interpret as text
                try:
                    xor_text = xor_result.decode('ascii', errors='ignore')
                    
                    # Check if result contains readable text or key phrases
                    # Focus on ASCII printable characters
                    printable_chars = sum(1 for c in xor_text if 32 <= ord(c) <= 126)
                    if printable_chars / len(xor_text) > 0.7:  # 70% threshold
                        results[f"xor_{term}"] = {
                            "readable_text": True,
                            "data": xor_text[:1000]  # Limit output size
                        }
                        found_something = True
                        
                    # Look for our specific target string
                    if "4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump" in xor_text:
                        target_pos = xor_text.find("4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump")
                        context_start = max(0, target_pos - 100)
                        context_end = min(len(xor_text), target_pos + len("4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump") + 200)
                        context = xor_text[context_start:context_end]
                        
                        # Enhanced check for false positives
                        is_false_positive = False
                        
                        # Case 1: Using the target string itself as the key
                        if term == "4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump":
                            # If we're using the target string as the key, check if the corresponding bytes in the image are zeros
                            block_position = target_pos // len(term_bytes) * len(term_bytes)
                            
                            # Check if this region is all zeros or has consistent values
                            value_counts = {}
                            for i in range(len(term_bytes)):
                                if block_position + i < len(flat_data):
                                    val = flat_data[block_position + i]
                                    value_counts[val] = value_counts.get(val, 0) + 1
                            
                            # If region is all zeros or has very few unique values, it's likely a false positive
                            if len(value_counts) <= 3 or 0 in value_counts:
                                is_false_positive = True
                                self.logger.info(f"Target string found, but it's due to XORing with a region of consistent values at position {block_position}")
                        
                        # Case 2: Check if the target string appears in an analysis header or metadata
                        header_indicators = [
                            "=== DECODED CONTENT FROM",
                            "WITH TARGET STRING HIGHLIGHTED",
                            "USING XOR KEY:",
                            "TARGET STRING:",
                            "FOUND AT POSITION:",
                            "CONTEXT AROUND"
                        ]
                        
                        # Check if any header indicator appears close to the target string
                        surrounding_text = xor_text[max(0, target_pos - 200):min(len(xor_text), target_pos + 200)]
                        if any(indicator in surrounding_text for indicator in header_indicators):
                            is_false_positive = True
                            self.logger.info(f"Target string found in what appears to be a header or metadata section")
                        
                        # Case 3: Check for other consistent patterns that would cause false positives
                        if not is_false_positive:
                            # Examine the area around the target
                            window_size = 100
                            start_idx = max(0, target_pos - window_size)
                            end_idx = min(len(xor_text), target_pos + len("4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump") + window_size)
                            
                            # Check the character distribution around the target
                            char_counts = {}
                            for i in range(start_idx, end_idx):
                                c = xor_text[i]
                                char_counts[c] = char_counts.get(c, 0) + 1
                            
                            # If character distribution is unusual (very few unique chars), likely a false positive
                            if len(char_counts) < 10:  # Very low diversity of characters
                                is_false_positive = True
                                self.logger.info(f"Target string found in a region with suspicious character distribution")
                        
                        if not is_false_positive:
                            results[f"xor_{term}_found_key"] = {
                                "found_target": True,
                                "context": context,
                                "target_position": target_pos,
                                "full_data": xor_text  # Store the full decoded text
                            }
                            found_something = True
                            genuine_findings = True
                            self.logger.warning(f"FOUND TARGET STRING with XOR key {term}!")
                            
                            # Save the full decoded content to a file
                            # Get the proper output directory
                            output_dir = self.get_strategy_output_dir(image_path)
                            output_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Get unique image identifier using parent directory if available
                            parent_dir = image_path.parent.name
                            unique_image_id = f"{parent_dir}_{image_path.stem}" if parent_dir else image_path.stem
                            
                            # Save full decoded content
                            decoded_file = output_dir / f"decoded_with_{term}.txt"
                            with open(decoded_file, "w", encoding="utf-8") as f:
                                f.write(f"=== DECODED CONTENT FROM {image_path.name} USING XOR KEY: {term} ===\n\n")
                                f.write(xor_text)
                                
                            # Save a highlighted version showing where target was found
                            highlight_file = output_dir / f"decoded_with_{term}_highlighted.txt"
                            with open(highlight_file, "w", encoding="utf-8") as f:
                                f.write(f"=== DECODED CONTENT FROM {image_path.name} WITH TARGET STRING HIGHLIGHTED ===\n\n")
                                f.write(f"TARGET STRING: 4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump\n")
                                f.write(f"FOUND AT POSITION: {target_pos}\n\n")
                                f.write(f"CONTEXT AROUND TARGET STRING:\n")
                                f.write("-" * 80 + "\n")
                                f.write(context)
                                f.write("\n" + "-" * 80 + "\n\n")
                                f.write("FULL DECODED CONTENT:\n")
                                f.write(xor_text)
                            
                            self.logger.info(f"Saved decoded content to {decoded_file}")
                            self.logger.info(f"Saved highlighted content to {highlight_file}")
                        else:
                            results[f"xor_{term}_false_positive"] = {
                                "false_positive": True,
                                "reason": "Target string found due to artifacts in the data",
                                "position": target_pos
                            }
                            
                    # Also look for '4NBT' or 'silver' phrases which might be significant
                    if "4NBT" in xor_text or "silver" in xor_text:
                        # Enhanced check for false positives
                        false_positive = False
                        
                        # Check if these terms appear in what looks like headers
                        header_indicators = [
                            "=== DECODED CONTENT",
                            "USING XOR KEY:",
                            "TARGET STRING:",
                            "FOUND AT POSITION:"
                        ]
                        
                        # First, find all occurrences of the terms
                        term_to_check = "4NBT" if "4NBT" in xor_text else "silver"
                        positions = [m.start() for m in re.finditer(term_to_check, xor_text)]
                        
                        for pos in positions:
                            # Check surrounding context
                            surrounding = xor_text[max(0, pos - 100):min(len(xor_text), pos + 100)]
                            
                            # If it appears in a header, mark as false positive
                            if any(indicator in surrounding for indicator in header_indicators):
                                false_positive = True
                                self.logger.info(f"'{term_to_check}' found in what appears to be a header or metadata section")
                                break
                            
                            # If the term is part of the key, also check for regions of zeros or consistent values
                            if term_to_check in term:
                                # Calculate corresponding position in the original data
                                block_pos = pos // len(term_bytes) * len(term_bytes)
                                value_counts = {}
                                for i in range(len(term_to_check)):
                                    if block_pos + i < len(flat_data):
                                        val = flat_data[block_pos + i]
                                        value_counts[val] = value_counts.get(val, 0) + 1
                                
                                # If consistent values or zeros, likely a false positive
                                if len(value_counts) <= 2 or 0 in value_counts:
                                    false_positive = True
                                    self.logger.info(f"'{term_to_check}' found, but it's due to XORing with a region of consistent values")
                                    break
                        
                        if not false_positive:
                            results[f"xor_{term}_found_clue"] = {
                                "found_clue": True,
                                "clue": term_to_check,
                                "context": xor_text
                            }
                            found_something = True
                            genuine_findings = True
                        
                except Exception as e:
                    self.logger.debug(f"Error decoding XOR result for {term}: {e}")
                    
            # Also try XOR with numeric keys
            for num in self.KEY_NUMBERS:
                xor_result = bytearray()
                for i in range(len(flat_data)):
                    xor_result.append(flat_data[i] ^ (num % 256))
                
                try:
                    xor_text = xor_result.decode('ascii', errors='ignore')
                    
                    # Check if result contains readable text
                    printable_chars = sum(1 for c in xor_text if 32 <= ord(c) <= 126)
                    if printable_chars / len(xor_text) > 0.7:  # 70% threshold
                        results[f"xor_num_{num}"] = {
                            "readable_text": True,
                            "data": xor_text[:1000]  # Limit output size
                        }
                        found_something = True
                        
                    # Check for target string
                    if "4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump" in xor_text:
                        target_pos = xor_text.find("4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump")
                        
                        # Enhanced check for false positives
                        is_false_positive = False
                        
                        # Check if the found string is just due to XORing with zeros or consistent values
                        if num == 0:  # If the key is 0, any data would remain unchanged
                            is_false_positive = True
                            self.logger.info("Target string found, but XOR key is 0 which doesn't change the data")
                        else:
                            # Check if the corresponding bytes in the original data have consistent values
                            start_pos = target_pos
                            value_counts = {}
                            for i in range(len("4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump")):
                                if start_pos + i < len(flat_data):
                                    val = flat_data[start_pos + i]
                                    value_counts[val] = value_counts.get(val, 0) + 1
                            
                            # If too few unique values or specific patterns, it's likely a false positive
                            if len(value_counts) <= 3 or 0 in value_counts or num in value_counts:
                                is_false_positive = True
                                self.logger.info(f"Target string found, but it's due to XORing with a region of consistent values")
                        
                        # Also check for header/metadata contexts as with term keys
                        if not is_false_positive:
                            header_indicators = [
                                "=== DECODED CONTENT FROM",
                                "WITH TARGET STRING HIGHLIGHTED",
                                "USING XOR KEY:",
                                "TARGET STRING:",
                                "FOUND AT POSITION:",
                                "CONTEXT AROUND"
                            ]
                            
                            surrounding_text = xor_text[max(0, target_pos - 200):min(len(xor_text), target_pos + 200)]
                            if any(indicator in surrounding_text for indicator in header_indicators):
                                is_false_positive = True
                                self.logger.info(f"Target string found in what appears to be a header or metadata section")
                        
                        if not is_false_positive:
                            context_start = max(0, target_pos - 100)
                            context_end = min(len(xor_text), target_pos + len("4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump") + 200)
                            context = xor_text[context_start:context_end]
                            
                            results[f"xor_num_{num}_found_key"] = {
                                "found_target": True,
                                "context": context,
                                "target_position": target_pos,
                                "full_data": xor_text  # Store the full decoded text
                            }
                            found_something = True
                            genuine_findings = True
                            self.logger.warning(f"FOUND TARGET STRING with XOR number {num}!")
                            
                            # Save the full decoded content to a file
                            # Get the proper output directory
                            output_dir = self.get_strategy_output_dir(image_path)
                            output_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Get unique image identifier using parent directory if available
                            parent_dir = image_path.parent.name
                            unique_image_id = f"{parent_dir}_{image_path.stem}" if parent_dir else image_path.stem
                            
                            # Save full decoded content
                            decoded_file = output_dir / f"decoded_with_num_{num}.txt"
                            with open(decoded_file, "w", encoding="utf-8") as f:
                                f.write(f"=== DECODED CONTENT FROM {image_path.name} USING XOR NUMBER: {num} ===\n\n")
                                f.write(xor_text)
                                
                            # Save a highlighted version showing where target was found
                            highlight_file = output_dir / f"decoded_with_num_{num}_highlighted.txt"
                            with open(highlight_file, "w", encoding="utf-8") as f:
                                f.write(f"=== DECODED CONTENT FROM {image_path.name} WITH TARGET STRING HIGHLIGHTED ===\n\n")
                                f.write(f"TARGET STRING: 4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump\n")
                                f.write(f"FOUND AT POSITION: {target_pos}\n\n")
                                f.write(f"CONTEXT AROUND TARGET STRING:\n")
                                f.write("-" * 80 + "\n")
                                f.write(context)
                                f.write("\n" + "-" * 80 + "\n\n")
                                f.write("FULL DECODED CONTENT:\n")
                                f.write(xor_text)
                            
                            self.logger.info(f"Saved decoded content to {decoded_file}")
                            self.logger.info(f"Saved highlighted content to {highlight_file}")
                        else:
                            results[f"xor_num_{num}_false_positive"] = {
                                "false_positive": True,
                                "reason": f"Target string found due to artifacts in the data",
                                "position": target_pos
                            }
                            
                except Exception as e:
                    self.logger.debug(f"Error decoding XOR result for number {num}: {e}")
            
            # Only return true if we found genuine findings, not just false positives
            return (genuine_findings, results if found_something else None)
            
        except Exception as e:
            self.logger.error(f"Error in keyword XOR analysis: {e}")
            return (False, {"error": str(e)})


class ShiftCipherStrategy(StegStrategy):
    """
    Strategy to detect and extract hidden data using shift cipher techniques.
    
    This strategy applies various character shifts (like Caesar cipher)
    to image data to try to reveal hidden text.
    """
    
    name: str = "shift_cipher_strategy"
    description: str = "Shift-based ciphers and Caesar cipher variants"
    
    def __init__(self) -> None:
        """Initialize the strategy."""
        super().__init__()
    
    def analyze(self, image_path: Path) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Analyze an image using shift cipher techniques.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple containing:
                - Boolean indicating if hidden data was detected
                - Optional dictionary with extracted data and analysis metadata
        """
        try:
            img = self.load_image(image_path)
            img_array = np.array(img)
            
            # Extract data from image using LSB first
            # as a source of potential encrypted text
            raw_data = self._extract_lsb_data(img_array)
            
            # Try to decode raw data as ASCII first
            try:
                text_data = raw_data.decode('ascii', errors='ignore')
            except:
                text_data = ''.join(chr(b) for b in raw_data if 32 <= b <= 126)
            
            results = {}
            found_something = False
            
            # Try each shift value, focusing on our key numbers
            shifts_to_try = list(range(1, 26)) + [4, 333 % 26, 353 % 26]
            shifts_to_try = sorted(list(set(shifts_to_try)))  # Remove duplicates
            
            for shift in shifts_to_try:
                shifted_text = self._apply_shift(text_data, shift)
                
                # Check if result contains meaningful data
                if self._is_readable_text(shifted_text):
                    results[f"shift_{shift}"] = {
                        "readable_text": True,
                        "data": shifted_text[:1000]  # Limit output size
                    }
                    found_something = True
                
                # Check for target string
                if "4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump" in shifted_text:
                    results[f"shift_{shift}_found_key"] = {
                        "found_target": True,
                        "context": shifted_text[max(0, shifted_text.find("4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump")-20):
                                          min(len(shifted_text), shifted_text.find("4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump")+80)]
                    }
                    found_something = True
                    self.logger.warning(f"FOUND TARGET STRING with shift {shift}!")
                    
                # Also look for '4NBT' or 'silver'
                if "4NBT" in shifted_text or "silver" in shifted_text:
                    results[f"shift_{shift}_found_clue"] = {
                        "found_clue": True,
                        "clue": "4NBT" if "4NBT" in shifted_text else "silver",
                        "context": shifted_text[:500]
                    }
                    found_something = True
            
            return (found_something, results if found_something else None)
            
        except Exception as e:
            self.logger.error(f"Error in shift cipher analysis: {e}")
            return (False, {"error": str(e)})
    
    def _extract_lsb_data(self, img_array: np.ndarray) -> bytearray:
        """
        Extract LSB data from image for further analysis.
        
        Args:
            img_array: Numpy array representation of the image
            
        Returns:
            Bytearray of extracted LSB data
        """
        if len(img_array.shape) == 3:  # Color image
            height, width, channels = img_array.shape
            bit_array = np.zeros(height * width * channels, dtype=np.uint8)
            
            index = 0
            for h in range(height):
                for w in range(width):
                    for c in range(channels):
                        bit_array[index] = img_array[h, w, c] & 1
                        index += 1
        else:  # Grayscale image
            height, width = img_array.shape
            bit_array = np.zeros(height * width, dtype=np.uint8)
            
            index = 0
            for h in range(height):
                for w in range(width):
                    bit_array[index] = img_array[h, w] & 1
                    index += 1
        
        # Convert bits to bytes
        byte_array = bytearray()
        for i in range(0, len(bit_array) - 7, 8):
            byte_val = 0
            for j in range(8):
                if i + j < len(bit_array):
                    byte_val |= bit_array[i + j] << j
            byte_array.append(byte_val)
            
        return byte_array
    
    def _apply_shift(self, text: str, shift: int) -> str:
        """
        Apply a Caesar-like shift to text.
        
        Args:
            text: Text to shift
            shift: Number of positions to shift
            
        Returns:
            Shifted text
        """
        result = ""
        for char in text:
            if 'a' <= char <= 'z':
                # Shift lowercase letters
                result += chr((ord(char) - ord('a') + shift) % 26 + ord('a'))
            elif 'A' <= char <= 'Z':
                # Shift uppercase letters
                result += chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
            else:
                # Keep non-alphabetic characters as is
                result += char
        return result
    
    def _is_readable_text(self, text: str) -> bool:
        """
        Check if text appears to be readable.
        
        Args:
            text: Text to check
            
        Returns:
            Boolean indicating if text is likely readable
        """
        # Simple heuristic: count spaces, common words
        space_ratio = text.count(' ') / len(text) if len(text) > 0 else 0
        common_words = ['the', 'be', 'to', 'of', 'and', 'in', 'that', 'have', 'it', 'for']
        word_count = sum(1 for word in common_words if word in text.lower())
        
        return space_ratio > 0.1 and word_count >= 2


class BitSequenceStrategy(StegStrategy):
    """
    Strategy to detect patterns in the bit sequences of an image.
    
    This strategy looks for meaningful patterns in the bit representation
    of pixel values, trying different arrangements and representations.
    """
    
    name: str = "bit_sequence_strategy"
    description: str = "Analysis of bit patterns and sequences"
    
    def __init__(self) -> None:
        """Initialize the strategy."""
        super().__init__()
    
    def analyze(self, image_path: Path) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Analyze an image for bit patterns.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple containing:
                - Boolean indicating if hidden data was detected
                - Optional dictionary with extracted data and analysis metadata
        """
        try:
            img = self.load_image(image_path)
            img_array = np.array(img)
            
            results = {}
            found_something = False
            
            # Extract a sequence of bits from the image
            bit_sequence = self._extract_bit_sequence(img_array)
            binary_string = ''.join(str(bit) for bit in bit_sequence)
            
            # Look for repeating patterns
            patterns = self._find_repeating_patterns(binary_string)
            if patterns:
                results["repeating_patterns"] = patterns
                found_something = True
            
            # Try different bit-to-byte conversions
            for bits_per_group in [7, 8]:
                # Convert bits to ASCII with MSB first
                msb_text = self._bits_to_text(bit_sequence, bits_per_group, msb_first=True)
                if self._is_interesting_text(msb_text):
                    results[f"msb_{bits_per_group}bit"] = {
                        "readable": True,
                        "text": msb_text[:1000]
                    }
                    found_something = True
                
                # Convert bits to ASCII with LSB first
                lsb_text = self._bits_to_text(bit_sequence, bits_per_group, msb_first=False)
                if self._is_interesting_text(lsb_text):
                    results[f"lsb_{bits_per_group}bit"] = {
                        "readable": True,
                        "text": lsb_text[:1000]
                    }
                    found_something = True
            
            # Try bit rearrangements based on key numbers
            for arrangement in [(4, 8), (333 % 20, 5), (353 % 20, 5)]:
                rows, cols = arrangement
                if len(bit_sequence) >= rows * cols:
                    # Rearrange bits into grid
                    grid = np.array(bit_sequence[:rows*cols]).reshape(rows, cols)
                    
                    # Read across rows
                    row_bits = grid.reshape(-1)
                    row_text = self._bits_to_text(row_bits, 8, msb_first=True)
                    
                    if self._is_interesting_text(row_text):
                        results[f"arrangement_{rows}x{cols}_rows"] = {
                            "readable": True,
                            "text": row_text[:1000]
                        }
                        found_something = True
                    
                    # Read down columns
                    col_bits = grid.T.reshape(-1)
                    col_text = self._bits_to_text(col_bits, 8, msb_first=True)
                    
                    if self._is_interesting_text(col_text):
                        results[f"arrangement_{rows}x{cols}_cols"] = {
                            "readable": True,
                            "text": col_text[:1000]
                        }
                        found_something = True
            
            return (found_something, results if found_something else None)
            
        except Exception as e:
            self.logger.error(f"Error in bit sequence analysis: {e}")
            return (False, {"error": str(e)})
    
    def _extract_bit_sequence(self, img_array: np.ndarray) -> List[int]:
        """
        Extract a sequence of bits from the image.
        
        Args:
            img_array: Numpy array representation of the image
            
        Returns:
            List of bits extracted from the image
        """
        # Extract LSBs and some regular bit patterns
        bits = []
        
        # Flatten the array appropriately
        if len(img_array.shape) == 3:  # Color image
            flat_array = img_array.reshape(-1)
        else:  # Grayscale image
            flat_array = img_array.reshape(-1)
        
        # Extract LSBs
        for pixel in flat_array[:min(len(flat_array), 10000)]:  # Limit to reasonable size
            bits.append(pixel & 1)
        
        return bits
    
    def _find_repeating_patterns(self, binary_string: str) -> List[Dict[str, Any]]:
        """
        Find repeating bit patterns in the binary string.
        
        Args:
            binary_string: String of binary digits
            
        Returns:
            List of detected patterns with their information
        """
        patterns = []
        
        # Look for repeated sequences of various lengths
        for length in range(4, 21):  # Patterns of length 4 to 20
            # Count occurrences of each pattern
            pattern_counts = {}
            for i in range(len(binary_string) - length + 1):
                pattern = binary_string[i:i+length]
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            # Find patterns that repeat more than expected
            significant_patterns = {}
            expected_count = len(binary_string) / (2 ** length)
            for pattern, count in pattern_counts.items():
                if count > max(3, expected_count * 1.5):  # At least 50% more than expected
                    significant_patterns[pattern] = count
            
            if significant_patterns:
                # Sort by count (most frequent first)
                sorted_patterns = sorted(significant_patterns.items(), key=lambda x: x[1], reverse=True)
                patterns.append({
                    "length": length,
                    "top_patterns": [{
                        "pattern": p,
                        "count": c,
                        "positions": [m.start() for m in re.finditer(f'(?={p})', binary_string)][:10]  # First 10 positions
                    } for p, c in sorted_patterns[:5]]  # Top 5 patterns
                })
        
        return patterns
    
    def _bits_to_text(self, bits: List[int], bits_per_group: int, msb_first: bool = True) -> str:
        """
        Convert a list of bits to ASCII text.
        
        Args:
            bits: List of binary digits
            bits_per_group: Number of bits to group together (7 or 8)
            msb_first: Whether to interpret bits with MSB first
            
        Returns:
            ASCII text representation
        """
        text = ""
        for i in range(0, len(bits) - bits_per_group + 1, bits_per_group):
            char_bits = bits[i:i+bits_per_group]
            char_val = 0
            
            if msb_first:
                # MSB first
                for j in range(bits_per_group):
                    char_val = (char_val << 1) | char_bits[j]
            else:
                # LSB first
                for j in range(bits_per_group):
                    char_val |= char_bits[j] << j
            
            # Convert to ASCII character if printable
            if 32 <= char_val <= 126:
                text += chr(char_val)
            else:
                text += '.'
        
        return text
    
    def _is_interesting_text(self, text: str) -> bool:
        """
        Check if text appears interesting or meaningful.
        
        Args:
            text: Text to check
            
        Returns:
            Boolean indicating if text is likely interesting
        """
        # Look for target markers
        if "4NBT" in text or "silver" in text:
            return True
            
        if "4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump" in text:
            self.logger.warning("FOUND TARGET STRING in bit sequence data!")
            return True
        
        # Check for readable text characteristics
        printable_ratio = sum(1 for c in text if 32 <= ord(c) <= 126) / len(text) if len(text) > 0 else 0
        alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text) if len(text) > 0 else 0
        space_ratio = text.count(' ') / len(text) if len(text) > 0 else 0
        
        # Look for words
        common_words = ['the', 'be', 'to', 'of', 'and', 'in', 'that', 'have', 'it', 'for']
        word_count = sum(1 for word in common_words if word in text.lower())
        
        return (printable_ratio > 0.8 and alpha_ratio > 0.6) or (space_ratio > 0.1 and word_count >= 2)


class BlakeHashStrategy(StegStrategy):
    """
    Strategy that uses Blake hash functions to analyze image data.
    
    This strategy explores the connection between William Blake (the poet/artist)
    and Blake hash functions, which might be relevant to decoding hidden data.
    It tries various permutations of pixel data with Blake2b and Blake2s hashes.
    """
    
    name: str = "blake_hash_strategy"
    description: str = "Blake hash-based steganography detection"
    
    # Blake poems/works that might be used as keys
    BLAKE_WORKS = [
        "Tyger",
        "TheTyger",
        "LambOfGod",
        "JerusalemTheEmanation",
        "SongsOfInnocence",
        "SongsOfExperience",
        "TheMarriageOfHeavenAndHell",
        "AuguriesOfInnocence",
        "LondonWilliam"
    ]
    
    # Common phrases from William Blake's works
    BLAKE_PHRASES = [
        "fearful symmetry",
        "burning bright",
        "tiger tiger",
        "heaven and hell",
        "lamb of god",
        "little lamb",
        "jerusalem",
        "albion",
        "fearful symmetry"
    ]
    
    # Key numbers associated with the project
    KEY_NUMBERS = [4, 333, 353]
    
    def __init__(self) -> None:
        """Initialize the strategy."""
        super().__init__()
        
        # Check if pyblake2 is available, otherwise use hashlib (Python 3.6+)
        try:
            import pyblake2
            self.blake2b = pyblake2.blake2b
            self.blake2s = pyblake2.blake2s
            self.logger.info("Using pyblake2 for Blake hash functions")
        except ImportError:
            # Fall back to hashlib for Blake2 (Python 3.6+)
            if hasattr(hashlib, 'blake2b'):
                self.blake2b = hashlib.blake2b
                self.blake2s = hashlib.blake2s
                self.logger.info("Using hashlib for Blake hash functions")
            else:
                self.logger.error("Neither pyblake2 nor hashlib.blake2b are available")
                raise ImportError("Blake2 hash functions not available")
    
    def analyze(self, image_path: Path) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Analyze an image using Blake hash functions.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple containing:
                - Boolean indicating if hidden data was detected
                - Optional dictionary with extracted data and analysis metadata
        """
        try:
            img = self.load_image(image_path)
            img_array = np.array(img)
            
            results = {}
            found_something = False
            
            # Extract basic data from image
            flat_data = self._extract_image_data(img_array)
            
            # 1. Try Blake hashes with various keys derived from William Blake
            blake_key_results = self._try_blake_hash_with_keys(flat_data)
            if blake_key_results:
                results["blake_key_results"] = blake_key_results
                found_something = True
            
            # 2. Try analyzing image regions with Blake hashes
            region_results = self._analyze_image_regions(img_array)
            if region_results:
                results["region_analysis"] = region_results
                found_something = True
            
            # 3. Try Blake hash with key numbers
            number_results = self._try_hash_with_numbers(flat_data)
            if number_results:
                results["number_hash_results"] = number_results
                found_something = True
            
            # 4. Try to extract a potential message using LSBs and Blake verification
            lsb_blake_results = self._analyze_lsb_with_blake(img_array)
            if lsb_blake_results:
                results["lsb_blake_results"] = lsb_blake_results
                found_something = True
            
            return (found_something, results if found_something else None)
            
        except Exception as e:
            self.logger.error(f"Error in Blake hash analysis: {e}")
            return (False, {"error": str(e)})
    
    def _extract_image_data(self, img_array: np.ndarray) -> bytes:
        """
        Extract serialized data from image array for hashing.
        
        Args:
            img_array: Numpy array of image data
            
        Returns:
            Bytes of serialized image data
        """
        # If color image, process each channel
        if len(img_array.shape) == 3:
            # Take the least significant bits, as they're most likely to contain hidden data
            lsb_data = (img_array & 1).reshape(-1).tobytes()
            return lsb_data
        else:
            # For grayscale, just take raw bytes
            return img_array.tobytes()
    
    def _try_blake_hash_with_keys(self, data: bytes) -> Optional[Dict[str, Any]]:
        """
        Try Blake hashing with William Blake-related keys.
        
        Args:
            data: Image data to analyze
            
        Returns:
            Dictionary with results or None if nothing found
        """
        results = {}
        found_something = False
        
        # Try each Blake work as a key for Blake2b and Blake2s
        for work in self.BLAKE_WORKS:
            key = work.encode('utf-8')
            # Blake2b
            b2b_hash = self.blake2b(data, key=key).hexdigest()
            b2s_hash = self.blake2s(data, key=key).hexdigest()
            
            # Check if the hash contains our target patterns
            b2b_result = self._check_hash_significance(b2b_hash, f"Blake2b with key '{work}'")
            b2s_result = self._check_hash_significance(b2s_hash, f"Blake2s with key '{work}'")
            
            if b2b_result:
                results[f"blake2b_{work}"] = b2b_result
                found_something = True
            
            if b2s_result:
                results[f"blake2s_{work}"] = b2s_result
                found_something = True
        
        # Also try Blake phrases
        for phrase in self.BLAKE_PHRASES:
            key = phrase.encode('utf-8')
            b2b_hash = self.blake2b(data, key=key).hexdigest()
            
            # Check if the hash contains our target patterns
            result = self._check_hash_significance(b2b_hash, f"Blake2b with phrase '{phrase}'")
            if result:
                results[f"blake2b_phrase_{phrase}"] = result
                found_something = True
        
        return results if found_something else None
    
    def _check_hash_significance(self, hash_str: str, source: str) -> Optional[Dict[str, Any]]:
        """
        Check if a hash contains significant patterns related to our investigation.
        
        Args:
            hash_str: Hexadecimal hash string to check
            source: Description of where this hash came from
            
        Returns:
            Dictionary with significance details or None
        """
        # Convert hash to ASCII by interpreting hex pairs as ASCII values
        ascii_text = ""
        try:
            # Take pairs of hex chars and convert to ASCII if in printable range
            for i in range(0, len(hash_str), 2):
                hex_pair = hash_str[i:i+2]
                byte_val = int(hex_pair, 16)
                if 32 <= byte_val <= 126:  # Printable ASCII range
                    ascii_text += chr(byte_val)
                else:
                    ascii_text += '.'
        except:
            ascii_text = ""
        
        # Look for target patterns in both hex and potential ASCII
        pattern_found = False
        significance = "low"
        findings = []
        
        # Check for our target signature
        target = "4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump"
        if target in hash_str or target in ascii_text:
            pattern_found = True
            significance = "high"
            findings.append({
                "type": "target_string",
                "pattern": target,
                "context": f"Found target string in {source}"
            })
            self.logger.warning(f"TARGET STRING found in {source}")
        
        # Check for key terms
        for term in ["4NBT", "silver"]:
            if term in hash_str or term in ascii_text:
                pattern_found = True
                significance = "medium" if significance != "high" else significance
                findings.append({
                    "type": "key_term",
                    "pattern": term,
                    "context": f"Found key term '{term}' in {source}"
                })
                self.logger.info(f"Key term '{term}' found in {source}")
        
        # Check for key numbers - only report if they appear distinctly (not just as a single digit)
        for num in self.KEY_NUMBERS:
            num_str = str(num)
            # Skip single-digit numbers to avoid excessive false positives
            if len(num_str) == 1:
                # Only check for single digits if they appear with special boundaries
                # This reduces false positives for common digits like "4"
                patterns = [f"_{num_str}_", f":{num_str}:", f".{num_str}.", f",{num_str},", f";{num_str};"]
                found = any(pattern in hash_str for pattern in patterns)
            else:
                # For multi-digit numbers, require they appear as distinct numbers
                # Either at start/end of hash or with non-numeric boundaries
                boundary_patterns = [
                    f"^{num_str}", f"{num_str}$",  # Start/end of string
                    f"[^0-9]{num_str}[^0-9]"       # Non-numeric boundaries
                ]
                found = any(re.search(pattern, hash_str) for pattern in boundary_patterns)
                
            if found:
                pattern_found = True
                findings.append({
                    "type": "key_number",
                    "pattern": num_str,
                    "context": f"Found key number {num_str} in {source}"
                })
        
        if pattern_found:
            return {
                "hash": hash_str,
                "source": source,
                "significance": significance,
                "ascii_interpretation": ascii_text,
                "findings": findings
            }
        
        return None
    
    def _analyze_image_regions(self, img_array: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Analyze different regions of the image with Blake hashes.
        
        Args:
            img_array: Numpy array of image data
            
        Returns:
            Dictionary with region analysis results or None
        """
        results = {}
        found_something = False
        
        # Get dimensions
        if len(img_array.shape) == 3:
            height, width, _ = img_array.shape
        else:
            height, width = img_array.shape
        
        # Define regions to analyze
        regions = [
            ("top_half", (0, height//2, 0, width)),
            ("bottom_half", (height//2, height, 0, width)),
            ("left_half", (0, height, 0, width//2)),
            ("right_half", (0, height, width//2, width)),
            ("center", (height//4, 3*height//4, width//4, 3*width//4))
        ]
        
        # Also try the key numbers for region sizes
        for num in self.KEY_NUMBERS:
            size = min(height, width, num)
            if size > 4:  # Ensure region is large enough
                regions.append(
                    (f"key_num_{num}", 
                     (0, min(size, height), 0, min(size, width)))
                )
        
        # Analyze each region
        for name, (y1, y2, x1, x2) in regions:
            try:
                if len(img_array.shape) == 3:
                    region_data = img_array[y1:y2, x1:x2, :].tobytes()
                else:
                    region_data = img_array[y1:y2, x1:x2].tobytes()
                
                # Try Blake2b hash on this region
                region_hash = self.blake2b(region_data).hexdigest()
                
                # Check if this hash contains anything significant
                result = self._check_hash_significance(region_hash, f"Region '{name}'")
                if result:
                    results[f"region_{name}"] = result
                    found_something = True
            except Exception as e:
                self.logger.debug(f"Error analyzing region {name}: {e}")
        
        return results if found_something else None
    
    def _try_hash_with_numbers(self, data: bytes) -> Optional[Dict[str, Any]]:
        """
        Try Blake hashes with key numbers as salt or personalization.
        
        Args:
            data: Image data to analyze
            
        Returns:
            Dictionary with results or None
        """
        results = {}
        found_something = False
        
        # Try each key number
        for num in self.KEY_NUMBERS:
            # Create personalizations with the number
            salt = str(num).encode('utf-8')
            personal = salt * 4  # Repeat to fill personalization space
            
            try:
                # Blake2b with personalization
                b2b_hash = self.blake2b(data, person=personal[:16]).hexdigest()  # Max 16 bytes
                result = self._check_hash_significance(b2b_hash, f"Blake2b with personalization {num}")
                if result:
                    results[f"blake2b_person_{num}"] = result
                    found_something = True
                
                # Blake2b with salt
                b2b_hash = self.blake2b(data, salt=salt).hexdigest()
                result = self._check_hash_significance(b2b_hash, f"Blake2b with salt {num}")
                if result:
                    results[f"blake2b_salt_{num}"] = result
                    found_something = True
                
                # Blake2s with personalization (max 8 bytes)
                b2s_hash = self.blake2s(data, person=personal[:8]).hexdigest()
                result = self._check_hash_significance(b2s_hash, f"Blake2s with personalization {num}")
                if result:
                    results[f"blake2s_person_{num}"] = result
                    found_something = True
            except Exception as e:
                self.logger.debug(f"Error with number {num}: {e}")
        
        return results if found_something else None
    
    def _analyze_lsb_with_blake(self, img_array: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Extract LSBs and verify with Blake hash for potential message extraction.
        
        Args:
            img_array: Numpy array of image data
            
        Returns:
            Dictionary with results or None
        """
        results = {}
        found_something = False
        
        try:
            # Extract LSBs from image
            if len(img_array.shape) == 3:  # Color image
                height, width, channels = img_array.shape
                bit_array = np.zeros(height * width * channels, dtype=np.uint8)
                
                index = 0
                for h in range(height):
                    for w in range(width):
                        for c in range(channels):
                            bit_array[index] = img_array[h, w, c] & 1
                            index += 1
            else:  # Grayscale image
                height, width = img_array.shape
                bit_array = np.zeros(height * width, dtype=np.uint8)
                
                index = 0
                for h in range(height):
                    for w in range(width):
                        bit_array[index] = img_array[h, w] & 1
                        index += 1
            
            # Convert bits to bytes
            byte_array = bytearray()
            for i in range(0, len(bit_array) - 7, 8):
                byte_val = 0
                for j in range(8):
                    if i + j < len(bit_array):
                        byte_val |= bit_array[i + j] << j
                byte_array.append(byte_val)
            
            # Try to extract a valid message
            # The theory: Some steganography methods include a Blake hash as verification
            # Try to find this structure: [message][hash of message]
            
            for msg_length in range(16, min(len(byte_array) - 64, 1000), 8):
                # Extract potential message and hash parts
                msg_part = byte_array[:msg_length]
                hash_part = byte_array[msg_length:msg_length+64]  # Blake2b hash is 64 bytes
                
                # Calculate Blake2b hash of the message part
                calc_hash = self.blake2b(msg_part).digest()
                
                # Check if the calculated hash matches the extracted hash part
                if calc_hash == hash_part or self._hash_similarity(calc_hash, hash_part) > 0.7:
                    # Try to decode the message part as text
                    try:
                        text = msg_part.decode('utf-8', errors='ignore')
                        significance = "low"
                        
                        # Check for our target patterns
                        if "4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump" in text:
                            significance = "high"
                            self.logger.warning(f"TARGET STRING found in Blake-verified message")
                        elif "4NBT" in text or "silver" in text:
                            significance = "medium"
                            self.logger.info(f"Key term found in Blake-verified message")
                        
                        results["verified_message"] = {
                            "message": text,
                            "message_bytes": base64.b64encode(msg_part).decode('ascii'),
                            "hash": base64.b64encode(hash_part).decode('ascii'),
                            "significance": significance
                        }
                        found_something = True
                        break
                    except:
                        pass
            
            # If we didn't find a verified message, try other patterns
            if not found_something:
                # Try to identify a Blake hash in the data
                data = bytes(byte_array)
                for i in range(0, min(len(data) - 64, 1000), 8):
                    potential_hash = data[i:i+64]
                    
                    # See if this looks like a Blake hash
                    if self._is_potential_hash(potential_hash):
                        # Extract the message before it
                        msg_before = data[max(0, i-100):i]
                        try:
                            text = msg_before.decode('utf-8', errors='ignore')
                            calc_hash = self.blake2b(msg_before).digest()
                            
                            # Compare with potential hash
                            if self._hash_similarity(calc_hash, potential_hash) > 0.5:
                                results["potential_hash_message"] = {
                                    "message": text,
                                    "hash_location": i,
                                    "hash_similarity": self._hash_similarity(calc_hash, potential_hash)
                                }
                                found_something = True
                                break
                        except:
                            pass
        except Exception as e:
            self.logger.debug(f"Error in LSB analysis with Blake: {e}")
        
        return results if found_something else None
    
    def _hash_similarity(self, hash1: bytes, hash2: bytes) -> float:
        """
        Calculate similarity between two hashes.
        
        Args:
            hash1: First hash
            hash2: Second hash
            
        Returns:
            Similarity score between 0 and 1
        """
        # Ensure equal length for comparison
        length = min(len(hash1), len(hash2))
        hash1 = hash1[:length]
        hash2 = hash2[:length]
        
        # Count matching bits
        matching_bits = 0
        total_bits = length * 8
        
        for i in range(length):
            xor_result = hash1[i] ^ hash2[i]
            # Count set bits in xor_result (non-matching bits)
            for j in range(8):
                if not (xor_result & (1 << j)):
                    matching_bits += 1
        
        return matching_bits / total_bits
    
    def _is_potential_hash(self, data: bytes) -> bool:
        """
        Check if data might be a hash value.
        
        Args:
            data: Binary data to check
            
        Returns:
            True if data could be a hash, False otherwise
        """
        # Check byte distribution - hashes typically have a uniform distribution
        byte_counts = {}
        for b in data:
            byte_counts[b] = byte_counts.get(b, 0) + 1
        
        # Calculate entropy
        entropy = 0
        for count in byte_counts.values():
            p = count / len(data)
            entropy -= p * np.log2(p)
        
        # High entropy suggests random/hash-like data
        return entropy > 7.0  # Threshold close to maximum (8 bits) 


class CustomRgbEncodingStrategy(StegStrategy):
    """
    Strategy to detect and extract data encoded with a custom RGB channel bit allocation.
    
    This strategy extracts bytes hidden in pixels where:
    - 3 bits are stored in the R channel
    - 3 bits are stored in the G channel
    - 2 bits are stored in the B channel
    It also looks for metadata in the format "filename|filetype|" at the beginning.
    """
    
    name: str = "custom_rgb_encoding_strategy"
    description: str = "Custom RGB bit allocation detection (3-3-2 bits)"
    
    def analyze(self, image_path: Path) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Analyze an image for custom RGB encoded data.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple containing:
                - Boolean indicating if hidden data was detected
                - Optional dictionary with extracted data and analysis metadata
        """
        try:
            img = self.load_image(image_path)
            
            # Ensure image is RGB
            if img.mode != "RGB":
                self.logger.info(f"Image {image_path} is not RGB, converting")
                img = img.convert("RGB")
                
            img_array = np.array(img)
            height, width, _ = img_array.shape
            
            # Extract bytes from pixels using the custom bit allocation
            extracted_bytes = bytearray()
            for y in range(height):
                for x in range(width):
                    r, g, b = img_array[y, x]
                    
                    # Extract bits according to the 3-3-2 pattern
                    r_bits = (r & 0xE0) >> 5  # top 3 bits from R (bits 7-5)
                    g_bits = (g & 0xE0) >> 2  # top 3 bits from G (bits 4-2)
                    b_bits = (b & 0xC0) >> 6  # top 2 bits from B (bits 1-0)
                    
                    # Combine into a single byte
                    byte = r_bits | g_bits | b_bits
                    extracted_bytes.append(byte)
            
            # Look for metadata pattern: filename|filetype|
            data = bytes(extracted_bytes)
            content = None
            file_info = None
            
            # Try to find the metadata separator pattern
            try:
                text_data = data.decode('utf-8', errors='ignore')
                pattern = r'([^|]+)\|([^|]+)\|'
                match = re.search(pattern, text_data)
                
                if match:
                    metadata_end = match.end()
                    filename = match.group(1)
                    filetype = match.group(2)
                    
                    # Extract content after metadata
                    content = data[metadata_end:]
                    
                    file_info = {
                        "filename": filename,
                        "filetype": filetype
                    }
                    
                    self.logger.info(f"Found encoded file: {filename} ({filetype})")
            except:
                # If metadata parsing fails, just use the raw data
                content = data
            
            # Check if content appears to be valid
            is_valid = self._validate_content(content)
            
            # Prepare results
            result_data = {
                "encoding_type": "custom_rgb_3_3_2",
                "file_info": file_info,
            }
            
            if content:
                # Provide a sample of the content
                if len(content) > 1000:
                    sample = content[:1000]
                    result_data["content_sample"] = {
                        "type": "binary",
                        "encoding": "base64",
                        "data": base64.b64encode(sample).decode('ascii'),
                        "note": "First 1000 bytes only"
                    }
                else:
                    result_data["content"] = {
                        "type": "binary",
                        "encoding": "base64",
                        "data": base64.b64encode(content).decode('ascii')
                    }
                
                # Try to interpret as text if it appears to be text
                try:
                    text_content = content.decode('utf-8', errors='ignore')
                    printable_ratio = sum(c.isprintable() for c in text_content) / len(text_content)
                    if printable_ratio > 0.8:  # If mostly printable chars
                        result_data["text_content"] = text_content[:2000]  # Limit text size
                except:
                    pass
                
                # If we found file metadata, save the extracted file
                if file_info:
                    output_dir = self.get_strategy_output_dir(image_path)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create a sanitized filename
                    safe_filename = re.sub(r'[^\w\-\.]', '_', file_info["filename"])
                    output_file = output_dir / safe_filename
                    
                    with open(output_file, 'wb') as f:
                        f.write(content)
                        
                    result_data["extracted_file_path"] = str(output_file)
                    self.logger.info(f"Saved extracted file to {output_file}")
            
            return (is_valid, result_data)
            
        except Exception as e:
            self.logger.error(f"Error in Custom RGB encoding analysis: {e}")
            return (False, {"error": str(e)})
    
    def _validate_content(self, content: bytes) -> bool:
        """
        Validate if the extracted content appears to be real data.
        
        Args:
            content: Extracted binary content
            
        Returns:
            Boolean indicating if content appears valid
        """
        if not content or len(content) < 8:
            return False
            
        # Check for known file signatures
        file_signatures = {
            b'\xFF\xD8\xFF': 'jpg',
            b'\x89PNG\r\n\x1a\n': 'png',
            b'GIF87a': 'gif',
            b'GIF89a': 'gif',
            b'%PDF': 'pdf',
            b'PK\x03\x04': 'zip',
            b'<!DOCTYPE html': 'html',
            b'<html': 'html'
        }
        
        for signature, _ in file_signatures.items():
            if content.startswith(signature):
                return True
                
        # If no file signature found, check if it's readable text
        try:
            text = content.decode('utf-8', errors='ignore')
            # Count printable characters
            printable_count = sum(1 for c in text if c.isprintable())
            if printable_count / len(text) > 0.8:
                return True
        except:
            pass
            
        # Other validation tests could be added here
        
        # If no validation passed, check byte distribution
        # A completely random distribution would suggest no real data
        byte_counts = {}
        for b in content[:1000]:  # Sample first 1000 bytes
            byte_counts[b] = byte_counts.get(b, 0) + 1
            
        # Calculate entropy
        entropy = 0
        for count in byte_counts.values():
            p = count / min(len(content), 1000)
            entropy -= p * np.log2(p) if p > 0 else 0
            
        # High entropy means more randomness
        # Valid data usually has some structure, so mid-range entropy
        return 3.0 < entropy < 7.5