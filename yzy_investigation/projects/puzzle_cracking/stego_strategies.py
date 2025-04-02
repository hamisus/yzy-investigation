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

from yzy_investigation.projects.puzzle_cracking.stego_analysis import StegStrategy


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