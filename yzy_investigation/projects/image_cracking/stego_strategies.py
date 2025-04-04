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
import os
import json

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
            # Check if this is a JPEG image - we need special handling
            is_jpeg = self._is_jpeg(image_path)
            
            # For JPEG images, we should be extra cautious as LSB steganography
            # is likely to be destroyed by JPEG compression
            if is_jpeg:
                self.logger.info(f"Image {image_path} is a JPEG. LSB detection may be unreliable.")
            
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
            
            # Enhanced randomness analysis
            # For natural images (especially JPEG), LSBs often appear random naturally
            randomness_score = self._calculate_randomness_score(lsb_data)
            
            # Adjust randomness threshold based on image type
            # JPEG images need a higher threshold as they naturally have more random-looking LSBs
            randomness_threshold = 0.65 if is_jpeg else 0.75
            is_random = randomness_score > randomness_threshold
            
            # Use stricter pattern detection for JPEGs
            if not is_random and is_jpeg:
                is_random = not self._has_significant_patterns(lsb_data)
            
            # Try to extract meaningful data
            extracted_data = None
            if not is_random:
                extracted_data = self._extract_data(img_array)
            
            result_data = {
                "is_random": is_random,
                "bits_checked": self.bits_to_check,
                "randomness_score": float(randomness_score),  # Convert numpy float to Python float
                "is_jpeg": is_jpeg,
                "confidence": "low" if is_jpeg else "medium"
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
            
            # For JPEG images, perform calibration to reduce false positives
            if is_jpeg and not is_random:
                calibration_result = self._perform_calibration(image_path, lsb_data)
                result_data["calibration"] = calibration_result
                
                # If calibration suggests the LSB patterns are likely JPEG artifacts,
                # adjust our finding
                if calibration_result.get("false_positive_likely", False):
                    self.logger.info("Calibration suggests the LSB patterns are likely JPEG artifacts")
                    return (False, result_data)
            
            # For JPEG images, always lower the confidence of LSB detection
            if is_jpeg:
                # Only report positive findings for JPEG if we're very confident
                if not is_random and extracted_data and "extracted_data" in result_data:
                    # Check if the extracted data has clear patterns or readable text
                    if self._extracted_data_has_clear_patterns(extracted_data):
                        return (True, result_data)
                    else:
                        # Not confident enough for JPEG
                        result_data["conclusion"] = "LSB patterns detected but insufficient confidence for JPEG"
                        return (False, result_data)
                else:
                    return (False, result_data)
            else:
                # For non-JPEG, proceed with normal detection
                return (not is_random, result_data)
            
        except Exception as e:
            self.logger.error(f"Error in LSB analysis: {e}")
            return (False, {"error": str(e)})
    
    def _extracted_data_has_clear_patterns(self, data: Union[str, bytes]) -> bool:
        """
        Check if the extracted data has clear patterns or readable text.
        
        This helps confirm that the data is likely real steganographic content
        rather than random artifacts, especially important for JPEG files.
        
        Args:
            data: The extracted data to check
            
        Returns:
            Boolean indicating if data has clear patterns
        """
        if isinstance(data, str):
            # Check for readable text
            if len(data) < 10:  # Too short to be meaningful
                return False
                
            # Count printable and alphanumeric characters
            printable_ratio = sum(c.isprintable() and not c.isspace() for c in data) / len(data)
            alpha_ratio = sum(c.isalnum() for c in data) / len(data)
            
            # Check for word-like patterns
            word_pattern = re.compile(r'\b[a-zA-Z]{2,}\b')
            words = word_pattern.findall(data)
            
            # If it has a good proportion of readable characters and some word-like structures
            return (printable_ratio > 0.7 and alpha_ratio > 0.4) or len(words) >= 3
        elif isinstance(data, bytes):
            # For binary data, check for known file signatures or structured data
            # Common file signatures
            signatures = [
                b'\xFF\xD8\xFF',  # JPEG
                b'\x89PNG\r\n\x1a\n',  # PNG
                b'GIF8',  # GIF
                b'PK\x03\x04',  # ZIP
                b'%PDF',  # PDF
                b'\x25\x50\x44\x46'  # PDF alternate
            ]
            
            if any(data.startswith(sig) for sig in signatures):
                return True
                
            # Check for structured binary data (not just random bytes)
            # Calculate entropy - structured data typically has lower entropy
            byte_counts = {}
            for b in data[:1000]:  # Check first 1000 bytes
                byte_counts[b] = byte_counts.get(b, 0) + 1
                
            entropy = 0
            for count in byte_counts.values():
                p = count / min(len(data), 1000)
                entropy -= p * np.log2(p) if p > 0 else 0
                
            # Lower entropy suggests structure (not random)
            # Max entropy for bytes is 8 bits
            return entropy < 7.2
        
        return False
    
    def _has_significant_patterns(self, lsb_data: Union[List[int], np.ndarray]) -> bool:
        """
        Check for significant patterns in LSB data that indicate steganography.
        
        This is a stricter test than the general randomness check and helps
        reduce false positives, especially for JPEG images.
        
        Args:
            lsb_data: Array of LSB values
            
        Returns:
            Boolean indicating if significant patterns were found
        """
        # Convert to numpy array if not already
        if not isinstance(lsb_data, np.ndarray):
            lsb_data = np.array(lsb_data)
            
        # Sample the data to keep computation reasonable
        max_samples = 100000
        if len(lsb_data) > max_samples:
            step = len(lsb_data) // max_samples
            lsb_data = lsb_data[::step]
            
        # 1. Check for runs of bits (too many consecutive identical values)
        runs = []
        current_run = 1
        for i in range(1, len(lsb_data)):
            if lsb_data[i] == lsb_data[i-1]:
                current_run += 1
            else:
                if current_run > 5:  # Only track significant runs
                    runs.append(current_run)
                current_run = 1
                
        # Add the last run if significant
        if current_run > 5:
            runs.append(current_run)
            
        # Calculate statistics on runs
        if runs:
            avg_run_length = sum(runs) / len(runs)
            max_run_length = max(runs)
            
            # Significant runs can indicate steganography
            # (natural images tend to have more random LSBs)
            if max_run_length > 20 or (avg_run_length > 8 and len(runs) > 10):
                return True
                
        # 2. Check for repeating patterns
        for pattern_length in [3, 4, 8, 16]:
            if len(lsb_data) < pattern_length * 3:
                continue
                
            # Count occurrences of each pattern
            pattern_counts = {}
            for i in range(len(lsb_data) - pattern_length):
                pattern = tuple(lsb_data[i:i+pattern_length])
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                
            # Check for unusually frequent patterns
            expected_freq = len(lsb_data) / (2 ** min(pattern_length, 8))  # Capped for large patterns
            
            for pattern, count in pattern_counts.items():
                if count > max(3, expected_freq * 3):  # At least 3x expected frequency
                    # Found a significantly overrepresented pattern
                    return True
                    
        # 3. Analyze bit transitions (0->1 and 1->0)
        # In natural images, transitions should be close to random
        transitions = 0
        for i in range(1, len(lsb_data)):
            if lsb_data[i] != lsb_data[i-1]:
                transitions += 1
                
        transition_ratio = transitions / (len(lsb_data) - 1)
        
        # For random data, transition ratio should be close to 0.5
        # Significant deviation may indicate steganography
        if abs(transition_ratio - 0.5) > 0.1:
            return True
            
        # No significant patterns detected
        return False
    
    def _is_jpeg(self, image_path: Path) -> bool:
        """
        Check if a file is a JPEG image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Boolean indicating if the file is a JPEG
        """
        return image_path.suffix.lower() in ['.jpg', '.jpeg', '.jpe', '.jfif']
    
    def _perform_calibration(self, image_path: Path, lsb_data: Union[List[int], np.ndarray]) -> Dict[str, Any]:
        """
        Perform calibration to reduce false positives in JPEG images.
        
        This technique compares the LSBs of the original image to a 
        slightly re-compressed version to distinguish JPEG artifacts
        from actual hidden data.
        
        Args:
            image_path: Path to the original image
            lsb_data: LSB data from the original image
            
        Returns:
            Dictionary with calibration results
        """
        try:
            # Create a temporary directory for calibration files
            output_dir = self.get_strategy_output_dir(image_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load the image
            img = self.load_image(image_path)
            
            # Save with slight re-compression (quality 92)
            calibration_path = output_dir / f"calibrated_{image_path.name}"
            img.save(calibration_path, "JPEG", quality=92)
            
            # Load the re-compressed image
            calib_img = self.load_image(calibration_path)
            calib_array = np.array(calib_img)
            
            # Extract LSBs from calibrated image
            if len(calib_array.shape) == 3:  # Color image
                calib_flat = calib_array.reshape(-1, calib_array.shape[2])
                calib_lsbs = []
                for channel in range(calib_flat.shape[1]):
                    channel_lsbs = calib_flat[:, channel] & ((1 << self.bits_to_check) - 1)
                    calib_lsbs.extend(channel_lsbs)
            else:  # Grayscale image
                calib_flat = calib_array.reshape(-1)
                calib_lsbs = calib_flat & ((1 << self.bits_to_check) - 1)
            
            # Convert to numpy arrays for easier comparison
            if not isinstance(lsb_data, np.ndarray):
                lsb_data = np.array(lsb_data)
            if not isinstance(calib_lsbs, np.ndarray):
                calib_lsbs = np.array(calib_lsbs)
            
            # Ensure both arrays are the same length
            min_length = min(len(lsb_data), len(calib_lsbs))
            lsb_data = lsb_data[:min_length]
            calib_lsbs = calib_lsbs[:min_length]
            
            # Calculate differences and similarities between original and calibrated LSBs
            differences = np.sum(lsb_data != calib_lsbs)
            diff_ratio = differences / min_length
            
            # Calculate randomness score of calibrated image
            calib_randomness = self._calculate_randomness_score(calib_lsbs)
            
            # Calculate bit change statistics
            bit_flips = {}
            for i in range(self.bits_to_check):
                orig_bits = (lsb_data >> i) & 1
                calib_bits = (calib_lsbs >> i) & 1
                flips = np.sum(orig_bits != calib_bits)
                bit_flips[f"bit_{i}"] = {
                    "flips": int(flips),
                    "flip_ratio": float(flips / min_length)
                }
            
            # Check for patterns in differences
            # If differences are evenly distributed, likely just compression artifacts
            # If differences show patterns, might be hidden data
            diff_positions = np.where(lsb_data != calib_lsbs)[0]
            
            # Check for clustering or patterns in differences
            clustering_score = self._calculate_difference_clustering(diff_positions, min_length)
            
            # Determine if the LSB differences are likely just JPEG artifacts
            # High clustering score suggests hidden data
            # Low clustering score suggests normal JPEG artifacts
            false_positive_likely = diff_ratio < 0.4 and clustering_score < 0.3
            
            # Clean up calibration file
            try:
                calibration_path.unlink()
            except:
                pass
            
            return {
                "diff_ratio": float(diff_ratio),
                "clustering_score": float(clustering_score),
                "calibrated_randomness": float(calib_randomness),
                "bit_flip_stats": bit_flips,
                "false_positive_likely": false_positive_likely
            }
            
        except Exception as e:
            self.logger.error(f"Error during LSB calibration: {e}")
            return {"error": str(e)}
    
    def _calculate_difference_clustering(self, diff_positions: np.ndarray, total_length: int) -> float:
        """
        Calculate a score indicating how clustered the differences are.
        
        In genuine steganography, differences between original and calibrated
        image often show patterns or clustering. Random differences from
        compression artifacts tend to be more evenly distributed.
        
        Args:
            diff_positions: Array of positions where differences occur
            total_length: Total length of the data
            
        Returns:
            Clustering score between 0 (evenly distributed) and 1 (highly clustered)
        """
        if len(diff_positions) < 2:
            return 0.0
            
        # Calculate distances between consecutive difference positions
        distances = diff_positions[1:] - diff_positions[:-1]
        
        # In random distribution, distances should follow an exponential distribution
        # Calculate mean distance
        mean_distance = np.mean(distances)
        
        # Calculate expected distances for a random distribution
        expected_mean = total_length / (len(diff_positions) + 1)
        
        # Calculate standard deviation
        std_dev = np.std(distances)
        
        # In random distribution, std dev should be close to mean
        # A much smaller std dev indicates clustering
        std_ratio = std_dev / mean_distance if mean_distance > 0 else 1.0
        
        # Calculate ratio of mean distance to expected mean
        # Values much smaller than 1 indicate clustering
        distance_ratio = mean_distance / expected_mean if expected_mean > 0 else 1.0
        
        # Calculate clustering score
        # Higher score = more clustering = more likely to be steganography
        clustering_score = 1.0 - min(1.0, (std_ratio * distance_ratio))
        
        return clustering_score
    
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
            is_jpeg = bool(self._is_jpeg(image_path))  # Ensure Python bool
            
            if is_jpeg:
                self.logger.info(f"Image {image_path} is a JPEG. Using adjusted histogram analysis.")
            
            # Convert to numpy array for analysis
            img_array = np.array(img)
            
            # Analyze each color channel separately
            channels = ['R', 'G', 'B'] if len(img_array.shape) == 3 else ['Gray']
            channel_results = {}
            anomalies_found = False
            
            for i, channel_name in enumerate(channels):
                # Get histogram for this channel
                if len(channels) == 1:
                    histogram = np.histogram(img_array, bins=256, range=(0, 256))[0]
                else:
                    histogram = np.histogram(img_array[:,:,i], bins=256, range=(0, 256))[0]
                
                # Find anomalies in the histogram
                anomalies = self._find_histogram_anomalies(histogram, is_jpeg)
                if anomalies:
                    anomalies_found = True
                
                # Calculate comb pattern score
                comb_score = self._calculate_comb_pattern_score(histogram, is_jpeg)
                
                channel_results[channel_name] = {
                    "anomalies": anomalies,
                    "comb_pattern_score": float(comb_score),
                    "histogram": histogram.tolist()
                }
            
            # Calculate overall confidence based on findings
            confidence = float(self._calculate_confidence(channel_results))  # Ensure float
            detected = bool(confidence > 0.6)  # Ensure Python bool
            
            return detected, {
                "channel_analysis": channel_results,
                "confidence": confidence,
                "is_jpeg": is_jpeg
            }
            
        except Exception as e:
            self.logger.error(f"Error in color histogram analysis: {e}")
            return False, {"error": str(e)}
    
    def _calculate_comb_pattern_score(self, histogram: np.ndarray, is_jpeg: bool = False) -> float:
        """
        Calculate a score indicating the presence of comb patterns in the histogram.
        
        Args:
            histogram: Array containing the histogram values
            is_jpeg: Whether the image is a JPEG (affects thresholds)
            
        Returns:
            Score between 0 and 1 indicating likelihood of comb patterns
        """
        if is_jpeg:
            return self._calculate_jpeg_aware_comb_score(histogram)
            
        # For non-JPEG images, look for regular patterns
        scores = []
        
        # Check different potential comb widths
        for width in range(2, 9):
            # Calculate variance for each offset
            variances = []
            for offset in range(width):
                values = histogram[offset::width]
                if len(values) > 1:
                    variances.append(np.var(values))
            
            if variances:
                # Low variance indicates regular pattern
                avg_var = np.mean(variances)
                max_val = np.max(histogram)
                if max_val > 0:
                    # Normalize variance by maximum value
                    norm_var = avg_var / (max_val * max_val)
                    # Convert to score (lower variance = higher score)
                    score = max(0, 1 - norm_var)
                    scores.append(score)
        
        return float(max(scores)) if scores else 0.0
    
    def _calculate_jpeg_aware_comb_score(self, histogram: np.ndarray) -> float:
        """
        Calculate comb pattern score accounting for JPEG quantization.
        
        Args:
            histogram: Array containing the histogram values
            
        Returns:
            Score between 0 and 1 indicating likelihood of suspicious patterns
        """
        # JPEG typically quantizes in steps of 8
        quant_step = 8
        scores = []
        
        # Check for patterns at quantization boundaries
        for offset in range(quant_step):
            values = histogram[offset::quant_step]
            if len(values) > 1:
                # Calculate statistics
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                if mean_val > 0:
                    # Calculate coefficient of variation
                    cv = std_val / mean_val
                    # Convert to score (lower CV = higher score)
                    score = max(0, 1 - cv)
                    scores.append(score)
        
        # Also check for unusual peaks between quantization steps
        between_steps = []
        for i in range(0, 256 - quant_step, quant_step):
            section = histogram[i:i+quant_step]
            if len(section) > 2:
                # Compare middle values to edges
                mid_vals = section[1:-1]
                edge_vals = np.array([section[0], section[-1]])
                if np.mean(edge_vals) > 0:
                    ratio = np.max(mid_vals) / np.mean(edge_vals)
                    if ratio > 2.0:  # Suspicious peak
                        between_steps.append(min(1.0, ratio / 10.0))
        
        if between_steps:
            scores.append(max(between_steps))
            
        return float(max(scores)) if scores else 0.0
    
    def _calculate_confidence(self, channel_results: Dict[str, Dict[str, Any]]) -> float:
        """
        Calculate overall confidence score from channel analysis results.
        
        Args:
            channel_results: Dictionary containing analysis results for each channel
            
        Returns:
            Confidence score between 0 and 1
        """
        scores = []
        
        for channel_data in channel_results.values():
            # Consider comb pattern score
            scores.append(channel_data["comb_pattern_score"])
            
            # Consider anomalies
            if channel_data["anomalies"]:
                anomaly_scores = [
                    anomaly.get("confidence", 0.0) 
                    for anomaly in channel_data["anomalies"]
                ]
                if anomaly_scores:
                    scores.append(max(anomaly_scores))
        
        # Weight multiple high scores more heavily
        if len(scores) > 1:
            scores.sort(reverse=True)
            # Give more weight to multiple high scores
            confidence = scores[0] * 0.6 + sum(scores[1:]) / (len(scores) - 1) * 0.4
        else:
            confidence = scores[0] if scores else 0.0
            
        return float(confidence)
    
    def _is_jpeg(self, image_path: Path) -> bool:
        """
        Check if the image is a JPEG file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Boolean indicating if the file is a JPEG
        """
        return image_path.suffix.lower() in ['.jpg', '.jpeg', '.jpe', '.jfif']
    
    def _find_histogram_anomalies(self, histogram: np.ndarray, is_jpeg: bool = False) -> List[Dict[str, Any]]:
        """
        Find anomalies in a color histogram that might indicate steganography.
        
        Args:
            histogram: Array containing the histogram values
            is_jpeg: Whether the image is a JPEG (affects thresholds)
            
        Returns:
            List of dictionaries describing found anomalies
        """
        anomalies = []
        
        # For JPEGs, we need to account for compression artifacts
        if is_jpeg:
            # Check for unusual peaks that aren't at quantization steps
            quant_step = 8  # Common JPEG quantization step
            for i in range(1, 255):
                if i % quant_step != 0:  # Not at a quantization step
                    if histogram[i] > 0:
                        # Compare to neighboring quantization steps
                        prev_step = (i // quant_step) * quant_step
                        next_step = prev_step + quant_step
                        if prev_step >= 0 and next_step < 256:
                            if histogram[i] > max(histogram[prev_step], histogram[next_step]) * 0.5:
                                anomalies.append({
                                    "type": "unusual_peak",
                                    "position": i,
                                    "value": float(histogram[i]),
                                    "expected_max": float(max(histogram[prev_step], histogram[next_step])),
                                    "confidence": min(0.8, histogram[i] / max(histogram[prev_step], histogram[next_step]))
                                })
        else:
            # For non-JPEG images, look for sharp discontinuities
            for i in range(1, 255):
                left_avg = np.mean(histogram[max(0, i-3):i])
                right_avg = np.mean(histogram[i+1:min(256, i+4)])
                if left_avg > 0 and right_avg > 0:
                    ratio = histogram[i] / ((left_avg + right_avg) / 2)
                    if ratio > 3.0:  # Significant peak
                        anomalies.append({
                            "type": "sharp_peak",
                            "position": i,
                            "value": float(histogram[i]),
                            "surrounding_avg": float((left_avg + right_avg) / 2),
                            "ratio": float(ratio),
                            "confidence": min(0.9, ratio / 10.0)
                        })
        
        # Look for periodic patterns
        if len(histogram) >= 16:
            for period in range(2, 9):
                periodic_scores = []
                for offset in range(period):
                    values = histogram[offset::period]
                    if len(values) > 1:
                        # Calculate variation coefficient
                        mean = np.mean(values)
                        std = np.std(values)
                        if mean > 0:
                            cv = std / mean
                            periodic_scores.append(cv)
                
                if periodic_scores:
                    avg_cv = np.mean(periodic_scores)
                    if avg_cv < 0.3:  # Strong periodicity
                        anomalies.append({
                            "type": "periodic_pattern",
                            "period": period,
                            "variation_coefficient": float(avg_cv),
                            "confidence": min(0.85, (0.3 - avg_cv) / 0.3)
                        })
        
        return anomalies


class MetadataAnalysisStrategy(StegStrategy):
    """
    Strategy to analyze image metadata for hidden data.
    
    This strategy looks for unusual or suspicious metadata that could
    contain hidden information.
    """
    
    name: str = "metadata_analysis_strategy"
    description: str = "Analysis of image metadata for hidden information"
    
    # Common metadata fields that often contain large amounts of legitimate data
    COMMON_LARGE_FIELDS = [
        'xml:XMP', 'XMP', 'XML:com.adobe.xmp', 
        'icc_profile', 'exif', 'photoshop',
        'makernotes', 'makerNote', 'MakerNote',
        'UserComment', 'ImageDescription', 'Software'
    ]
    
    # Metadata fields that are commonly used for steganography
    SUSPICIOUS_FIELD_NAMES = [
        'comment', 'remarks', 'notes', 'keywords', 'custom', 
        'private', 'hidden', 'secret', 'data', 'embed'
    ]
    
    # Camera brands that often have large MakerNotes
    KNOWN_CAMERA_BRANDS = [
        'NIKON', 'CANON', 'SONY', 'PENTAX', 'OLYMPUS', 
        'FUJIFILM', 'PANASONIC', 'LEICA', 'SIGMA'
    ]
    
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
            
            # Check if file is a JPEG to handle EXIF properly
            is_jpeg = self._is_jpeg(image_path)
            
            if hasattr(img, 'info'):
                metadata = dict(img.info)
                
                # Get image format for context
                format_name = img.format if hasattr(img, 'format') else "unknown"
                img_size = img.size if hasattr(img, 'size') else (0, 0)
                camera_brand = self._extract_camera_brand(metadata)
                
                # Analyze metadata for suspicious content with context
                for key, value in metadata.items():
                    self._analyze_metadata_field(key, value, suspicious_fields, format_name, camera_brand, img_size)
            
            # Save any exif data specifically
            exif_data = self._extract_exif(img)
            if exif_data:
                metadata["EXIF"] = exif_data
                
                # Analyze EXIF data separately for suspicious fields
                # This might catch things that were missed in the general metadata
                for tag, value in exif_data.items():
                    self._analyze_exif_field(tag, value, suspicious_fields, camera_brand)
            
            # Extract and analyze any XMP data (Adobe metadata)
            xmp_data = self._extract_xmp(metadata)
            if xmp_data:
                metadata["XMP"] = xmp_data
                
                # Look for suspicious content in XMP
                suspicious_xmp = self._analyze_xmp_data(xmp_data)
                if suspicious_xmp:
                    suspicious_fields.extend(suspicious_xmp)
            
            # Calculate an overall suspicion score
            suspicion_score = self._calculate_suspicion_score(suspicious_fields)
            
            # A higher threshold for considering metadata suspicious
            # This helps reduce false positives from legitimate large metadata
            is_suspicious = suspicion_score > 0.7
            
            # If we have suspicious fields but score is below threshold,
            # treat them as "potential" findings rather than definite ones
            if suspicious_fields and not is_suspicious:
                for field in suspicious_fields:
                    field["confidence"] = "low"
            
            # Prepare results
            result = {
                "metadata": self._clean_metadata_for_output(metadata),
                "suspicious_fields": suspicious_fields,
                "suspicion_score": suspicion_score,
                "is_suspicious": is_suspicious,
                "extracted_data": suspicious_fields[0].get("text") if suspicious_fields and is_suspicious else None
            }
            
            # Only report as suspicious if the score is high enough
            return (is_suspicious, result)
            
        except Exception as e:
            self.logger.error(f"Error in metadata analysis: {e}")
            return (False, {"error": str(e)})
    
    def _is_jpeg(self, image_path: Path) -> bool:
        """
        Check if the file is a JPEG.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Boolean indicating if the file is a JPEG
        """
        return image_path.suffix.lower() in ['.jpg', '.jpeg', '.jpe', '.jfif']
    
    def _extract_camera_brand(self, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Extract camera brand from metadata if available.
        
        Args:
            metadata: Image metadata dictionary
            
        Returns:
            Camera brand string or None
        """
        # Check various fields that might contain camera info
        for field in ['Make', 'CameraMake', 'EquipMake', 'make']:
            if field in metadata and isinstance(metadata[field], str):
                return metadata[field].upper()
        
        # Check EXIF if it's directly available
        if 'exif' in metadata and isinstance(metadata['exif'], dict):
            exif = metadata['exif']
            for field in ['Make', 'CameraMake', 'EquipMake']:
                if field in exif and isinstance(exif[field], str):
                    return exif[field].upper()
        
        return None
    
    def _analyze_metadata_field(self, key: str, value: Any, suspicious_fields: List[Dict[str, Any]], 
                                format_name: str, camera_brand: Optional[str], img_size: Tuple[int, int]) -> None:
        """
        Analyze a metadata field for suspicious content.
        
        Args:
            key: Metadata field name
            value: Metadata field value
            suspicious_fields: List to append suspicious findings to
            format_name: Image format name
            camera_brand: Camera brand if known
            img_size: Image dimensions
        """
        # Skip some fields that are almost always large and legitimate
        if any(common_key in key for common_key in self.COMMON_LARGE_FIELDS):
            # Still check these fields, but with higher thresholds
            size_threshold = 2000  # Much higher threshold for common large fields
            if isinstance(value, (bytes, str)) and len(str(value)) > size_threshold:
                suspicious_fields.append({
                    "field": key,
                    "reason": "unusually_large_even_for_common_field",
                    "size": len(str(value)),
                    "confidence": "medium"
                })
            return
        
        # Check for unusually large metadata fields
        if isinstance(value, (bytes, str)):
            # Adjust size threshold based on image size and format
            size_threshold = 100  # Default threshold
            
            # JPEG images with EXIF can legitimately have larger metadata
            if format_name.lower() == 'jpeg' and camera_brand:
                # Known camera brands often have large maker notes
                if camera_brand in self.KNOWN_CAMERA_BRANDS:
                    size_threshold = 500
                else:
                    size_threshold = 200
            
            # For larger images, metadata might be larger too
            width, height = img_size
            if width * height > 3000000:  # > 3 megapixels
                size_threshold *= 2
            
            if len(str(value)) > size_threshold:
                # Add suspiciousness if field name matches suspicious patterns
                confidence = "medium"
                if any(suspicious in key.lower() for suspicious in self.SUSPICIOUS_FIELD_NAMES):
                    confidence = "high"
                
                suspicious_fields.append({
                    "field": key,
                    "reason": "unusually_large",
                    "size": len(str(value)),
                    "confidence": confidence
                })
        
        # Check for base64-encoded data
        if isinstance(value, str) and self._looks_like_base64(value) and len(value) > 20:
            suspicious_fields.append({
                "field": key,
                "reason": "possible_base64",
                "confidence": "high" if len(value) > 100 else "medium"
            })
            
        # Check for hidden text in metadata
        if isinstance(value, str) and len(value) > 10:
            hidden_text = self._extract_hidden_text(value)
            if hidden_text:
                suspicious_fields.append({
                    "field": key,
                    "reason": "contains_text",
                    "text": hidden_text,
                    "confidence": "high"
                })
    
    def _analyze_exif_field(self, tag: str, value: Any, suspicious_fields: List[Dict[str, Any]], 
                           camera_brand: Optional[str]) -> None:
        """
        Analyze an EXIF field for suspicious content.
        
        Args:
            tag: EXIF tag name
            value: EXIF tag value
            suspicious_fields: List to append suspicious findings to
            camera_brand: Camera brand if known
        """
        # Skip tags that are known to be large for certain camera brands
        if camera_brand in self.KNOWN_CAMERA_BRANDS and tag in ['MakerNote', 'MakerNotes']:
            # Even for known cameras, maker notes shouldn't be extremely large
            if isinstance(value, (str, bytes)) and len(str(value)) > 5000:
                suspicious_fields.append({
                    "field": f"EXIF.{tag}",
                    "reason": "extremely_large_makernote",
                    "size": len(str(value)),
                    "confidence": "medium"
                })
            return
        
        # Check for unusual or non-standard EXIF tags
        if not self._is_standard_exif_tag(tag):
            suspicious_fields.append({
                "field": f"EXIF.{tag}",
                "reason": "non_standard_exif_tag",
                "confidence": "medium"
            })
        
        # Check for unusually large data in standard fields
        if isinstance(value, (str, bytes)) and len(str(value)) > 200:
            suspicious_fields.append({
                "field": f"EXIF.{tag}",
                "reason": "unusually_large_exif_field",
                "size": len(str(value)),
                "confidence": "medium"
            })
        
        # Check for base64-encoded data
        if isinstance(value, str) and self._looks_like_base64(value) and len(value) > 20:
            suspicious_fields.append({
                "field": f"EXIF.{tag}",
                "reason": "possible_base64_in_exif",
                "confidence": "high"
            })
            
        # Check for hidden text
        if isinstance(value, str) and len(value) > 10:
            hidden_text = self._extract_hidden_text(value)
            if hidden_text:
                suspicious_fields.append({
                    "field": f"EXIF.{tag}",
                    "reason": "contains_text",
                    "text": hidden_text,
                    "confidence": "high"
                })
    
    def _is_standard_exif_tag(self, tag: str) -> bool:
        """
        Check if an EXIF tag is standard or non-standard.
        
        Args:
            tag: EXIF tag to check
            
        Returns:
            Boolean indicating if tag is standard
        """
        # List of common standard EXIF tags
        standard_tags = [
            # Camera and image information
            'Make', 'Model', 'Software', 'Artist', 'Copyright', 'DateTime',
            'DateTimeOriginal', 'DateTimeDigitized', 'SubsecTime', 'ExposureTime',
            'FNumber', 'ExposureProgram', 'ISOSpeedRatings', 'ShutterSpeedValue',
            'ApertureValue', 'BrightnessValue', 'ExposureBiasValue', 'MaxApertureValue',
            'SubjectDistance', 'MeteringMode', 'LightSource', 'Flash', 'FocalLength',
            'SubjectArea', 'FlashEnergy', 'SpatialFrequencyResponse', 'FocalPlaneXResolution',
            'FocalPlaneYResolution', 'FocalPlaneResolutionUnit', 'SubjectLocation', 'ExposureIndex',
            'SensingMethod', 'FileSource', 'SceneType', 'CFAPattern', 'CustomRendered',
            'ExposureMode', 'WhiteBalance', 'DigitalZoomRatio', 'FocalLengthIn35mmFilm',
            'SceneCaptureType', 'GainControl', 'Contrast', 'Saturation', 'Sharpness', 
            'DeviceSettingDescription', 'SubjectDistanceRange',
            
            # Image data structure
            'ImageWidth', 'ImageLength', 'BitsPerSample', 'Compression', 'PhotometricInterpretation',
            'Orientation', 'SamplesPerPixel', 'PlanarConfiguration', 'YCbCrSubSampling',
            'YCbCrPositioning', 'XResolution', 'YResolution', 'ResolutionUnit',
            
            # User comments
            'UserComment', 'ImageDescription',
            
            # GPS tags
            'GPSVersionID', 'GPSLatitudeRef', 'GPSLatitude', 'GPSLongitudeRef', 'GPSLongitude',
            'GPSAltitudeRef', 'GPSAltitude', 'GPSTimeStamp', 'GPSSatellites', 'GPSStatus',
            'GPSMeasureMode', 'GPSDOP', 'GPSSpeedRef', 'GPSSpeed', 'GPSTrackRef', 'GPSTrack',
            'GPSImgDirectionRef', 'GPSImgDirection', 'GPSMapDatum', 'GPSDestLatitudeRef',
            'GPSDestLatitude', 'GPSDestLongitudeRef', 'GPSDestLongitude', 'GPSDestBearingRef',
            'GPSDestBearing', 'GPSDestDistanceRef', 'GPSDestDistance', 'GPSProcessingMethod',
            'GPSAreaInformation', 'GPSDateStamp', 'GPSDifferential',
            
            # Maker notes
            'MakerNote', 'MakerNotes'
        ]
        
        # Check if tag is standard (case-insensitive)
        return any(standard.lower() == tag.lower() for standard in standard_tags)
    
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
    
    def _extract_xmp(self, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract and parse XMP data from image metadata.
        
        Args:
            metadata: Image metadata dictionary
            
        Returns:
            Parsed XMP data dictionary or None
        """
        xmp_data = None
        
        # Look for XMP data in several possible metadata fields
        xmp_fields = ['xml:XMP', 'XMP', 'XML:com.adobe.xmp', 'xmp']
        
        for field in xmp_fields:
            if field in metadata and isinstance(metadata[field], (str, bytes)):
                try:
                    xmp_content = metadata[field]
                    if isinstance(xmp_content, bytes):
                        xmp_content = xmp_content.decode('utf-8', errors='replace')
                    
                    # Very basic XMP parsing - extract key/value pairs
                    # For a real implementation, you would use a proper XML parser
                    xmp_data = {}
                    
                    # Look for tags in XMP
                    tag_pattern = r'<([^:>]+):([^>]+)>(.*?)</\1:\2>'
                    for match in re.finditer(tag_pattern, xmp_content):
                        namespace, tag, value = match.groups()
                        key = f"{namespace}:{tag}"
                        xmp_data[key] = value
                    
                    # Also look for attributes
                    attr_pattern = r'<[^>]+ ([^:]+):([^=]+)="([^"]*)"'
                    for match in re.finditer(attr_pattern, xmp_content):
                        namespace, attr, value = match.groups()
                        key = f"{namespace}:{attr}"
                        xmp_data[key] = value
                    
                    break
                except Exception as e:
                    self.logger.warning(f"Error parsing XMP data: {e}")
        
        return xmp_data
    
    def _analyze_xmp_data(self, xmp_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze XMP data for suspicious content.
        
        Args:
            xmp_data: XMP data dictionary
            
        Returns:
            List of suspicious findings
        """
        suspicious_findings = []
        
        for key, value in xmp_data.items():
            # Check for unusually large XMP fields
            if isinstance(value, str) and len(value) > 300:
                suspicious_findings.append({
                    "field": f"XMP.{key}",
                    "reason": "unusually_large_xmp_field",
                    "size": len(value),
                    "confidence": "medium"
                })
                
            # Check for base64-encoded data
            if isinstance(value, str) and self._looks_like_base64(value) and len(value) > 20:
                suspicious_findings.append({
                    "field": f"XMP.{key}",
                    "reason": "possible_base64_in_xmp",
                    "confidence": "high"
                })
                
            # Check for hidden text
            if isinstance(value, str) and len(value) > 10:
                hidden_text = self._extract_hidden_text(value)
                if hidden_text:
                    suspicious_findings.append({
                        "field": f"XMP.{key}",
                        "reason": "contains_text",
                        "text": hidden_text,
                        "confidence": "high"
                    })
        
        return suspicious_findings
    
    def _looks_like_base64(self, s: str) -> bool:
        """
        Check if a string looks like Base64-encoded data.
        
        Args:
            s: String to check
            
        Returns:
            Boolean indicating if string looks like Base64
        """
        # Base64 typically consists of alphanumeric characters, +, /, and possibly = at the end
        pattern = r'^[A-Za-z0-9+/]+={0,2}$'
        
        # String should be reasonably long and match the pattern
        if len(s) > 20 and bool(re.match(pattern, s)):
            # Check character frequency distribution - base64 has a fairly even distribution
            char_counts = {}
            for c in s:
                if c != '=':  # Skip padding
                    char_counts[c] = char_counts.get(c, 0) + 1
            
            # Calculate entropy of distribution
            entropy = 0
            for count in char_counts.values():
                p = count / (len(s) - s.count('='))
                entropy -= p * np.log2(p)
            
            # Base64 data typically has high entropy
            # Maximum entropy for 64 symbols is 6 bits
            return entropy > 4.0
        
        return False
    
    def _extract_hidden_text(self, text: str) -> Optional[str]:
        """
        Extract hidden readable text from a string.
        
        Args:
            text: String to analyze
            
        Returns:
            Extracted text if found, None otherwise
        """
        # Look for sequences of readable ASCII characters
        readable_parts = re.findall(r'[A-Za-z0-9\s.,!?:;(){}\[\]\'\"]{5,}', text)
        
        if readable_parts:
            # Filter out very common words/patterns that might appear in normal metadata
            filtered_parts = []
            common_phrases = ['image', 'photo', 'picture', 'camera', 'jpeg', 'canon', 'nikon', 
                              'sony', 'adobe', 'photoshop', 'lightroom', 'copyright']
            
            for part in readable_parts:
                is_common = False
                for phrase in common_phrases:
                    if phrase.lower() in part.lower():
                        is_common = True
                        break
                
                if not is_common and len(part) > 5:
                    filtered_parts.append(part)
            
            if filtered_parts:
                # Join and return parts
                return ' '.join(filtered_parts)
                
        # Try to decode as base64 and check for readable text
        if self._looks_like_base64(text):
            try:
                decoded = base64.b64decode(text).decode('utf-8', errors='ignore')
                readable_parts = re.findall(r'[A-Za-z0-9\s.,!?:;(){}\[\]\'\"]{5,}', decoded)
                if readable_parts:
                    return 'Base64 decoded: ' + ' '.join(readable_parts)
            except:
                pass
                
        return None
    
    def _calculate_suspicion_score(self, suspicious_fields: List[Dict[str, Any]]) -> float:
        """
        Calculate an overall suspicion score based on the findings.
        
        Args:
            suspicious_fields: List of suspicious field findings
            
        Returns:
            Suspicion score between 0.0 and 1.0
        """
        if not suspicious_fields:
            return 0.0
            
        # Calculate a weighted score based on confidence and reason
        total_score = 0.0
        
        for field in suspicious_fields:
            # Base score by confidence
            if field.get("confidence") == "high":
                score = 0.8
            elif field.get("confidence") == "medium":
                score = 0.5
            else:
                score = 0.2
            
            # Adjust by reason
            reason = field.get("reason", "")
            if "base64" in reason:
                score += 0.1
            elif "contains_text" in reason:
                score += 0.2
            elif "non_standard" in reason:
                score += 0.1
            elif "unusually_large" in reason and not "common_field" in reason:
                score += 0.1
            
            # Cap individual scores at 1.0
            total_score += min(1.0, score)
        
        # Normalize score based on number of findings
        # More findings = higher overall score
        return min(1.0, total_score / (len(suspicious_fields) + 2))
    
    def _clean_metadata_for_output(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean metadata for output, ensuring all values are JSON serializable.
        
        Args:
            metadata: Raw metadata dictionary
            
        Returns:
            Cleaned metadata dictionary
        """
        cleaned = {}
        
        for key, value in metadata.items():
            if isinstance(value, dict):
                cleaned[key] = self._clean_metadata_for_output(value)
            elif isinstance(value, (str, int, float, bool, list, tuple)):
                # Keep common scalar types and lists
                if isinstance(value, (list, tuple)):
                    # Ensure list elements are serializable
                    cleaned[key] = [str(item) if not isinstance(item, (str, int, float, bool)) else item 
                                   for item in value]
                else:
                    cleaned[key] = value
            elif isinstance(value, bytes):
                # Convert binary data to base64
                if len(value) > 100:
                    # For large binary values, just note the size
                    cleaned[key] = f"<binary data, {len(value)} bytes>"
                else:
                    # For smaller binary values, include the data
                    cleaned[key] = base64.b64encode(value).decode('ascii')
            elif isinstance(value, IFDRational):
                # Convert IFDRational to float
                cleaned[key] = float(value)
            else:
                # Convert other types to string
                cleaned[key] = str(value)
        
        return cleaned


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
            # Check if this strategy is applicable to the image
            strategy_validation = FindingValidator.validate_steganography_approach(self.name, image_path)
            if not strategy_validation["applicable"]:
                self.logger.info(f"KeywordXorStrategy may not be suitable for {image_path}: {strategy_validation['reason']}")
            
            img = self.load_image(image_path)
            img_array = np.array(img)
            
            # For JPEGs, check if we're likely to get meaningful results
            is_jpeg = FindingValidator.is_jpeg_decompressed(image_path)
            if is_jpeg:
                self.logger.info(f"Image {image_path} is a JPEG. XOR operations on decompressed data may produce artifacts.")
                
                # Reduce the set of keys to try for JPEGs to minimize false positives
                key_terms = [self.KEY_TERMS[0], self.KEY_TERMS[-1]]  # Just try the first and last terms
                key_numbers = [self.KEY_NUMBERS[0]]  # Just try the first number
            else:
                key_terms = self.KEY_TERMS
                key_numbers = self.KEY_NUMBERS
            
            # Flatten image data for processing
            if len(img_array.shape) == 3:  # Color image
                flat_data = img_array.reshape(-1)
            else:  # Grayscale image
                flat_data = img_array.reshape(-1)
                
            results = {}
            found_genuine = False
            found_something = False
            highest_confidence = 0.0
            
            # Try each key term
            for term in key_terms:
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
                    
                    # Validate the extracted text
                    text_validation = FindingValidator.validate_text_content(xor_text)
                    confidence = text_validation["confidence"]
                    
                    if text_validation["valid"]:
                        results[f"xor_{term}"] = {
                            "readable_text": True,
                            "data": xor_text[:1000],  # Limit output size
                            "validation": text_validation,
                            "confidence": confidence
                        }
                        found_something = True
                        highest_confidence = max(highest_confidence, confidence)
                        
                        # Consider it genuine if confidence is high or has investigation keywords
                        if confidence > 0.6 or text_validation["investigation_keyword_count"] > 0:
                            found_genuine = True
                    
                    # Check for target string
                    target_string = "4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump"
                    if target_string in xor_text:
                        # Validate this is not a coincidence
                        is_false_positive = self._check_target_false_positive(flat_data, term_bytes, xor_text, target_string)
                        
                        if not is_false_positive:
                            target_pos = xor_text.find(target_string)
                            context_start = max(0, target_pos - 100)
                            context_end = min(len(xor_text), target_pos + len(target_string) + 200)
                            context = xor_text[context_start:context_end]
                            
                            results[f"xor_{term}_found_key"] = {
                                "found_target": True,
                                "context": context,
                                "target_position": target_pos,
                                "full_data": xor_text,  # Store the full decoded text
                                "confidence": 0.9  # High confidence for target string
                            }
                            found_something = True
                            found_genuine = True
                            highest_confidence = 0.9
                            self.logger.warning(f"FOUND TARGET STRING with XOR key {term}!")
                            
                            # Save the full decoded content to a file
                            self._save_decoded_content(image_path, term, xor_text, target_pos)
                        else:
                            results[f"xor_{term}_false_positive"] = {
                                "false_positive": True,
                                "reason": "Target string found due to artifacts in the data",
                                "position": xor_text.find(target_string)
                            }
                    
                    # Also look for '4NBT' or 'silver' phrases which might be significant
                    for clue in ["4NBT", "silver"]:
                        if clue in xor_text:
                            # Validate this is not a coincidence
                            is_clue_false_positive = self._check_clue_false_positive(xor_text, clue)
                            
                            if not is_clue_false_positive:
                                results[f"xor_{term}_found_clue"] = {
                                    "found_clue": True,
                                    "clue": clue,
                                    "context": xor_text[:1000],
                                    "confidence": 0.7  # Good confidence for finding a clue
                                }
                                found_something = True
                                found_genuine = True
                                highest_confidence = max(highest_confidence, 0.7)
                except Exception as e:
                    self.logger.debug(f"Error decoding XOR result for {term}: {e}")
            
            # Also try XOR with numeric keys
            for num in key_numbers:
                xor_result = bytearray()
                for i in range(len(flat_data)):
                    xor_result.append(flat_data[i] ^ (num % 256))
                
                try:
                    xor_text = xor_result.decode('ascii', errors='ignore')
                    
                    # Validate the extracted text
                    text_validation = FindingValidator.validate_text_content(xor_text)
                    confidence = text_validation["confidence"]
                    
                    if text_validation["valid"]:
                        results[f"xor_num_{num}"] = {
                            "readable_text": True,
                            "data": xor_text[:1000],  # Limit output size
                            "validation": text_validation,
                            "confidence": confidence
                        }
                        found_something = True
                        highest_confidence = max(highest_confidence, confidence)
                        
                        # Consider it genuine if confidence is high or has investigation keywords
                        if confidence > 0.6 or text_validation["investigation_keyword_count"] > 0:
                            found_genuine = True
                    
                    # Check for target string
                    target_string = "4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump"
                    if target_string in xor_text:
                        # Validate this is not a coincidence
                        is_false_positive = self._check_numeric_target_false_positive(flat_data, num, xor_text, target_string)
                        
                        if not is_false_positive:
                            target_pos = xor_text.find(target_string)
                            context_start = max(0, target_pos - 100)
                            context_end = min(len(xor_text), target_pos + len(target_string) + 200)
                            context = xor_text[context_start:context_end]
                            
                            results[f"xor_num_{num}_found_key"] = {
                                "found_target": True,
                                "context": context,
                                "target_position": target_pos,
                                "full_data": xor_text,  # Store the full decoded text
                                "confidence": 0.9  # High confidence for target string
                            }
                            found_something = True
                            found_genuine = True
                            highest_confidence = 0.9
                            self.logger.warning(f"FOUND TARGET STRING with XOR number {num}!")
                            
                            # Save the full decoded content to a file
                            self._save_decoded_content(image_path, f"num_{num}", xor_text, target_pos)
                        else:
                            results[f"xor_num_{num}_false_positive"] = {
                                "false_positive": True,
                                "reason": f"Target string found due to artifacts in the data",
                                "position": xor_text.find(target_string)
                            }
                except Exception as e:
                    self.logger.debug(f"Error decoding XOR result for number {num}: {e}")
            
            # Add context about the confidence level
            if found_something:
                results["overall_confidence"] = highest_confidence
                
                # For JPEGs, we need a higher confidence threshold
                if is_jpeg and highest_confidence < 0.7:
                    self.logger.info(f"Found potential matches in JPEG but confidence ({highest_confidence:.2f}) below threshold (0.7)")
                    return (False, {
                        "potential_findings": results,
                        "confidence": highest_confidence,
                        "message": "Potential findings in JPEG but confidence too low"
                    })
            
            # Only return true if we found genuine findings, not just false positives
            return (found_genuine, results if found_something else None)
            
        except Exception as e:
            self.logger.error(f"Error in keyword XOR analysis: {e}")
            return (False, {"error": str(e)})
    
    def _check_target_false_positive(self, flat_data: np.ndarray, term_bytes: bytes, 
                                    xor_text: str, target_string: str) -> bool:
        """
        Check if a target string match is likely a false positive.
        
        Args:
            flat_data: Original image data
            term_bytes: XOR key bytes
            xor_text: XORed text result
            target_string: Target string that was found
            
        Returns:
            Boolean indicating if this is likely a false positive
        """
        target_pos = xor_text.find(target_string)
        
        # Case 1: Using the target string itself as the key
        if target_string in term_bytes.decode('utf-8', errors='ignore'):
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
                self.logger.info(f"Target string found, but it's due to XORing with a region of consistent values")
                return True
        
        # Case 2: Check for other consistent patterns that would cause false positives
        # Examine the area around the target
        window_size = 100
        
        if target_pos - window_size >= 0 and target_pos + len(target_string) + window_size <= len(xor_text):
            start_idx = target_pos - window_size
            end_idx = target_pos + len(target_string) + window_size
            
            # Check the character distribution around the target
            char_counts = {}
            for i in range(start_idx, end_idx):
                c = xor_text[i]
                char_counts[c] = char_counts.get(c, 0) + 1
            
            # If character distribution is unusual (very few unique chars), likely a false positive
            if len(char_counts) < 20:  # Very low diversity of characters
                self.logger.info(f"Target string found in a region with suspicious character distribution")
                return True
            
            # Check if there's readable text around the target
            surrounding_text = xor_text[start_idx:end_idx]
            text_validation = FindingValidator.validate_text_content(surrounding_text)
            
            # If the surrounding text is not valid, target may be an artifact
            if not text_validation["valid"] and text_validation["confidence"] < 0.3:
                return True
        
        return False
    
    def _check_numeric_target_false_positive(self, flat_data: np.ndarray, num: int, 
                                           xor_text: str, target_string: str) -> bool:
        """
        Check if a target string match using numeric XOR is likely a false positive.
        
        Args:
            flat_data: Original image data
            num: XOR numeric key
            xor_text: XORed text result
            target_string: Target string that was found
            
        Returns:
            Boolean indicating if this is likely a false positive
        """
        target_pos = xor_text.find(target_string)
        
        # If the key is 0, everything stays the same
        if num == 0:
            self.logger.info("Target string found, but XOR key is 0 which doesn't change the data")
            return True
        
        # Check if the corresponding bytes in the original data have consistent values
        start_pos = target_pos
        value_counts = {}
        for i in range(len(target_string)):
            if start_pos + i < len(flat_data):
                val = flat_data[start_pos + i]
                value_counts[val] = value_counts.get(val, 0) + 1
        
        # If too few unique values or specific patterns, it's likely a false positive
        if len(value_counts) <= 5 or 0 in value_counts or num in value_counts:
            self.logger.info(f"Target string found, but it's due to XORing with a region of consistent values")
            return True
        
        # Check if there's readable text around the target
        window_size = 100
        if target_pos - window_size >= 0 and target_pos + len(target_string) + window_size <= len(xor_text):
            surrounding_text = xor_text[target_pos-window_size:target_pos+len(target_string)+window_size]
            text_validation = FindingValidator.validate_text_content(surrounding_text)
            
            # If the surrounding text is not valid, target may be an artifact
            if not text_validation["valid"] and text_validation["confidence"] < 0.3:
                return True
        
        return False
    
    def _check_clue_false_positive(self, xor_text: str, clue: str) -> bool:
        """
        Check if a clue match is likely a false positive.
        
        Args:
            xor_text: XORed text result
            clue: Clue string that was found
            
        Returns:
            Boolean indicating if this is likely a false positive
        """
        # Short clues like "4NBT" can appear randomly
        # Check how many times it appears - multiple occurrences are suspicious
        count = xor_text.count(clue)
        if count > 3:
            self.logger.info(f"Clue '{clue}' appears {count} times, probably random")
            return True
        
        # Check context around each occurrence
        positions = [m.start() for m in re.finditer(clue, xor_text)]
        valid_contexts = 0
        
        for pos in positions:
            # Get 50 chars before and after the clue
            start = max(0, pos - 50)
            end = min(len(xor_text), pos + len(clue) + 50)
            context = xor_text[start:end]
            
            # Validate the context
            validation = FindingValidator.validate_text_content(context)
            if validation["valid"]:
                valid_contexts += 1
        
        # If none of the occurrences have valid context, likely a false positive
        return valid_contexts == 0
    
    def _save_decoded_content(self, image_path: Path, key: str, decoded_text: str, target_pos: int = -1) -> None:
        """
        Save decoded content to files.
        
        Args:
            image_path: Path to the original image
            key: The key used for decoding
            decoded_text: The decoded text content
            target_pos: Position of target string (-1 if not found)
        """
        # Get the proper output directory
        output_dir = self.get_strategy_output_dir(image_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get unique image identifier using parent directory if available
        parent_dir = image_path.parent.name
        unique_image_id = f"{parent_dir}_{image_path.stem}" if parent_dir else image_path.stem
        
        # Save full decoded content
        decoded_file = output_dir / f"decoded_with_{key}.txt"
        with open(decoded_file, "w", encoding="utf-8") as f:
            f.write(f"=== DECODED CONTENT FROM {image_path.name} USING XOR KEY: {key} ===\n\n")
            f.write(decoded_text)
            
        # If target was found, save a highlighted version
        if target_pos >= 0:
            highlight_file = output_dir / f"decoded_with_{key}_highlighted.txt"
            target_string = "4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump"
            with open(highlight_file, "w", encoding="utf-8") as f:
                f.write(f"=== DECODED CONTENT FROM {image_path.name} WITH TARGET STRING HIGHLIGHTED ===\n\n")
                f.write(f"TARGET STRING: {target_string}\n")
                f.write(f"FOUND AT POSITION: {target_pos}\n\n")
                f.write(f"CONTEXT AROUND TARGET STRING:\n")
                f.write("-" * 80 + "\n")
                
                context_start = max(0, target_pos - 100)
                context_end = min(len(decoded_text), target_pos + len(target_string) + 200)
                f.write(decoded_text[context_start:context_end])
                
                f.write("\n" + "-" * 80 + "\n\n")
                f.write("FULL DECODED CONTENT:\n")
                f.write(decoded_text)
            
            self.logger.info(f"Saved highlighted content to {highlight_file}")
        
        self.logger.info(f"Saved decoded content to {decoded_file}")


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
            # Check if this strategy is applicable to the image
            strategy_validation = FindingValidator.validate_steganography_approach(self.name, image_path)
            if not strategy_validation["applicable"]:
                self.logger.info(f"ShiftCipherStrategy may not be suitable for {image_path}: {strategy_validation['reason']}")
            
            # For JPEGs, warn about potential artifacts
            is_jpeg = FindingValidator.is_jpeg_decompressed(image_path)
            if is_jpeg:
                self.logger.info(f"Image {image_path} is a JPEG. Shift cipher on decompressed data may produce artifacts.")
            
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
            highest_confidence = 0.0
            
            # Try each shift value, focusing on our key numbers
            shifts_to_try = list(range(1, 26)) + [4, 333 % 26, 353 % 26]
            shifts_to_try = sorted(list(set(shifts_to_try)))  # Remove duplicates
            
            # For JPEGs, limit the shifts to try to reduce false positives
            if is_jpeg:
                shifts_to_try = [4, 13, 333 % 26, 353 % 26]  # Key numbers plus ROT13
            
            for shift in shifts_to_try:
                shifted_text = self._apply_shift(text_data, shift)
                
                # Use the validator to check if result contains meaningful data
                validation = FindingValidator.validate_text_content(shifted_text)
                confidence = validation["confidence"]
                
                if validation["valid"]:
                    results[f"shift_{shift}"] = {
                        "readable_text": True,
                        "data": shifted_text[:1000],  # Limit output size
                        "validation": validation,
                        "confidence": confidence
                    }
                    found_something = True
                    highest_confidence = max(highest_confidence, confidence)
                
                # Check for target string
                target_string = "4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump"
                if target_string in shifted_text:
                    # For target string matches, we need more validation to avoid false positives
                    is_false_positive = self._check_target_false_positive(shifted_text, target_string)
                    
                    if not is_false_positive:
                        target_pos = shifted_text.find(target_string)
                        context_start = max(0, target_pos - 50)
                        context_end = min(len(shifted_text), target_pos + len(target_string) + 100)
                        context = shifted_text[context_start:context_end]
                        
                        results[f"shift_{shift}_found_key"] = {
                            "found_target": True,
                            "context": context,
                            "confidence": 0.9  # High confidence for target string
                        }
                        found_something = True
                        highest_confidence = 0.9
                        self.logger.warning(f"FOUND TARGET STRING with shift {shift}!")
                    else:
                        self.logger.info(f"Found target string with shift {shift}, but it appears to be a false positive")
                    
                # Also look for clues like '4NBT' or 'silver'
                for clue in ["4NBT", "silver"]:
                    if clue in shifted_text:
                        # Validate this is not just a coincidence
                        is_false_positive = self._check_clue_false_positive(shifted_text, clue)
                        
                        if not is_false_positive:
                            results[f"shift_{shift}_found_clue"] = {
                                "found_clue": True,
                                "clue": clue,
                                "context": shifted_text[:500],
                                "confidence": 0.7  # Good confidence for finding a clue
                            }
                            found_something = True
                            highest_confidence = max(highest_confidence, 0.7)
            
            # Add overall confidence score
            if found_something:
                results["overall_confidence"] = highest_confidence
                
                # For JPEGs, we need a higher confidence threshold
                if is_jpeg and highest_confidence < 0.7:
                    self.logger.info(f"Found potential matches in JPEG but confidence ({highest_confidence:.2f}) below threshold (0.7)")
                    return (False, {
                        "potential_findings": results,
                        "confidence": highest_confidence,
                        "message": "Potential findings in JPEG but confidence too low"
                    })
            
            return (highest_confidence > 0.5, results if found_something else None)
            
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
    
    def _check_target_false_positive(self, shifted_text: str, target_string: str) -> bool:
        """
        Check if a target string match is likely a false positive.
        
        Args:
            shifted_text: Shifted text result
            target_string: Target string that was found
            
        Returns:
            Boolean indicating if this is likely a false positive
        """
        target_pos = shifted_text.find(target_string)
        
        # Check for meaningful context around the target
        window_size = 100
        if target_pos - window_size >= 0 and target_pos + len(target_string) + window_size <= len(shifted_text):
            surrounding_text = shifted_text[target_pos-window_size:target_pos+len(target_string)+window_size]
            
            # Use validator to check the context
            validation = FindingValidator.validate_text_content(surrounding_text)
            
            # If the surrounding text is not valid, target may be an artifact
            if not validation["valid"] and validation["confidence"] < 0.3:
                return True
            
            # If surrounding text has almost no dictionary words, likely a false positive
            if validation["common_word_ratio"] < 0.1 and validation["common_word_count"] < 2:
                return True
        
        # Check if target string appears multiple times
        # Multiple occurrences of the same long string are suspicious
        if shifted_text.count(target_string) > 1:
            return True
        
        # Check if target string is the only readable part in the text
        overall_validation = FindingValidator.validate_text_content(
            shifted_text[:target_pos] + shifted_text[target_pos+len(target_string):]
        )
        
        # If rest of text is gibberish but target string is perfect, it's suspicious
        if not overall_validation["valid"] and overall_validation["confidence"] < 0.2:
            return True
        
        return False
    
    def _check_clue_false_positive(self, shifted_text: str, clue: str) -> bool:
        """
        Check if a clue match is likely a false positive.
        
        Args:
            shifted_text: Shifted text result
            clue: Clue string that was found
            
        Returns:
            Boolean indicating if this is likely a false positive
        """
        # Short clues can appear randomly
        # Check how many times it appears - multiple occurrences are suspicious
        count = shifted_text.count(clue)
        if count > 3:
            return True
        
        # Check context around each occurrence
        positions = [m.start() for m in re.finditer(clue, shifted_text)]
        valid_contexts = 0
        
        for pos in positions:
            # Get 50 chars before and after the clue
            start = max(0, pos - 50)
            end = min(len(shifted_text), pos + len(clue) + 50)
            context = shifted_text[start:end]
            
            # Validate the context
            validation = FindingValidator.validate_text_content(context)
            if validation["valid"]:
                valid_contexts += 1
        
        # If none of the occurrences have valid context, likely a false positive
        return valid_contexts == 0


class BlakeHashStrategy(StegStrategy):
    """
    Strategy that uses Blake hash functions to detect hidden data.
    
    This strategy applies Blake2b and Blake2s hash functions with various
    keys derived from keywords to try to reveal hidden data.
    """
    
    name: str = "blake_hash_strategy"
    description: str = "Blake hash function analysis with literary keys"
    
    def __init__(self) -> None:
        """Initialize the strategy."""
        super().__init__()
        
        # Load config
        config_path = Path(__file__).parent / "config" / "keywords.json"
        try:
            with open(config_path) as f:
                config = json.load(f)
            self.KEYWORDS = config.get("keywords", [])
            self.KEY_NUMBERS = config.get("key_numbers", [4, 333, 353])  # Default if not found
            self.SIGNIFICANCE_MARKERS = config.get("significance_markers", {})
        except Exception as e:
            self.logger.error(f"Error loading keywords config: {e}")
            # Fallback to defaults if config load fails
            self.KEYWORDS = ["4NBT", "silver", "YZY", "Tyger", "Blake"]
            self.KEY_NUMBERS = [4, 333, 353]
            self.SIGNIFICANCE_MARKERS = {
                "4nbt": 0.8,
                "yzy": 0.7,
                "silver": 0.7
            }
        
        # Initialize hash functions
        self.hash_functions = {
            "blake2b": hashlib.blake2b,
            "blake2s": hashlib.blake2s
        }
    
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
            # Check if this strategy is applicable to the image
            strategy_validation = FindingValidator.validate_steganography_approach(self.name, image_path)
            if not strategy_validation["applicable"]:
                self.logger.info(f"BlakeHashStrategy may not be suitable for {image_path}: {strategy_validation['reason']}")
            
            # For JPEGs, warn about potential artifacts
            is_jpeg = FindingValidator.is_jpeg_decompressed(image_path)
            if is_jpeg:
                self.logger.info(f"Image {image_path} is a JPEG. Hash operations on decompressed data may produce artifacts.")
            
            img = self.load_image(image_path)
            img_array = np.array(img)
            
            results = {}
            found_something = False
            highest_confidence = 0.0
            
            # Extract image data
            raw_data = self._extract_image_data(img_array)
            
            # Try Blake hash with key terms
            key_results = self._try_blake_hash_with_keys(raw_data)
            if key_results:
                results.update({
                    "key_hashing": key_results,
                    "confidence": key_results.get("confidence", 0.0)
                })
                found_something = True
                highest_confidence = max(highest_confidence, key_results.get("confidence", 0.0))
            
            # Only perform other analyses if key-based approach didn't find strong evidence
            # or if not dealing with JPEG (to reduce false positives in JPEGs)
            if not found_something or (not is_jpeg and highest_confidence < 0.7):
                # Don't analyze image regions for JPEGs (too prone to artifacts)
                if not is_jpeg:
                    # Analyze image regions
                    region_results = self._analyze_image_regions(img_array)
                    if region_results:
                        results.update({
                            "region_analysis": region_results,
                            "confidence": region_results.get("confidence", 0.0)
                        })
                        found_something = True
                        highest_confidence = max(highest_confidence, region_results.get("confidence", 0.0))
                
                # Try numeric hash analysis
                # Use a reduced set of options for JPEGs
                num_results = self._try_hash_with_numbers(raw_data, is_jpeg=is_jpeg)
                if num_results:
                    results.update({
                        "numeric_hashing": num_results,
                        "confidence": num_results.get("confidence", 0.0)
                    })
                    found_something = True
                    highest_confidence = max(highest_confidence, num_results.get("confidence", 0.0))
                
                # Analyze LSB data with Blake hash
                if not is_jpeg:  # Skip LSB analysis for JPEGs
                    lsb_results = self._analyze_lsb_with_blake(img_array)
                    if lsb_results:
                        results.update({
                            "lsb_blake_analysis": lsb_results,
                            "confidence": lsb_results.get("confidence", 0.0)
                        })
                        found_something = True
                        highest_confidence = max(highest_confidence, lsb_results.get("confidence", 0.0))
            
            # Add overall confidence
            if found_something:
                results["overall_confidence"] = highest_confidence
                
                # For JPEGs, we need a higher confidence threshold
                if is_jpeg and highest_confidence < 0.7:
                    self.logger.info(f"Found potential matches in JPEG but confidence ({highest_confidence:.2f}) below threshold (0.7)")
                    return (False, {
                        "potential_findings": results,
                        "confidence": highest_confidence,
                        "message": "Potential findings in JPEG but confidence too low"
                    })
            
            return (highest_confidence > 0.6, results if found_something else None)
            
        except Exception as e:
            self.logger.error(f"Error in Blake hash analysis: {e}")
            return (False, {"error": str(e)})
    
    def _extract_image_data(self, img_array: np.ndarray) -> bytes:
        """
        Extract binary data from image.
        
        Args:
            img_array: Numpy array representation of the image
            
        Returns:
            Binary data extracted from the image
        """
        # Convert image array to bytes
        if len(img_array.shape) == 3:  # Color image
            flat_data = img_array.reshape(-1)
        else:  # Grayscale
            flat_data = img_array.reshape(-1)
            
        return flat_data.tobytes()
    
    def _try_blake_hash_with_keys(self, data: bytes) -> Optional[Dict[str, Any]]:
        """
        Try various Blake hash operations with keywords.
        
        Args:
            data: Binary data to hash
            
        Returns:
            Optional dictionary with hash results
        """
        results = {}
        found_something = False
        highest_confidence = 0.0
        
        # Try Blake hash with each keyword
        for keyword in self.KEYWORDS:
            key = keyword.encode('utf-8')
            
            for hash_name, hash_func in self.hash_functions.items():
                try:
                    # Apply hash with key
                    hash_result = hash_func(data, key=key)
                    
                    # Check if hash reveals something interesting
                    significance = self._check_hash_significance(hash_result.hex(), f"{hash_name}_{keyword}")
                    if significance:
                        found_something = True
                        results[f"{hash_name}_{keyword}"] = significance
                        highest_confidence = max(highest_confidence, significance.get("confidence", 0.0))
                except Exception as e:
                    self.logger.debug(f"Error with {hash_name} and key {keyword}: {e}")
        
        if found_something:
            results["confidence"] = highest_confidence
            return results
        
        return None
    
    def _check_hash_significance(self, hash_str: str, source: str) -> Optional[Dict[str, Any]]:
        """
        Check if a hash result has significance to the investigation.
        
        Args:
            hash_str: Hexadecimal hash string
            source: Description of what produced this hash
            
        Returns:
            Optional dictionary with significance information
        """
        # Check for hex groups that might be ASCII
        ascii_text = ''
        for i in range(0, len(hash_str), 2):
            if i + 1 < len(hash_str):
                hex_val = hash_str[i:i+2]
                try:
                    char_val = int(hex_val, 16)
                    if 32 <= char_val <= 126:  # Printable ASCII
                        ascii_text += chr(char_val)
                    else:
                        ascii_text += '.'
                except:
                    ascii_text += '.'
        
        # Check if ASCII representation has meaningful content
        text_validation = FindingValidator.validate_text_content(ascii_text)
        
        # Look for significance markers
        found_markers = []
        confidence = 0.0
        
        for marker, confidence_value in self.SIGNIFICANCE_MARKERS.items():
            if marker in hash_str.lower():
                found_markers.append({
                    "marker": marker,
                    "confidence": confidence_value,
                    "position": hash_str.lower().find(marker)
                })
                confidence = max(confidence, confidence_value)
                
        # Check for 4NBT in the ASCII representation
        if '4NBT' in ascii_text:
            # This is stronger evidence
            found_markers.append({
                "marker": "4NBT in ASCII",
                "confidence": 0.8,
                "position": ascii_text.find('4NBT')
            })
            confidence = max(confidence, 0.8)
            
        # Check for the full investigation key
        target_string = "4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump"
        if target_string in ascii_text:
            found_markers.append({
                "marker": "Full target key in ASCII",
                "confidence": 0.95,
                "position": ascii_text.find(target_string)
            })
            confidence = max(confidence, 0.95)
            
            # This is a major finding - log it prominently
            self.logger.warning(f"FOUND TARGET STRING in hash with {source}!")
        
        # If ASCII content looks meaningful, that's significant too
        if text_validation["valid"]:
            found_markers.append({
                "marker": "Meaningful ASCII content",
                "confidence": text_validation["confidence"],
                "text": ascii_text,
                "validation": text_validation
            })
            confidence = max(confidence, text_validation["confidence"])
        
        # If meaningful markers found, return significance info
        if found_markers:
            return {
                "hash": hash_str,
                "ascii_interpretation": ascii_text,
                "significance_markers": found_markers,
                "confidence": confidence,
                "source": source
            }
        
        # If no significant markers or patterns, check if hash itself looks suspicious
        # This is a last resort and most prone to false positives
        if self._is_potential_hash(bytes.fromhex(hash_str)):
            return {
                "hash": hash_str,
                "ascii_interpretation": ascii_text,
                "significance_markers": [{"marker": "Unusual hash structure", "confidence": 0.3}],
                "confidence": 0.3,
                "source": source
            }
        
        return None
    
    def _analyze_image_regions(self, img_array: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Apply Blake hash analysis to different regions of the image.
        
        Args:
            img_array: Numpy array representation of the image
            
        Returns:
            Optional dictionary with analysis results
        """
        results = {}
        found_something = False
        highest_confidence = 0.0
        
        # Get image dimensions
        if len(img_array.shape) == 3:  # Color image
            height, width, _ = img_array.shape
        else:  # Grayscale
            height, width = img_array.shape
        
        # Skip regions analysis if image is too small
        if height < 50 or width < 50:
            return None
        
        # Define regions to analyze
        regions = {
            "top_left": (0, 0, width // 3, height // 3),
            "top_right": (width * 2 // 3, 0, width, height // 3),
            "bottom_left": (0, height * 2 // 3, width // 3, height),
            "bottom_right": (width * 2 // 3, height * 2 // 3, width, height),
            "center": (width // 3, height // 3, width * 2 // 3, height * 2 // 3)
        }
        
        # Analyze each region
        for region_name, (x1, y1, x2, y2) in regions.items():
            try:
                # Extract region data
                if len(img_array.shape) == 3:  # Color image
                    region_data = img_array[y1:y2, x1:x2].reshape(-1).tobytes()
                else:  # Grayscale
                    region_data = img_array[y1:y2, x1:x2].reshape(-1).tobytes()
                
                # Try hashing with Blake works
                for work in [self.KEYWORDS[0], self.KEYWORDS[-1]]:  # Just use first and last to reduce false positives
                    key = work.encode('utf-8')
                    
                    for hash_name, hash_func in self.hash_functions.items():
                        try:
                            # Apply hash with key
                            hash_result = hash_func(region_data, key=key)
                            
                            # Check if hash reveals something interesting
                            significance = self._check_hash_significance(hash_result.hex(), f"{region_name}_{hash_name}_{work}")
                            if significance:
                                found_something = True
                                results[f"{region_name}_{hash_name}_{work}"] = significance
                                highest_confidence = max(highest_confidence, significance.get("confidence", 0.0))
                        except Exception as e:
                            self.logger.debug(f"Error with {region_name} {hash_name} and key {work}: {e}")
            except Exception as e:
                self.logger.debug(f"Error analyzing region {region_name}: {e}")
        
        if found_something:
            results["confidence"] = highest_confidence
            return results
        
        return None
    
    def _try_hash_with_numbers(self, data: bytes, is_jpeg: bool = False) -> Optional[Dict[str, Any]]:
        """
        Try Blake hash operations with key numbers.
        
        Args:
            data: Binary data to hash
            is_jpeg: Whether the image is a JPEG
            
        Returns:
            Optional dictionary with hash results
        """
        results = {}
        found_something = False
        highest_confidence = 0.0
        
        # Use all key numbers for non-JPEGs, limited set for JPEGs
        key_numbers = [4] if is_jpeg else self.KEY_NUMBERS
        
        # Try different numeric keys
        for num in key_numbers:
            key = num.to_bytes(4, byteorder='big')
            
            for hash_name, hash_func in self.hash_functions.items():
                try:
                    # Apply hash with numeric key
                    hash_result = hash_func(data, key=key)
                    
                    # Check if hash reveals something interesting
                    significance = self._check_hash_significance(hash_result.hex(), f"{hash_name}_num_{num}")
                    if significance:
                        found_something = True
                        results[f"{hash_name}_num_{num}"] = significance
                        highest_confidence = max(highest_confidence, significance.get("confidence", 0.0))
                except Exception as e:
                    self.logger.debug(f"Error with {hash_name} and numeric key {num}: {e}")
            
            # For key number 4, which is most significant, try more variations
            if num == 4 and not is_jpeg:
                # Try XORing data with key before hashing
                try:
                    xor_data = bytearray()
                    for i in range(len(data)):
                        xor_data.append(data[i] ^ (num % 256))
                    
                    for hash_name, hash_func in self.hash_functions.items():
                        # Hash the XORed data
                        hash_result = hash_func(bytes(xor_data))
                        
                        # Check if hash reveals something interesting
                        significance = self._check_hash_significance(hash_result.hex(), f"{hash_name}_xor_{num}")
                        if significance:
                            found_something = True
                            results[f"{hash_name}_xor_{num}"] = significance
                            highest_confidence = max(highest_confidence, significance.get("confidence", 0.0))
                except Exception as e:
                    self.logger.debug(f"Error with XOR and numeric key {num}: {e}")
        
        if found_something:
            results["confidence"] = highest_confidence
            return results
        
        return None
    
    def _analyze_lsb_with_blake(self, img_array: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Extract LSB data and analyze it with Blake hash.
        
        Args:
            img_array: Numpy array representation of the image
            
        Returns:
            Optional dictionary with analysis results
        """
        results = {}
        found_something = False
        highest_confidence = 0.0
        
        try:
            # Extract LSB from each channel
            if len(img_array.shape) == 3:  # Color image
                height, width, channels = img_array.shape
                
                for channel in range(channels):
                    # Extract LSBs from this channel
                    lsb_data = bytearray()
                    for h in range(height):
                        for w in range(width):
                            lsb_data.append(img_array[h, w, channel] & 1)
                    
                    # Convert bits to bytes
                    byte_array = bytearray()
                    for i in range(0, len(lsb_data) - 7, 8):
                        byte_val = 0
                        for j in range(8):
                            if i + j < len(lsb_data):
                                byte_val |= lsb_data[i + j] << j
                        byte_array.append(byte_val)
                    
                    # Try Blake hash on LSB data
                    for hash_name, hash_func in self.hash_functions.items():
                        try:
                            hash_result = hash_func(bytes(byte_array))
                            
                            # Check if hash reveals something interesting
                            significance = self._check_hash_significance(hash_result.hex(), f"lsb_channel_{channel}_{hash_name}")
                            if significance:
                                found_something = True
                                results[f"lsb_channel_{channel}_{hash_name}"] = significance
                                highest_confidence = max(highest_confidence, significance.get("confidence", 0.0))
                        except Exception as e:
                            self.logger.debug(f"Error with LSB channel {channel} {hash_name}: {e}")
                    
                    # Try with key numbers
                    for num in self.KEY_NUMBERS:
                        key = num.to_bytes(4, byteorder='big')
                        try:
                            hash_result = hash_func(bytes(byte_array), key=key)
                            
                            # Check if hash reveals something interesting
                            significance = self._check_hash_significance(hash_result.hex(), f"lsb_channel_{channel}_{hash_name}_num_{num}")
                            if significance:
                                found_something = True
                                results[f"lsb_channel_{channel}_{hash_name}_num_{num}"] = significance
                                highest_confidence = max(highest_confidence, significance.get("confidence", 0.0))
                        except Exception as e:
                            self.logger.debug(f"Error with LSB channel {channel} {hash_name} num {num}: {e}")
            else:  # Grayscale
                height, width = img_array.shape
                
                # Extract LSBs
                lsb_data = bytearray()
                for h in range(height):
                    for w in range(width):
                        lsb_data.append(img_array[h, w] & 1)
                
                # Convert bits to bytes
                byte_array = bytearray()
                for i in range(0, len(lsb_data) - 7, 8):
                    byte_val = 0
                    for j in range(8):
                        if i + j < len(lsb_data):
                            byte_val |= lsb_data[i + j] << j
                    byte_array.append(byte_val)
                
                # Try Blake hash on LSB data
                for hash_name, hash_func in self.hash_functions.items():
                    try:
                        hash_result = hash_func(bytes(byte_array))
                        
                        # Check if hash reveals something interesting
                        significance = self._check_hash_significance(hash_result.hex(), f"lsb_grayscale_{hash_name}")
                        if significance:
                            found_something = True
                            results[f"lsb_grayscale_{hash_name}"] = significance
                            highest_confidence = max(highest_confidence, significance.get("confidence", 0.0))
                    except Exception as e:
                        self.logger.debug(f"Error with LSB grayscale {hash_name}: {e}")
                
                # Try with key numbers
                for num in self.KEY_NUMBERS:
                    key = num.to_bytes(4, byteorder='big')
                    try:
                        hash_result = hash_func(bytes(byte_array), key=key)
                        
                        # Check if hash reveals something interesting
                        significance = self._check_hash_significance(hash_result.hex(), f"lsb_grayscale_{hash_name}_num_{num}")
                        if significance:
                            found_something = True
                            results[f"lsb_grayscale_{hash_name}_num_{num}"] = significance
                            highest_confidence = max(highest_confidence, significance.get("confidence", 0.0))
                    except Exception as e:
                        self.logger.debug(f"Error with LSB grayscale {hash_name} num {num}: {e}")
        except Exception as e:
            self.logger.error(f"Error in LSB Blake analysis: {e}")
        
        if found_something:
            results["confidence"] = highest_confidence
            return results
        
        return None
    
    def _hash_similarity(self, hash1: bytes, hash2: bytes) -> float:
        """
        Calculate similarity between two hashes (0.0 to 1.0).
        
        Args:
            hash1: First hash
            hash2: Second hash
            
        Returns:
            Similarity score between 0.0 (different) and 1.0 (identical)
        """
        min_len = min(len(hash1), len(hash2))
        if min_len == 0:
            return 0.0
            
        matching_bytes = sum(1 for i in range(min_len) if hash1[i] == hash2[i])
        return matching_bytes / min_len
    
    def _is_potential_hash(self, data: bytes) -> bool:
        """
        Check if data might be a hash based on byte distribution.
        
        Args:
            data: Data to check
            
        Returns:
            Boolean indicating if data resembles a hash
        """
        if len(data) < 4:
            return False
            
        # Calculate byte distribution
        byte_counts = {}
        for b in data:
            byte_counts[b] = byte_counts.get(b, 0) + 1
            
        # Calculate entropy (randomness)
        entropy = 0.0
        for count in byte_counts.values():
            p = count / len(data)
            entropy -= p * np.log2(p) if p > 0 else 0
            
        # Most cryptographic hashes have high entropy (close to 8.0)
        if entropy > 7.0:
            return True
            
        # Check if there are recurring patterns that might indicate structure
        for length in range(2, min(8, len(data) // 2)):
            for start in range(len(data) - length * 2 + 1):
                pattern = data[start:start+length]
                if data.count(pattern) > 2:
                    return True
                    
        return False


class CustomRgbEncodingStrategy(StegStrategy):
    """
    Strategy to detect and extract data encoded with a custom RGB channel bit allocation.
    
    This strategy extracts bytes hidden in pixels where:
    - 3 bits are stored in the R channel
    - 3 bits are stored in the G channel
    - 2 bits are stored in the B channel
    It also looks for metadata in the format "filename|filetype|" at the beginning.
    
    NOTE: This strategy is inefficient for JPEG images because:
    1. JPEG uses lossy compression that works in the frequency domain (DCT coefficients)
    2. The 3-3-2 bit allocation pattern would be destroyed during JPEG compression
    3. It processes every pixel individually, making it extremely slow for large images
    4. Use JpegDomainStrategy instead for JPEG files
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


class JpegDomainStrategy(StegStrategy):
    """
    Strategy to analyze JPEG files in the compressed domain.
    
    This strategy works directly with the JPEG file structure, inspecting DCT coefficients
    and quantization tables to detect steganography techniques that operate in the 
    frequency domain rather than the pixel domain.
    """
    
    name: str = "jpeg_domain_strategy"
    description: str = "JPEG compressed domain analysis for steganography detection"
    
    def analyze(self, image_path: Path) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Analyze a JPEG image in the compressed domain.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple containing:
                - Boolean indicating if hidden data was detected
                - Optional dictionary with analysis results
        """
        try:
            # First, check if file is a JPEG
            if not self._is_jpeg(image_path):
                self.logger.info(f"{image_path} is not a JPEG file, skipping JPEG domain analysis")
                return (False, {"error": "Not a JPEG file"})
            
            # Parse JPEG structure and extract DCT coefficients
            jpeg_data = self._parse_jpeg(image_path)
            if not jpeg_data:
                return (False, {"error": "Failed to parse JPEG structure"})
            
            results = {}
            found_something = False
            confidence_score = 0.0
            
            # Check for data after EOI marker (FF D9) - this is a very reliable indicator
            with open(image_path, 'rb') as f:
                data = f.read()
                
            eoi_position = self._find_last_jpeg_eoi(data)
            if eoi_position > 0 and eoi_position + 2 < len(data):
                # There's data after the EOI marker - highly suspicious
                trailer_data = data[eoi_position + 2:]
                
                if len(trailer_data) > 4:  # Require at least a few bytes to avoid false positives
                    results["after_eoi_data"] = {
                        "position": eoi_position + 2,
                        "size": len(trailer_data),
                        "data_sample": trailer_data[:100].hex(),
                        "confidence": 0.9
                    }
                    
                    # Validate the data to reduce false positives
                    validation = FindingValidator.validate_binary_content(trailer_data)
                    results["after_eoi_data"]["validation"] = validation
                    
                    if validation["valid"]:
                        confidence_score = max(confidence_score, 0.95)
                        found_something = True
                    else:
                        # Even with invalid content, appended data is suspicious
                        confidence_score = max(confidence_score, 0.7)
                        found_something = True
            
            # Run various analyses on the JPEG data
            dct_analysis = self._analyze_dct_coefficients(jpeg_data)
            if dct_analysis:
                results["dct_analysis"] = dct_analysis
                found_something = True
                confidence_score = max(confidence_score, dct_analysis.get("confidence", 0.0))
            
            # Perform chi-square analysis on DCT coefficients
            chi_square_result = self._perform_chi_square_analysis(jpeg_data)
            if chi_square_result:
                results["chi_square_analysis"] = chi_square_result
                if chi_square_result.get("suspicious", False):
                    found_something = True
                    confidence_score = max(confidence_score, chi_square_result.get("confidence", 0.0))
            
            # Analyze quantization tables
            quant_analysis = self._analyze_quantization_tables(jpeg_data)
            if quant_analysis:
                results["quantization_analysis"] = quant_analysis
                found_something = True
                confidence_score = max(confidence_score, 0.6)  # Quantization anomalies are good indicators
            
            # Perform histogram analysis
            histogram_analysis = self._analyze_dct_histograms(jpeg_data)
            if histogram_analysis:
                results["histogram_analysis"] = histogram_analysis
                if histogram_analysis.get("suspicious", False):
                    found_something = True
                    confidence_score = max(confidence_score, histogram_analysis.get("confidence", 0.0))
            
            # Perform calibration to reduce false positives
            if found_something:
                calibration_results = self._perform_calibration(image_path, jpeg_data)
                results["calibration"] = calibration_results
                
                # If calibration suggests all findings are false positives, adjust the found_something flag
                if calibration_results.get("all_findings_explained", False):
                    # Lower confidence but don't completely eliminate if we found EOI data
                    if "after_eoi_data" in results:
                        confidence_score = max(0.7, confidence_score * 0.8)
                    else:
                        confidence_score *= 0.5
                        if confidence_score < 0.4:
                            found_something = False
            
            results["confidence_score"] = confidence_score
            return (found_something, results)
            
        except Exception as e:
            self.logger.error(f"Error in JPEG domain analysis: {e}")
            return (False, {"error": str(e)})
    
    def _find_last_jpeg_eoi(self, data: bytes) -> int:
        """
        Find the position of the last JPEG EOI (End of Image) marker.
        
        Args:
            data: Binary data to search
            
        Returns:
            Position of the EOI marker, or -1 if not found
        """
        # Start search from near the end and work backwards
        # Many steganography tools append data after the EOI marker
        pos = len(data) - 2
        while pos >= 0:
            if data[pos] == 0xFF and data[pos + 1] == 0xD9:
                return pos
            pos -= 1
        
        return -1
    
    def _perform_chi_square_analysis(self, jpeg_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Perform chi-square analysis on DCT coefficients.
        
        This method looks for statistical anomalies in the distribution of 
        coefficient values that may indicate steganography.
        
        Args:
            jpeg_data: Parsed JPEG structure data
            
        Returns:
            Analysis results or None if analysis is not applicable
        """
        try:
            if "dct_coefficients" not in jpeg_data or not jpeg_data["dct_coefficients"]:
                return None
                
            dct_stats = jpeg_data["dct_coefficients"]
            
            # Extract byte frequencies if available
            if "byte_frequencies" not in dct_stats:
                return None
                
            # Convert string keys back to integers
            byte_freqs = {}
            for k, v in dct_stats["byte_frequencies"].items():
                byte_freqs[int(k)] = v
                
            # Perform chi-square test on pairs of values
            # In unmodified JPEGs, adjacent DCT coefficient values often follow
            # expected distributions. Steganography can disrupt these patterns.
            chi_square_values = []
            suspicious_pairs = []
            
            # We'll analyze distributions of even vs odd values
            even_values = {k: v for k, v in byte_freqs.items() if k % 2 == 0}
            odd_values = {k: v for k, v in byte_freqs.items() if k % 2 == 1}
            
            # Compute means to create pair expectations
            total_even = sum(even_values.values())
            total_odd = sum(odd_values.values())
            
            # Skip if insufficient data
            if total_even == 0 or total_odd == 0:
                return None
                
            # For each adjacent pair of values
            for even_val in range(0, 250, 2):
                observed_even = even_values.get(even_val, 0)
                observed_odd = odd_values.get(even_val + 1, 0)
                
                # Expected counts in unmodified image
                total_pair = observed_even + observed_odd
                if total_pair < 5:  # Skip pairs with too few occurrences
                    continue
                    
                expected_even = total_pair / 2
                expected_odd = total_pair / 2
                
                # Chi-square statistic
                if expected_even > 0 and expected_odd > 0:
                    chi_square = ((observed_even - expected_even) ** 2) / expected_even + \
                                 ((observed_odd - expected_odd) ** 2) / expected_odd
                    
                    chi_square_values.append(chi_square)
                    
                    # Values above 3.84 are significant at p=0.05
                    if chi_square > 3.84:
                        suspicious_pairs.append({
                            "even_value": even_val,
                            "odd_value": even_val + 1,
                            "observed_even": observed_even,
                            "observed_odd": observed_odd,
                            "expected_even": expected_even,
                            "expected_odd": expected_odd,
                            "chi_square": chi_square
                        })
            
            # Overall analysis
            if not chi_square_values:
                return None
                
            avg_chi_square = sum(chi_square_values) / len(chi_square_values)
            suspicious_count = sum(1 for chi in chi_square_values if chi > 3.84)
            suspicious_ratio = suspicious_count / len(chi_square_values) if chi_square_values else 0
            
            # Determine if the results are suspicious
            is_suspicious = avg_chi_square > 5.0 or suspicious_ratio > 0.2
            
            # Calculate confidence score based on chi-square results
            confidence = min(0.9, (avg_chi_square / 10.0) + (suspicious_ratio * 0.5))
            
            return {
                "avg_chi_square": avg_chi_square,
                "suspicious_pairs_count": suspicious_count,
                "total_pairs_count": len(chi_square_values),
                "suspicious_ratio": suspicious_ratio,
                "suspicious": is_suspicious,
                "confidence": confidence,
                "suspicious_pairs": suspicious_pairs[:10]  # Limit to top 10
            }
            
        except Exception as e:
            self.logger.error(f"Error in chi-square analysis: {e}")
            return None
    
    def _analyze_dct_histograms(self, jpeg_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze histograms of DCT coefficients for signs of steganography.
        
        Args:
            jpeg_data: Parsed JPEG data
            
        Returns:
            Analysis results or None if analysis is not applicable
        """
        try:
            if "dct_coefficients" not in jpeg_data or not jpeg_data["dct_coefficients"]:
                return None
                
            dct_stats = jpeg_data["dct_coefficients"]
            
            # Check for byte frequencies
            if "byte_frequencies" not in dct_stats:
                return None
                
            # Convert string keys back to integers
            byte_freqs = {}
            for k, v in dct_stats["byte_frequencies"].items():
                byte_freqs[int(k)] = v
                
            # In JPEG steganography, coefficient histograms often show:
            # 1. Unexpected peaks at 0, 1 values
            # 2. Unusual distributions of even vs odd coefficients
            
            # Analyze distribution of values close to zero
            near_zero_counts = {k: byte_freqs.get(k, 0) for k in range(-5, 6)}
            
            # Check for suspicious patterns in near-zero distribution
            suspicious_patterns = []
            
            # Check ratio of -1:0:1 values
            zero_count = near_zero_counts.get(0, 0)
            minus_one_count = near_zero_counts.get(-1, 0)
            plus_one_count = near_zero_counts.get(1, 0)
            
            # For normal JPEG compression, we expect a smooth distribution
            # Steganography often disrupts this, especially for values -1, 0, 1
            if zero_count > 0:
                minus_one_ratio = minus_one_count / zero_count if zero_count else 0
                plus_one_ratio = plus_one_count / zero_count if zero_count else 0
                
                # Check for asymmetry in -1/+1 distribution
                if abs(minus_one_ratio - plus_one_ratio) > 0.2:
                    suspicious_patterns.append({
                        "type": "asymmetric_near_zero",
                        "minus_one_ratio": minus_one_ratio,
                        "plus_one_ratio": plus_one_ratio,
                        "difference": abs(minus_one_ratio - plus_one_ratio)
                    })
            
            # Analyze even vs odd distribution
            even_sum = sum(byte_freqs.get(i, 0) for i in range(-50, 51, 2))
            odd_sum = sum(byte_freqs.get(i, 0) for i in range(-49, 51, 2))
            
            even_odd_ratio = even_sum / (even_sum + odd_sum) if (even_sum + odd_sum) > 0 else 0.5
            
            # In normal JPEGs, this ratio is typically close to 0.5
            # Significant deviation suggests steganography
            if abs(even_odd_ratio - 0.5) > 0.1:
                suspicious_patterns.append({
                    "type": "even_odd_imbalance",
                    "even_odd_ratio": even_odd_ratio,
                    "deviation": abs(even_odd_ratio - 0.5)
                })
            
            # Calculate overall confidence based on the suspicious patterns
            confidence = 0.0
            for pattern in suspicious_patterns:
                if pattern["type"] == "asymmetric_near_zero":
                    confidence = max(confidence, min(0.8, pattern["difference"] * 2))
                elif pattern["type"] == "even_odd_imbalance":
                    confidence = max(confidence, min(0.8, pattern["deviation"] * 4))
            
            return {
                "near_zero_distribution": {k: v for k, v in near_zero_counts.items()},
                "even_odd_ratio": even_odd_ratio,
                "suspicious_patterns": suspicious_patterns,
                "suspicious": bool(suspicious_patterns),
                "confidence": confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error in DCT histogram analysis: {e}")
            return None
    
    def _is_jpeg(self, image_path: Path) -> bool:
        """
        Check if a file is a JPEG image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Boolean indicating if the file is a JPEG
        """
        try:
            with open(image_path, 'rb') as f:
                # JPEG files start with SOI marker (FF D8)
                header = f.read(2)
                return header == b'\xFF\xD8'
        except Exception as e:
            self.logger.error(f"Error checking if {image_path} is JPEG: {e}")
            return False
    
    def _parse_jpeg(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """
        Parse JPEG file and extract its structure.
        
        This function reads the JPEG file segments including quantization tables,
        Huffman tables, and DCT coefficients.
        
        Args:
            image_path: Path to the JPEG file
            
        Returns:
            Dictionary containing JPEG structure information or None if parsing failed
        """
        try:
            jpeg_data = {
                "quantization_tables": [],
                "huffman_tables": [],
                "dct_coefficients": [],
                "segments": []
            }
            
            with open(image_path, 'rb') as f:
                data = f.read()
            
            # Parse JPEG segments
            pos = 0
            while pos < len(data) - 1:
                # Look for marker (0xFF followed by marker ID)
                if data[pos] == 0xFF and data[pos + 1] != 0x00:
                    marker = data[pos + 1]
                    
                    # Store segment info
                    segment = {"marker": marker, "offset": pos}
                    
                    # Handle different segment types
                    if marker == 0xDB:  # DQT (Define Quantization Table)
                        # Get segment length
                        length = (data[pos + 2] << 8) | data[pos + 3]
                        segment["length"] = length
                        
                        # Extract quantization table data
                        table_data = data[pos + 4:pos + 2 + length]
                        quant_table = self._parse_quantization_table(table_data)
                        if quant_table:
                            jpeg_data["quantization_tables"].append(quant_table)
                            segment["table_id"] = quant_table["table_id"]
                        
                        pos += 2 + length
                    
                    elif marker == 0xC0 or marker == 0xC2:  # SOF0 or SOF2 (Start of Frame)
                        # Get segment length
                        length = (data[pos + 2] << 8) | data[pos + 3]
                        segment["length"] = length
                        
                        # Extract frame parameters
                        precision = data[pos + 4]
                        height = (data[pos + 5] << 8) | data[pos + 6]
                        width = (data[pos + 7] << 8) | data[pos + 8]
                        components = data[pos + 9]
                        
                        segment["frame_info"] = {
                            "precision": precision,
                            "height": height,
                            "width": width,
                            "components": components
                        }
                        
                        pos += 2 + length
                    
                    elif marker == 0xC4:  # DHT (Define Huffman Table)
                        # Get segment length
                        length = (data[pos + 2] << 8) | data[pos + 3]
                        segment["length"] = length
                        
                        # Extract Huffman table data
                        table_data = data[pos + 4:pos + 2 + length]
                        huffman_table = self._parse_huffman_table(table_data)
                        if huffman_table:
                            jpeg_data["huffman_tables"].append(huffman_table)
                        
                        pos += 2 + length
                    
                    elif marker == 0xDA:  # SOS (Start of Scan)
                        # Get segment length
                        length = (data[pos + 2] << 8) | data[pos + 3]
                        segment["length"] = length
                        
                        # Skip past SOS marker to find compressed data
                        pos += 2 + length
                        
                        # Read compressed data (until next marker)
                        scan_start = pos
                        while pos < len(data) - 1:
                            if data[pos] == 0xFF and data[pos + 1] != 0x00:
                                break
                            pos += 1
                        
                        # Store scan data info
                        segment["scan_data_offset"] = scan_start
                        segment["scan_data_length"] = pos - scan_start
                        
                        # Extract DCT coefficients (this requires deeper parsing)
                        # For now, we'll just store basic statistics about the compressed data
                        scan_data = data[scan_start:pos]
                        jpeg_data["dct_coefficients"] = self._extract_dct_statistics(scan_data)
                    
                    elif marker == 0xD9:  # EOI (End of Image)
                        segment["is_end"] = True
                        pos += 2
                        break
                    
                    else:
                        # Handle other markers
                        if marker >= 0xE0 and marker <= 0xEF:  # APP markers
                            # Get segment length
                            length = (data[pos + 2] << 8) | data[pos + 3]
                            segment["length"] = length
                            pos += 2 + length
                        else:
                            # Unknown or unhandled marker, just move past it
                            if pos + 2 < len(data) and data[pos + 2] == 0xFF:
                                # No length field, just a marker
                                pos += 2
                            elif pos + 3 < len(data):
                                # Has length field
                                length = (data[pos + 2] << 8) | data[pos + 3]
                                segment["length"] = length
                                pos += 2 + length
                            else:
                                # End of data
                                pos += 2
                    
                    # Add segment to list
                    jpeg_data["segments"].append(segment)
                else:
                    pos += 1
            
            return jpeg_data
            
        except Exception as e:
            self.logger.error(f"Error parsing JPEG file {image_path}: {e}")
            return None
    
    def _parse_quantization_table(self, data: bytes) -> Optional[Dict[str, Any]]:
        """
        Parse a JPEG quantization table segment.
        
        Args:
            data: Bytes containing the quantization table data
            
        Returns:
            Dictionary with parsed quantization table or None
        """
        try:
            if len(data) < 1:
                return None
                
            # First byte contains table precision and ID
            precision_and_id = data[0]
            table_id = precision_and_id & 0x0F
            precision = (precision_and_id >> 4) & 0x0F
            
            # Parse table values
            table_size = 64  # 8x8 table
            table_values = []
            
            offset = 1  # Skip the first byte
            for i in range(table_size):
                if precision == 0:  # 8-bit values
                    if offset < len(data):
                        table_values.append(data[offset])
                        offset += 1
                else:  # 16-bit values
                    if offset + 1 < len(data):
                        value = (data[offset] << 8) | data[offset + 1]
                        table_values.append(value)
                        offset += 2
            
            # Reshape to 8x8 matrix in zigzag order
            zigzag_order = [
                0,  1,  5,  6,  14, 15, 27, 28,
                2,  4,  7,  13, 16, 26, 29, 42,
                3,  8,  12, 17, 25, 30, 41, 43,
                9,  11, 18, 24, 31, 40, 44, 53,
                10, 19, 23, 32, 39, 45, 52, 54,
                20, 22, 33, 38, 46, 51, 55, 60,
                21, 34, 37, 47, 50, 56, 59, 61,
                35, 36, 48, 49, 57, 58, 62, 63
            ]
            
            matrix = [0] * 64
            for i in range(min(len(table_values), 64)):
                matrix[zigzag_order[i]] = table_values[i]
            
            # Reshape to 8x8
            matrix_8x8 = []
            for i in range(0, 64, 8):
                matrix_8x8.append(matrix[i:i+8])
            
            return {
                "table_id": table_id,
                "precision": precision,
                "values": table_values,
                "matrix": matrix_8x8
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing quantization table: {e}")
            return None
    
    def _parse_huffman_table(self, data: bytes) -> Optional[Dict[str, Any]]:
        """
        Parse a JPEG Huffman table segment.
        
        Args:
            data: Bytes containing the Huffman table data
            
        Returns:
            Dictionary with parsed Huffman table or None
        """
        try:
            if len(data) < 1:
                return None
                
            # First byte contains table class and ID
            class_and_id = data[0]
            table_id = class_and_id & 0x0F
            table_class = (class_and_id >> 4) & 0x0F  # 0 = DC, 1 = AC
            
            # Read code lengths
            code_lengths = []
            for i in range(16):
                if i + 1 < len(data):
                    code_lengths.append(data[i + 1])
            
            # Simple statistical analysis rather than full Huffman tree construction
            return {
                "table_id": table_id,
                "table_class": table_class,
                "code_lengths": code_lengths,
                "total_codes": sum(code_lengths)
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing Huffman table: {e}")
            return None
    
    def _extract_dct_statistics(self, scan_data: bytes) -> Dict[str, Any]:
        """
        Extract statistical information about DCT coefficients.
        
        Note: Full DCT coefficient extraction requires Huffman decoding,
        which is complex. Here we perform statistical analysis on the
        compressed data to look for anomalies.
        
        Args:
            scan_data: JPEG scan data containing Huffman-coded DCT coefficients
            
        Returns:
            Dictionary with DCT coefficient statistics
        """
        # For now, just collect basic statistics about the scan data
        byte_frequencies = {}
        for b in scan_data:
            byte_frequencies[b] = byte_frequencies.get(b, 0) + 1
        
        # Calculate entropy
        total_bytes = len(scan_data)
        entropy = 0
        for count in byte_frequencies.values():
            probability = count / total_bytes
            entropy -= probability * np.log2(probability)
        
        # Look for unusual patterns in byte sequences
        unusual_sequences = []
        for pattern_len in range(3, 6):
            pattern_counts = {}
            for i in range(len(scan_data) - pattern_len):
                pattern = scan_data[i:i+pattern_len]
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            # Find unusually frequent patterns
            for pattern, count in pattern_counts.items():
                # Expected frequency for random data
                expected = total_bytes / (256 ** pattern_len)
                if count > expected * 5:  # 5x more frequent than expected
                    unusual_sequences.append({
                        "pattern": pattern.hex(),
                        "count": count,
                        "expected": expected,
                        "ratio": count / expected
                    })
        
        return {
            "total_bytes": total_bytes,
            "entropy": entropy,
            "byte_frequencies": {str(k): v for k, v in sorted(byte_frequencies.items(), key=lambda x: x[1], reverse=True)[:20]},  # Top 20 bytes
            "unusual_sequences": unusual_sequences[:10]  # Top 10 unusual sequences
        }
    
    def _analyze_dct_coefficients(self, jpeg_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze DCT coefficient statistics for signs of steganography.
        
        Args:
            jpeg_data: Parsed JPEG structure
            
        Returns:
            Dictionary with analysis results or None
        """
        if "dct_coefficients" not in jpeg_data:
            return None
        
        dct_stats = jpeg_data["dct_coefficients"]
        results = {}
        found_something = False
        
        # Check entropy - compressed data should have high entropy
        # Too low entropy might indicate data has been tampered with
        if "entropy" in dct_stats:
            entropy = dct_stats["entropy"]
            results["entropy"] = entropy
            
            # For compressed data, entropy should be close to 8 bits
            if entropy < 7.0:
                results["low_entropy"] = {
                    "value": entropy,
                    "suspicion": "Compressed data has unusually low entropy"
                }
                found_something = True
        
        # Check for unusual sequences
        if "unusual_sequences" in dct_stats and dct_stats["unusual_sequences"]:
            results["unusual_sequences"] = dct_stats["unusual_sequences"]
            found_something = True
        
        # Complex patterns can indicate hidden data
        # Note: More sophisticated analysis would require decoding the DCT coefficients
        
        return results if found_something else None
    
    def _analyze_quantization_tables(self, jpeg_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze quantization tables for signs of steganography or manipulation.
        
        Args:
            jpeg_data: Parsed JPEG structure
            
        Returns:
            Dictionary with analysis results or None
        """
        if "quantization_tables" not in jpeg_data or not jpeg_data["quantization_tables"]:
            return None
        
        results = {}
        found_something = False
        
        # Analyze each quantization table
        for i, table in enumerate(jpeg_data["quantization_tables"]):
            if "matrix" not in table:
                continue
                
            matrix = table["matrix"]
            table_analysis = {}
            
            # Check for unusual values in the quantization table
            # Standard JPEG encoders use specific tables, so deviations might indicate tampering
            
            # Check for unusual high-frequency quantization values
            # Typically, high-frequency coefficients have larger quantization values
            high_freq_values = []
            for row in range(4, 8):
                for col in range(4, 8):
                    if row < len(matrix) and col < len(matrix[row]):
                        high_freq_values.append(matrix[row][col])
            
            # Calculate statistics
            if high_freq_values:
                avg_high_freq = sum(high_freq_values) / len(high_freq_values)
                min_high_freq = min(high_freq_values)
                
                # Check for unusually small high-frequency quantization values
                # These might indicate hidden data in high-frequency coefficients
                if min_high_freq < 10 or avg_high_freq < 20:
                    table_analysis["unusual_high_freq"] = {
                        "min_value": min_high_freq,
                        "avg_value": avg_high_freq,
                        "suspicion": "Unusually small high-frequency quantization values"
                    }
                    found_something = True
            
            # Check for non-standard table structure
            # Most common encoders use tables with a specific structure
            # where values increase as you move away from the DC coefficient
            unusual_structure = False
            prev_val = 0
            unusual_positions = []
            
            # Check zigzag pattern
            zigzag_order = [
                (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
                (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
                (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
                (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
                (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
                (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
                (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
                (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
            ]
            
            for idx, (row, col) in enumerate(zigzag_order):
                if row < len(matrix) and col < len(matrix[row]):
                    curr_val = matrix[row][col]
                    
                    # In normal tables, values generally increase along the zigzag
                    # with occasional small decreases
                    if idx > 0 and curr_val < prev_val * 0.5 and idx > 5:
                        unusual_structure = True
                        unusual_positions.append((row, col, curr_val, prev_val))
                    
                    prev_val = curr_val
            
            if unusual_structure:
                table_analysis["unusual_structure"] = {
                    "positions": unusual_positions,
                    "suspicion": "Quantization table has unusual structure"
                }
                found_something = True
            
            if table_analysis:
                results[f"table_{i}"] = table_analysis
        
        return results if found_something else None
    
    def _perform_calibration(self, image_path: Path, jpeg_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform calibration to reduce false positives.
        
        This technique compares the suspect image to a re-compressed version
        to distinguish artifacts from actual hidden data.
        
        Args:
            image_path: Path to the original image
            jpeg_data: Parsed JPEG structure
            
        Returns:
            Dictionary with calibration results
        """
        try:
            # Create a calibration image by slight re-compression
            calibration_results = {"performed": True}
            
            # Load the image with PIL
            img = self.load_image(image_path)
            
            # Create a temporary file for the calibrated image
            output_dir = self.get_strategy_output_dir(image_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            calibrated_path = output_dir / f"calibrated_{image_path.name}"
            
            # Save with slight re-compression
            img.save(calibrated_path, "JPEG", quality=92)
            
            # Parse the re-compressed image
            calibrated_jpeg_data = self._parse_jpeg(calibrated_path)
            
            if not calibrated_jpeg_data:
                calibration_results["success"] = False
                calibration_results["error"] = "Failed to parse calibrated image"
                return calibration_results
            
            # Compare quantization tables
            all_findings_explained = True
            
            if "quantization_tables" in jpeg_data and "quantization_tables" in calibrated_jpeg_data:
                orig_tables = jpeg_data["quantization_tables"]
                calib_tables = calibrated_jpeg_data["quantization_tables"]
                
                if len(orig_tables) == len(calib_tables):
                    table_comparisons = []
                    
                    for i in range(len(orig_tables)):
                        if "matrix" in orig_tables[i] and "matrix" in calib_tables[i]:
                            orig_matrix = orig_tables[i]["matrix"]
                            calib_matrix = calib_tables[i]["matrix"]
                            
                            # Calculate the difference
                            differences = []
                            for row in range(min(len(orig_matrix), len(calib_matrix))):
                                row_diffs = []
                                for col in range(min(len(orig_matrix[row]), len(calib_matrix[row]))):
                                    diff = abs(orig_matrix[row][col] - calib_matrix[row][col])
                                    row_diffs.append(diff)
                                differences.append(row_diffs)
                            
                            # If differences are significant in high frequencies, the findings
                            # might not just be compression artifacts
                            high_freq_diffs = []
                            for row in range(4, min(len(differences), 8)):
                                for col in range(4, min(len(differences[row]), 8)):
                                    high_freq_diffs.append(differences[row][col])
                            
                            avg_high_freq_diff = sum(high_freq_diffs) / len(high_freq_diffs) if high_freq_diffs else 0
                            
                            if avg_high_freq_diff > 5:
                                # Significant differences in high frequencies might indicate steganography
                                all_findings_explained = False
                            
                            table_comparisons.append({
                                "table_id": i,
                                "avg_high_freq_diff": avg_high_freq_diff,
                                "significant_difference": avg_high_freq_diff > 5
                            })
                    
                    calibration_results["table_comparisons"] = table_comparisons
            
            # Compare DCT coefficient statistics
            if "dct_coefficients" in jpeg_data and "dct_coefficients" in calibrated_jpeg_data:
                orig_stats = jpeg_data["dct_coefficients"]
                calib_stats = calibrated_jpeg_data["dct_coefficients"]
                
                # Compare entropy
                if "entropy" in orig_stats and "entropy" in calib_stats:
                    entropy_diff = abs(orig_stats["entropy"] - calib_stats["entropy"])
                    calibration_results["entropy_difference"] = entropy_diff
                    
                    # If entropy difference is significant, might indicate hidden data
                    if entropy_diff > 0.3:
                        all_findings_explained = False
            
            # Clean up the temporary file
            try:
                calibrated_path.unlink()
            except:
                pass
            
            calibration_results["all_findings_explained"] = all_findings_explained
            calibration_results["success"] = True
            
            return calibration_results
            
        except Exception as e:
            self.logger.error(f"Error performing calibration: {e}")
            return {
                "performed": True,
                "success": False,
                "error": str(e)
            }


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
            
            # Check if this is a JPEG file - handle specially
            is_jpeg = False
            eoi_position = -1
            results = {}
            confidence_score = 0.0
            
            if data.startswith(b'\xFF\xD8\xFF'):
                is_jpeg = True
                # Find JPEG EOI (End of Image) marker - 0xFF 0xD9
                eoi_position = self._find_last_jpeg_eoi(data)
                
                # Check for data after the EOI marker
                if eoi_position > 0 and eoi_position + 2 < len(data):
                    # There's data after the EOI marker - highly suspicious
                    trailer_data = data[eoi_position + 2:]
                    
                    if len(trailer_data) > 4:  # Require at least a few bytes to avoid false positives
                        results["after_eoi_data"] = {
                            "position": eoi_position + 2,
                            "size": len(trailer_data),
                            "file_type": self._identify_appended_data_type(trailer_data),
                            "confidence": 0.9
                        }
                        
                        # Extract the appended data properly
                        extracted_data = self._extract_appended_data(trailer_data)
                        
                        # Validate the extracted data
                        validation = FindingValidator.validate_binary_content(extracted_data)
                        results["after_eoi_data"]["validation"] = validation
                        results["after_eoi_data"]["data_sample"] = trailer_data[:100].hex()
                        
                        # If data is valid structured data or contains a file signature, increase confidence
                        if validation["valid"]:
                            confidence_score = 0.95
                            results["extracted_data"] = {
                                "type": "binary",
                                "encoding": "base64",
                                "file_type": validation.get("file_type", "unknown"),
                                "data": base64.b64encode(extracted_data[:1024*1024]).decode('ascii')  # Limit to 1MB
                            }
                            return (True, results)
                        else:
                            # Even with invalid content, appended data is suspicious
                            confidence_score = 0.7
            
            # Find all file signatures in the data
            found_signatures = []
            valid_regions = []
            
            # For JPEG files, we need to be smarter about where we look
            if is_jpeg:
                # We'll define regions to exclude (valid JPEG segments)
                # and only search in areas that aren't part of the regular JPEG structure
                valid_regions = self._get_valid_jpeg_regions(data)
            
            for signature, file_type in self.FILE_SIGNATURES.items():
                # Skip signatures that match the file's own type
                if is_jpeg and file_type in ['jpg', 'jpeg'] and data.startswith(signature):
                    continue
                
                # For non-JPEGs or for JPEGs when looking for non-JPEG signatures
                pos = 0
                while True:
                    pos = data.find(signature, pos)
                    if pos == -1:
                        break
                    
                    # For JPEGs, check if this position is inside a valid JPEG segment
                    if is_jpeg:
                        if any(start <= pos <= end for start, end in valid_regions):
                            # This signature is in a valid JPEG segment, likely a false positive
                            pos += len(signature)
                            continue
                    else:
                        # For non-JPEGs, skip the first few bytes to avoid detecting the file's own signature
                        if pos < len(signature) + 10:
                            pos += len(signature)
                            continue
                    
                    # Check context around the signature to reduce false positives
                    confidence = self._evaluate_signature_confidence(data, pos, signature, file_type)
                    
                    found_signatures.append({
                        "position": pos,
                        "signature": signature.hex(),
                        "file_type": file_type,
                        "confidence": confidence
                    })
                    pos += len(signature)
            
            # Filter out low-confidence signatures
            high_confidence_signatures = [sig for sig in found_signatures if sig["confidence"] > 0.5]
            
            # Extract potential hidden file content
            extracted_files = []
            for sig_info in high_confidence_signatures:
                # Try to extract content after the signature
                start = sig_info["position"]
                
                # For some file types, we know their structure and can extract them correctly
                extracted_data = self._extract_file_data(data, start, sig_info["file_type"])
                
                # If we couldn't extract properly, just take a chunk after the signature
                if extracted_data is None:
                    # Limit extraction to 1MB to avoid memory issues
                    extracted_data = data[start:start+1024*1024]
                
                # Validate the extracted content
                validation = FindingValidator.validate_binary_content(extracted_data)
                
                # Only include validated content or high confidence matches
                if validation["valid"] or sig_info["confidence"] > 0.7:
                    extracted_files.append({
                        "start_position": start,
                        "file_type": sig_info["file_type"],
                        "size": len(extracted_data),
                        "confidence": max(sig_info["confidence"], validation.get("confidence", 0.0) * 0.8),
                        "validation": validation,
                        "data": base64.b64encode(extracted_data[:1024*1024]).decode('ascii')  # Limit to 1MB
                    })
            
            # Update confidence score based on signatures
            if high_confidence_signatures:
                confidence_score = max(confidence_score, max(sig["confidence"] for sig in high_confidence_signatures))
                
                # Add extracted files to results
                results["signatures_found"] = high_confidence_signatures
                
                if extracted_files:
                    # Sort by confidence
                    extracted_files.sort(key=lambda x: x["confidence"], reverse=True)
                    best_file = extracted_files[0]
                    
                    results["extracted_data"] = {
                        "type": "binary",
                        "encoding": "base64",
                        "data": best_file["data"],
                        "file_type": best_file["file_type"],
                        "confidence": best_file["confidence"]
                    }
            
            # Overall results
            results["confidence_score"] = confidence_score
            
            # For JPEGs with after_eoi_data, we already determined the result
            if "after_eoi_data" in results:
                return (True, results)
            
            # Return findings based on confidence score
            if confidence_score > 0.6:
                return (True, results)
            elif found_signatures:
                # Return low confidence signatures if we have them but didn't find high confidence ones
                results["message"] = "Found potential file signatures but with moderate/low confidence"
                return (confidence_score > 0.4, results)
            
            return (False, None)
            
        except Exception as e:
            self.logger.error(f"Error in file signature analysis: {e}")
            return (False, {"error": str(e)})
    
    def _find_last_jpeg_eoi(self, data: bytes) -> int:
        """
        Find the position of the last JPEG EOI (End of Image) marker.
        
        Args:
            data: Binary data to search
            
        Returns:
            Position of the EOI marker, or -1 if not found
        """
        # Start search from near the end and work backwards
        # Many steganography tools append data after the EOI marker
        pos = len(data) - 2
        while pos >= 0:
            if data[pos] == 0xFF and data[pos + 1] == 0xD9:
                return pos
            pos -= 1
        
        return -1
    
    def _identify_appended_data_type(self, data: bytes) -> str:
        """
        Try to identify the type of appended data.
        
        Args:
            data: Binary data to identify
            
        Returns:
            Identified file type or 'unknown'
        """
        # Check for known file signatures
        for signature, file_type in self.FILE_SIGNATURES.items():
            if data.startswith(signature):
                return file_type
        
        # Check if it might be text
        try:
            text = data.decode('utf-8', errors='ignore')
            # Count printable characters
            printable_count = sum(1 for c in text if c.isprintable())
            if printable_count / len(text) > 0.8 and len(text) > 10:
                return 'text'
        except:
            pass
        
        return 'unknown'
    
    def _extract_appended_data(self, data: bytes) -> bytes:
        """
        Extract and process appended data.
        
        Args:
            data: Binary data after the EOI marker
            
        Returns:
            Processed binary data
        """
        # Try to determine if this is a known file type and extract properly
        for signature, file_type in self.FILE_SIGNATURES.items():
            if data.startswith(signature):
                extracted = self._extract_file_data(data, 0, file_type)
                if extracted:
                    return extracted
        
        # If no specific extraction method worked, return as is
        return data
    
    def _get_valid_jpeg_regions(self, data: bytes) -> List[Tuple[int, int]]:
        """
        Identify valid JPEG regions that should be excluded from signature search.
        
        Args:
            data: JPEG binary data
            
        Returns:
            List of (start, end) tuples defining valid JPEG regions
        """
        regions = []
        pos = 0
        
        # Add the JPEG header (0-20 bytes) as a valid region
        regions.append((0, min(20, len(data))))
        
        # Parse JPEG segments
        while pos < len(data) - 1:
            if data[pos] == 0xFF and data[pos + 1] not in [0x00, 0xFF]:
                marker = data[pos + 1]
                
                # Handle different marker types
                if marker == 0xD9:  # EOI - End of Image
                    regions.append((pos, pos + 2))
                    break
                elif marker in [0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7]:
                    # RST markers - no length field
                    regions.append((pos, pos + 2))
                    pos += 2
                else:
                    # Most markers have a length field
                    if pos + 3 < len(data):
                        length = (data[pos + 2] << 8) | data[pos + 3]
                        if length < 2 or pos + length + 2 > len(data):
                            # Invalid length, skip this marker
                            pos += 2
                        else:
                            # Mark this whole segment as valid
                            regions.append((pos, pos + length + 2))
                            pos += length + 2
                    else:
                        # Not enough data for length field
                        pos += 2
            else:
                pos += 1
        
        return regions
    
    def _evaluate_signature_confidence(self, data: bytes, pos: int, signature: bytes, file_type: str) -> float:
        """
        Evaluate the confidence that a signature match is a real embedded file.
        
        Args:
            data: Full file data
            pos: Position of the signature match
            signature: The signature that was matched
            file_type: Type of file corresponding to the signature
            
        Returns:
            Confidence score between 0 (definitely false) and 1 (definitely real)
        """
        # Start with a moderate confidence
        confidence = 0.5
        
        # If the signature is very short, lower confidence
        if len(signature) <= 2:
            confidence -= 0.2
        
        # If the signature is longer, increase confidence
        if len(signature) >= 6:
            confidence += 0.2
        
        # Check if the signature is preceded by null bytes or other padding
        # (common for hiding data)
        if pos > 4:
            preceding = data[pos-4:pos]
            if all(b == 0 for b in preceding) or all(b == 0xFF for b in preceding):
                confidence += 0.1
            
        # For certain file types, verify by checking additional structure
        if file_type == 'zip':
            # Zip files should have a valid structure following the signature
            if pos + 30 < len(data):  # Central directory record is at least 30 bytes
                confidence += 0.15
                
                # Check for local file header structure
                if data[pos+6:pos+8] in [b'\x00\x00', b'\x08\x00', b'\x08\x08']:  # Common compression methods
                    confidence += 0.1
        
        elif file_type == 'png':
            # PNG files should have IHDR chunk after the signature
            if pos + 24 < len(data) and data[pos+12:pos+16] == b'IHDR':
                confidence += 0.3
        
        elif file_type.startswith('jpg'):
            # JPEG files should have valid marker structure
            if pos + 6 < len(data) and data[pos+2] == 0xFF:
                confidence += 0.2
        
        # Check contextual anomalies - signatures in weird places are more suspicious
        # For example, a PNG signature in the middle of a JPEG file is highly suspicious
        if pos > 1000:  # Not at the beginning of the file
            confidence += 0.1
            
        # Cap confidence between 0 and 1
        return max(0.0, min(1.0, confidence))
    
    def _extract_file_data(self, data: bytes, start: int, file_type: str) -> Optional[bytes]:
        """
        Extract embedded file data based on the file type and structure.
        
        Args:
            data: Full file data
            start: Position where the file signature starts
            file_type: Type of file to extract
            
        Returns:
            Extracted file data or None if extraction isn't possible
        """
        try:
            # Extract based on file type
            if file_type == 'zip':
                # Find the end of central directory record
                eocd_pos = data.rfind(b'PK\x05\x06', start)
                if eocd_pos > start and eocd_pos < len(data) - 22:
                    # The last 4 bytes of the EOCD give the size of the central directory
                    cd_size = int.from_bytes(data[eocd_pos+20:eocd_pos+24], byteorder='little')
                    # The 8 bytes before that give the offset of the central directory
                    cd_offset = int.from_bytes(data[eocd_pos+16:eocd_pos+20], byteorder='little')
                    
                    # Calculate the total ZIP file size
                    if cd_offset > 0 and cd_size > 0:
                        zip_size = eocd_pos + 22 - start
                        return data[start:start+zip_size]
            
            elif file_type == 'png':
                # PNG files end with an IEND chunk
                iend_pos = data.find(b'IEND\xae\x42\x60\x82', start)
                if iend_pos > start:
                    # Add 12 bytes for the chunk length, type, data, and CRC
                    return data[start:iend_pos+12]
            
            elif file_type.startswith('jpg'):
                # Find the EOI marker
                pos = start
                while pos < len(data) - 1:
                    if data[pos] == 0xFF and data[pos + 1] == 0xD9:
                        return data[start:pos+2]
                    pos += 1
            
            # For other file types, or if the structured extraction failed,
            # we just return None and let the caller use a fixed-size chunk
            return None
            
        except Exception as e:
            self.logger.debug(f"Error extracting file data: {e}")
            return None


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
            # Check if this strategy is applicable to the image
            strategy_validation = FindingValidator.validate_steganography_approach(self.name, image_path)
            if not strategy_validation["applicable"]:
                self.logger.info(f"BitSequenceStrategy may not be suitable for {image_path}: {strategy_validation['reason']}")
            
            # For JPEGs, warn about potential artifacts
            is_jpeg = FindingValidator.is_jpeg_decompressed(image_path)
            if is_jpeg:
                self.logger.info(f"Image {image_path} is a JPEG. Bit sequence analysis on decompressed data may produce artifacts.")
            
            img = self.load_image(image_path)
            img_array = np.array(img)
            
            results = {}
            found_something = False
            highest_confidence = 0.0
            
            # Extract a sequence of bits from the image
            bit_sequence = self._extract_bit_sequence(img_array)
            binary_string = ''.join(str(bit) for bit in bit_sequence)
            
            # Look for repeating patterns
            patterns = self._find_repeating_patterns(binary_string)
            if patterns:
                results["repeating_patterns"] = patterns
                found_something = True
                
                # Assign confidence based on pattern significance
                pattern_confidence = min(0.5, len(patterns) * 0.1)
                highest_confidence = max(highest_confidence, pattern_confidence)
            
            # Try different bit-to-byte conversions
            # For JPEGs, limit the conversions to reduce false positives
            bits_per_group_options = [8] if is_jpeg else [7, 8]
            
            for bits_per_group in bits_per_group_options:
                # Convert bits to ASCII with MSB first
                msb_text = self._bits_to_text(bit_sequence, bits_per_group, msb_first=True)
                
                # Validate the text
                msb_validation = FindingValidator.validate_text_content(msb_text)
                if msb_validation["valid"]:
                    results[f"msb_{bits_per_group}bit"] = {
                        "readable": True,
                        "text": msb_text[:1000],
                        "validation": msb_validation,
                        "confidence": msb_validation["confidence"]
                    }
                    found_something = True
                    highest_confidence = max(highest_confidence, msb_validation["confidence"])
                
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