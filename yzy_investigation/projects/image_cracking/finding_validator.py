"""
Utility class for validating findings from steganography analysis.
"""
from pathlib import Path
import re
import os
from typing import Any, Dict, List
import numpy as np


class FindingValidator:
    """
    Utility class to validate findings and compute confidence scores.
    
    This class provides methods for validating findings from various strategies
    and computing confidence scores to help reduce false positives.
    """
    
    # Dictionary of common words for text validation
    COMMON_WORDS = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I', 'it', 'for', 'not', 'on',
        'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we',
        'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
        'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when',
        'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take', 'person', 'into',
        'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now',
        'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two',
        'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any',
        'these', 'give', 'day', 'most', 'us'
    }
    
    # Known investigation keywords
    INVESTIGATION_KEYWORDS = {
        '4NBT', 'silver', 'YZY', 'Blake', 'William', 'tyger', 'albion', 'jerusalem',
        'auguries', 'innocence', 'experience', 'heaven', 'hell', 'symmetry'
    }
    
    # File signatures for various file types
    FILE_SIGNATURES = {
        b'\xFF\xD8\xFF': 'jpg',
        b'\x89PNG\r\n\x1a\n': 'png',
        b'GIF87a': 'gif',
        b'GIF89a': 'gif',
        b'%PDF': 'pdf',
        b'PK\x03\x04': 'zip',
        b'<!DOCTYPE html': 'html',
        b'<html': 'html',
        b'fLaC': 'flac',
        b'ID3': 'mp3',
        b'\xFF\xFB': 'mp3',
        b'OggS': 'ogg',
        b'RIFF': 'wav',
        b'\x00\x00\x01\xba': 'mpeg',
        b'\x00\x00\x01\xb3': 'mpeg'
    }
    
    @staticmethod
    def validate_text_content(text: str) -> Dict[str, Any]:
        """
        Validate if text content appears to be meaningful rather than random.
        
        Args:
            text: Text to validate
            
        Returns:
            Dictionary with validation results and confidence score
        """
        if not text or len(text) < 5:
            return {"valid": False, "confidence": 0.0, "reason": "Text too short"}
        
        # Remove non-printable characters
        cleaned_text = ''.join(c for c in text if c.isprintable())
        
        # Check if mostly printable
        printable_ratio = len(cleaned_text) / len(text)
        if printable_ratio < 0.7:
            return {"valid": False, "confidence": 0.0, "reason": "Too many non-printable characters"}
        
        # Split into words
        words = re.findall(r'\b[a-zA-Z]{2,}\b', cleaned_text.lower())
        
        if not words:
            return {"valid": False, "confidence": 0.0, "reason": "No words found"}
        
        # Calculate what percentage of words are real dictionary words
        common_word_count = sum(1 for word in words if word in FindingValidator.COMMON_WORDS)
        common_word_ratio = common_word_count / len(words) if words else 0
        
        # Check for investigation-specific keywords
        investigation_keyword_count = sum(1 for word in words 
                                      if any(keyword.lower() in word.lower() 
                                            for keyword in FindingValidator.INVESTIGATION_KEYWORDS))
        
        # Calculate base confidence
        confidence = 0.0
        
        # If text contains common words, it's more likely to be real
        if common_word_ratio > 0.3:
            confidence += 0.4
        elif common_word_ratio > 0.1:
            confidence += 0.2
        
        # Longer texts with multiple words are more likely to be real
        if len(words) > 10:
            confidence += 0.2
        elif len(words) > 5:
            confidence += 0.1
        
        # Investigation keywords are strong indicators
        if investigation_keyword_count > 0:
            confidence += min(0.3, investigation_keyword_count * 0.1)
        
        # Check for sentence-like structures (capital letters followed by periods)
        sentence_pattern = re.compile(r'[A-Z][^.!?]*[.!?]')
        sentences = sentence_pattern.findall(cleaned_text)
        if sentences:
            confidence += min(0.2, len(sentences) * 0.05)
        
        # Calculate overall valid status
        valid = confidence > 0.4 or investigation_keyword_count > 0
        
        return {
            "valid": valid,
            "confidence": confidence,
            "common_word_ratio": common_word_ratio,
            "common_word_count": common_word_count,
            "total_words": len(words),
            "investigation_keyword_count": investigation_keyword_count,
            "sentence_count": len(sentences) if sentences else 0,
            "reason": "Valid text content" if valid else "Insufficient evidence of meaningful text"
        }
    
    @staticmethod
    def validate_binary_content(data: bytes) -> Dict[str, Any]:
        """
        Validate if binary content appears to be meaningful data rather than random bytes.
        
        Args:
            data: Binary data to validate
            
        Returns:
            Dictionary with validation results and confidence score
        """
        if not data or len(data) < 8:
            return {"valid": False, "confidence": 0.0, "reason": "Data too short"}
        
        confidence = 0.0
        file_type = None
        
        # Check for file signatures
        for signature, type_name in FindingValidator.FILE_SIGNATURES.items():
            if data.startswith(signature):
                confidence += 0.8
                file_type = type_name
                break
        
        # If no file signature found, analyze byte distribution
        if not file_type:
            # Calculate entropy
            byte_counts = {}
            for b in data[:1000]:  # Sample first 1000 bytes
                byte_counts[b] = byte_counts.get(b, 0) + 1
            
            entropy = 0.0
            for count in byte_counts.values():
                p = count / min(len(data), 1000)
                entropy -= p * np.log2(p) if p > 0 else 0
            
            # Most valid data has entropy between 3.0 and 7.5
            # Random data typically has entropy close to 8.0
            # Very structured data has low entropy
            if 3.0 < entropy < 7.5:
                confidence += 0.3
                
                # Files usually have patterns of repeated bytes
                # Count sequences of repeated bytes
                repeated_sequences = 0
                for b in range(256):
                    seq = bytes([b]) * 3  # Look for 3+ repeated bytes
                    if seq in data:
                        repeated_sequences += 1
                
                if repeated_sequences > 5:
                    confidence += 0.1
            
            # Try to interpret as text
            try:
                text = data.decode('utf-8', errors='ignore')
                text_validation = FindingValidator.validate_text_content(text)
                if text_validation["valid"]:
                    confidence += text_validation["confidence"]
                    if not file_type:
                        file_type = "text"
            except:
                pass
        
        # Final validity determination
        valid = confidence > 0.4
        
        return {
            "valid": valid,
            "confidence": min(1.0, confidence),
            "file_type": file_type,
            "reason": f"Valid {file_type} content" if valid else "Insufficient evidence of meaningful data"
        }
    
    @staticmethod
    def validate_steganography_approach(strategy_name: str, image_path: Path) -> Dict[str, Any]:
        """
        Validate if a steganography approach is reasonable for the given image.
        
        Args:
            strategy_name: Name of the steganography strategy
            image_path: Path to the image file
            
        Returns:
            Dictionary with validation results including relevance score
        """
        # Check file extension
        file_extension = image_path.suffix.lower()
        
        # Get file size
        file_size = os.path.getsize(image_path)
        
        # Strategy compatibility matrix
        compatibility = {
            "lsb_strategy": {
                ".jpg": 0.3,  # LSB is not very reliable for JPEGs
                ".jpeg": 0.3,
                ".png": 0.9,  # LSB works well for lossless formats
                ".bmp": 0.9,
                ".gif": 0.7,
                ".tif": 0.9,
                ".tiff": 0.9
            },
            "color_histogram_strategy": {
                ".jpg": 0.5,
                ".jpeg": 0.5,
                ".png": 0.8,
                ".bmp": 0.8,
                ".gif": 0.6,
                ".tif": 0.8,
                ".tiff": 0.8
            },
            "jpeg_domain_strategy": {
                ".jpg": 0.9,  # Very relevant for JPEGs
                ".jpeg": 0.9,
                ".png": 0.0,  # Not applicable to other formats
                ".bmp": 0.0,
                ".gif": 0.0,
                ".tif": 0.0,
                ".tiff": 0.0
            },
            "keyword_xor_strategy": {
                ".jpg": 0.5,
                ".jpeg": 0.5,
                ".png": 0.7,
                ".bmp": 0.7,
                ".gif": 0.6,
                ".tif": 0.7,
                ".tiff": 0.7
            },
            "shift_cipher_strategy": {
                ".jpg": 0.5,
                ".jpeg": 0.5,
                ".png": 0.7,
                ".bmp": 0.7,
                ".gif": 0.6,
                ".tif": 0.7,
                ".tiff": 0.7
            },
            "bit_sequence_strategy": {
                ".jpg": 0.4,
                ".jpeg": 0.4,
                ".png": 0.8,
                ".bmp": 0.8,
                ".gif": 0.7,
                ".tif": 0.8,
                ".tiff": 0.8
            },
            "blake_hash_strategy": {
                ".jpg": 0.5,
                ".jpeg": 0.5,
                ".png": 0.7,
                ".bmp": 0.7,
                ".gif": 0.6,
                ".tif": 0.7,
                ".tiff": 0.7
            },
            "custom_rgb_encoding_strategy": {
                ".jpg": 0.4,
                ".jpeg": 0.4,
                ".png": 0.9,
                ".bmp": 0.9,
                ".gif": 0.7,
                ".tif": 0.9,
                ".tiff": 0.9
            }
        }
        
        # Get compatibility score for this strategy and file type
        strategy_compatibility = compatibility.get(strategy_name, {})
        relevance = strategy_compatibility.get(file_extension, 0.5)  # Default to medium relevance
        
        # Additional checks for specific strategies
        if strategy_name == "lsb_strategy" and file_extension in [".jpg", ".jpeg"]:
            # For JPEG, LSB in decompressed domain is usually just noise
            relevance = 0.2
            reason = "LSB analysis on JPEG files is unreliable due to lossy compression"
        elif strategy_name == "keyword_xor_strategy" or strategy_name == "shift_cipher_strategy":
            # These strategies work better with smaller files
            if file_size > 5 * 1024 * 1024:  # > 5MB
                relevance *= 0.8
                reason = "Large files may produce coincidental matches with these strategies"
            else:
                reason = "Strategy is applicable to this file type"
        elif strategy_name == "jpeg_domain_strategy":
            if file_extension in [".jpg", ".jpeg"]:
                reason = "JPEG domain analysis is highly relevant for JPEG files"
            else:
                reason = "JPEG domain analysis is not applicable to this file type"
        else:
            reason = f"Strategy has {relevance:.1f} relevance for this file type"
        
        return {
            "relevance": relevance,
            "file_type": file_extension,
            "strategy": strategy_name,
            "reason": reason,
            "applicable": relevance > 0.3
        }
    
    @staticmethod
    def is_jpeg_decompressed(image_path: Path) -> bool:
        """
        Check if an image is a JPEG that has been decompressed.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Boolean indicating if the image is a decompressed JPEG
        """
        return image_path.suffix.lower() in ['.jpg', '.jpeg', '.jpe', '.jfif']
    
    @staticmethod
    def combine_strategy_scores(strategy_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine results from multiple strategies into a single confidence assessment.
        
        Args:
            strategy_results: Dictionary mapping strategy names to their results
            
        Returns:
            Dictionary containing combined assessment including:
                - combined_confidence: Overall confidence score
                - evidence_count: Number of strategies that found evidence
                - conclusion: Text summary of findings
                - strategy_weights: How each strategy contributed
        """
        evidence_count = 0
        total_confidence = 0.0
        strategy_weights = {}
        
        # Base weights for different types of evidence
        WEIGHT_FACTORS = {
            "jpeg_domain_strategy": 1.2,  # Most reliable for JPEGs
            "lsb_strategy": 0.8,  # Good for PNG/BMP but less for JPEG
            "color_histogram_strategy": 0.9,
            "metadata_analysis_strategy": 0.7,
            "keyword_xor_strategy": 0.8,
            "shift_cipher_strategy": 0.7,
            "bit_sequence_strategy": 0.8,
            "blake_hash_strategy": 0.9,
            "file_signature_strategy": 1.0,
            "custom_rgb_encoding_strategy": 0.6  # Less reliable due to potential artifacts
        }
        
        # First pass - count evidence and calculate base confidence
        for strategy_name, result in strategy_results.items():
            if result.get("detected", False):
                evidence_count += 1
                
                # Get confidence from result data
                confidence = 0.0
                result_data = result.get("data", {})
                
                if isinstance(result_data, dict):
                    # Try different confidence fields that strategies might use
                    confidence = result_data.get("confidence", 
                                result_data.get("overall_confidence",
                                result_data.get("combined_confidence", 0.0)))
                    
                    # If no explicit confidence, but we have validation results
                    if confidence == 0.0 and "validation" in result_data:
                        confidence = result_data["validation"].get("confidence", 0.0)
                
                # Apply strategy-specific weight
                weight = WEIGHT_FACTORS.get(strategy_name, 0.7)
                weighted_confidence = confidence * weight
                
                total_confidence += weighted_confidence
                strategy_weights[strategy_name] = weighted_confidence
        
        # Calculate final combined confidence
        if evidence_count > 0:
            # Average confidence across strategies that found something
            combined_confidence = total_confidence / evidence_count
            
            # Boost confidence if multiple strategies agree
            if evidence_count > 1:
                # Add a bonus for corroborating evidence
                confidence_boost = min(0.2, (evidence_count - 1) * 0.1)
                combined_confidence = min(1.0, combined_confidence + confidence_boost)
        else:
            combined_confidence = 0.0
        
        # Generate conclusion text
        if combined_confidence > 0.8:
            conclusion = "High confidence detection of hidden data"
        elif combined_confidence > 0.6:
            conclusion = "Moderate confidence detection of hidden data"
        elif combined_confidence > 0.4:
            conclusion = "Possible hidden data detected but low confidence"
        elif combined_confidence > 0.2:
            conclusion = "Weak indicators of hidden data, likely false positive"
        else:
            conclusion = "No significant evidence of hidden data"
            
        # Add evidence count to conclusion if multiple strategies found something
        if evidence_count > 1:
            conclusion += f" ({evidence_count} strategies found evidence)"
        
        return {
            "combined_confidence": float(combined_confidence),  # Ensure it's a Python float
            "evidence_count": evidence_count,
            "conclusion": conclusion,
            "strategy_weights": strategy_weights
        } 