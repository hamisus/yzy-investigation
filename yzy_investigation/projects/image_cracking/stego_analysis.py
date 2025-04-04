"""
Steganography analysis module for detecting hidden data in images.
"""
from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import os
import json
import base64
import time

from PIL import Image
import numpy as np
try:
    from PIL.TiffImagePlugin import IFDRational
except ImportError:
    # Define a fallback class if IFDRational is not available
    class IFDRational:
        pass

from yzy_investigation.core.base_pipeline import BasePipeline
from yzy_investigation.core.data_manager import DataManager


# Create a named logger
logger = logging.getLogger("yzy_investigation.image_cracking.stego_analysis")


def make_json_serializable(data: Any) -> Any:
    """
    Recursively convert data to JSON serializable format.
    
    Args:
        data: Any data structure to convert
        
    Returns:
        JSON serializable version of the data
    """
    if data is None:
        return None
        
    # Handle numpy types first
    if hasattr(data, 'dtype'):
        # Handle numpy boolean types explicitly
        if str(data.dtype).startswith('bool'):  # More robust check for boolean types
            return bool(data)
        # Handle other numpy types
        elif np.issubdtype(data.dtype, np.integer):
            return int(data)
        elif np.issubdtype(data.dtype, np.floating):
            return float(data)
        elif isinstance(data, np.ndarray):
            return [make_json_serializable(x) for x in data.tolist()]
    
    # Handle standard Python containers
    if isinstance(data, dict):
        return {str(k): make_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [make_json_serializable(item) for item in data]
    elif isinstance(data, set):
        return [make_json_serializable(item) for item in sorted(data)]
    elif isinstance(data, (bytes, bytearray)):
        return {
            "type": "binary",
            "encoding": "base64",
            "data": base64.b64encode(data).decode('ascii')
        }
    elif isinstance(data, bool):  # Handle Python bool
        return bool(data)
    elif isinstance(data, (int, float)):
        return data
    elif hasattr(data, '__class__') and data.__class__.__name__ == 'IFDRational':
        return float(data)
    elif hasattr(data, 'isoformat'):  # Handle date/datetime
        return data.isoformat()
    elif hasattr(data, 'to_dict'):  # Handle objects with to_dict method
        return make_json_serializable(data.to_dict())
    else:
        try:
            # Try to convert to string if all else fails
            return str(data)
        except Exception:
            return None


class StegStrategy(ABC):
    """Base class for all steganography analysis strategies."""
    
    name: str = "base_strategy"
    description: str = "Base steganography analysis strategy"
    
    def __init__(self) -> None:
        """Initialize the strategy."""
        self.logger = logging.getLogger(f"yzy_investigation.image_cracking.stego_analysis.{self.name}")
        self._output_dir = None  # This will be set by the pipeline
    
    def set_output_dir(self, output_dir: Path) -> None:
        """
        Set the output directory for this strategy.
        
        Args:
            output_dir: Directory where strategy outputs should be saved
        """
        self._output_dir = output_dir
    
    def get_strategy_output_dir(self, image_path: Path) -> Path:
        """
        Get the output directory for a specific image analysis.
        
        Args:
            image_path: Path to the image being analyzed
            
        Returns:
            Path to output directory for this strategy and image
        """
        if not self._output_dir:
            # Default to current directory if output dir not set
            return Path(".")
            
        # Create unique directory name using parent directory + image name
        # to prevent collisions when multiple images have the same filename
        parent_dir = image_path.parent.name
        unique_image_id = f"{parent_dir}_{image_path.stem}" if parent_dir else image_path.stem
        
        # Create directory structure: output_dir/extracted_data/image_id/
        output_dir = self._output_dir / "extracted_data" / unique_image_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir
    
    @abstractmethod
    def analyze(self, image_path: Path) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Analyze an image for hidden data using this strategy.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple containing:
                - Boolean indicating if hidden data was detected
                - Optional dictionary with extracted data and analysis metadata
        """
        pass
    
    def load_image(self, image_path: Path) -> Image.Image:
        """
        Load an image file using PIL.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object
        """
        try:
            return Image.open(image_path)
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            raise


class StegAnalysisResult:
    """Class to store and manage steganography analysis results."""
    
    def __init__(self, image_path: Path) -> None:
        """
        Initialize the result container.
        
        Args:
            image_path: Path to the analyzed image
        """
        self.image_path = image_path
        self.image_name = image_path.name
        self.strategy_results: Dict[str, Dict[str, Any]] = {}
        self.has_hidden_data: bool = False
        self.combined_assessment: Optional[Dict[str, Any]] = None
        self.potential_false_positive: bool = False
    
    def add_strategy_result(self, strategy_name: str, detected: bool, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a strategy's analysis result.
        
        Args:
            strategy_name: Name of the strategy
            detected: Whether hidden data was detected
            data: Optional dictionary with extracted data and analysis metadata
        """
        if detected:
            self.has_hidden_data = True
            
        # Ensure data is JSON serializable
        if data:
            data = self._make_json_serializable(data)
            
        self.strategy_results[strategy_name] = {
            "detected": bool(detected),  # Explicitly convert to Python bool
            "data": data or {}
        }
    
    def _make_json_serializable(self, data: Any) -> Any:
        """
        Convert data to be JSON serializable.
        
        Args:
            data: Data to convert
            
        Returns:
            JSON serializable version of the data
        """
        return make_json_serializable(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary for serialization.
        
        Returns:
            Dictionary representation of the result
        """
        # Create base dictionary with explicit bool conversions
        result_dict = {
            "image_path": str(self.image_path),
            "image_name": self.image_name,
            "has_hidden_data": bool(self.has_hidden_data),
            "strategy_results": self._make_json_serializable(self.strategy_results)
        }
        
        # Add combined assessment if available
        if self.combined_assessment is not None:
            result_dict["combined_assessment"] = self._make_json_serializable(self.combined_assessment)
            
        # Add potential false positive flag if true
        if self.potential_false_positive:
            result_dict["potential_false_positive"] = bool(self.potential_false_positive)
            
        # Final pass to ensure everything is serializable
        return self._make_json_serializable(result_dict)


class StegAnalysisPipeline(BasePipeline):
    """Pipeline for analyzing images for steganography."""
    
    def __init__(
        self, 
        input_path: Path, 
        output_path: Optional[Path] = None,
        strategies: Optional[List[Type[StegStrategy]]] = None,
        timestamp: Optional[str] = None
    ) -> None:
        """
        Initialize the steganography analysis pipeline.
        
        Args:
            input_path: Path to directory containing images to analyze
            output_path: Optional path for output data base directory. If None, uses results/
            strategies: Optional list of StegStrategy classes to use
            timestamp: Optional timestamp for directory naming (uses current time if None)
        """
        # Set up the base output directory
        base_output_dir = output_path or Path("results")
        base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped directory
        if timestamp is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        self.timestamp = timestamp
        run_dir = base_output_dir / f"run_{timestamp}"
        
        # Call parent constructor with timestamped directory
        super().__init__(input_path, run_dir)
        
        self.strategies = strategies or []
        self.results: List[StegAnalysisResult] = []
        self.data_manager = DataManager()
        self.logger = logger
        
        # Set up log directory
        log_dir = self.output_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a summary file in the base output directory
        with open(base_output_dir / f"latest_run_summary_{timestamp}.txt", "w") as f:
            f.write(f"Latest analysis run: {timestamp}\n\n")
            f.write(f"Input directory: {input_path}\n")
            f.write(f"Full results in: {self.output_path}\n")
    
    def add_strategy(self, strategy_class: Type[StegStrategy]) -> None:
        """
        Add a strategy to the pipeline.
        
        Args:
            strategy_class: StegStrategy class to add
        """
        self.strategies.append(strategy_class)
        self.logger.info(f"Added strategy: {strategy_class.name}")
    
    def validate_input(self) -> bool:
        """
        Validate input directory exists and contains images.
        
        Returns:
            Boolean indicating if input is valid
        """
        if not self.input_path.exists():
            self.logger.error(f"Input path does not exist: {self.input_path}")
            return False
        
        if not self.input_path.is_dir():
            self.logger.error(f"Input path is not a directory: {self.input_path}")
            return False
        
        # Check if there are any image files
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
        has_images = any(
            f.suffix.lower() in image_extensions 
            for f in self.input_path.glob("**/*") 
            if f.is_file()
        )
        
        if not has_images:
            self.logger.error(f"No image files found in {self.input_path}")
            return False
            
        return True
    
    def find_images(self) -> List[Path]:
        """
        Find all image files in the input directory.
        
        Returns:
            List of paths to image files
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
        return [
            f for f in self.input_path.glob("**/*") 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
    
    def analyze_image(self, image_path: Path) -> StegAnalysisResult:
        """
        Analyze a single image with all strategies.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            StegAnalysisResult object with analysis results
        """
        self.logger.info(f"Analyzing image: {image_path.name}")
        result = StegAnalysisResult(image_path)
        strategy_results = {}
        
        # For JPEG images, determine appropriate strategies
        is_jpeg = image_path.suffix.lower() in ['.jpg', '.jpeg', '.jpe', '.jfif']
        
        for strategy_class in self.strategies:
            strategy = strategy_class()
            
            # Skip CustomRgbEncodingStrategy for JPEG files - it's inefficient and ineffective
            if is_jpeg and strategy.name == "custom_rgb_encoding_strategy":
                self.logger.info(f"Skipping {strategy.name} for JPEG image {image_path.name} (not applicable)")
                result.add_strategy_result(strategy.name, False, {
                    "skipped": True, 
                    "reason": "Strategy not applicable to JPEG images"
                })
                continue
                
            self.logger.info(f"Applying strategy {strategy.name} to {image_path.name}")
            
            try:
                strategy.set_output_dir(self.output_path)
                detected, data = strategy.analyze(image_path)
                
                # Store results for multi-factor scoring
                strategy_results[strategy.name] = {
                    "detected": detected,
                    "data": data or {}
                }
                
                result.add_strategy_result(strategy.name, detected, data)
                
                if detected:
                    self.logger.info(f"Strategy {strategy.name} detected hidden data in {image_path.name}")
                    # Save extracted data if available
                    if data and "extracted_data" in data:
                        self._save_extracted_data(image_path, strategy.name, data)
            except Exception as e:
                self.logger.error(f"Error applying strategy {strategy.name} to {image_path.name}: {e}")
                result.add_strategy_result(strategy.name, False, {"error": str(e)})
        
        # Apply multi-factor scoring if we have more than one strategy result
        if len(strategy_results) > 1:
            try:
                from yzy_investigation.projects.image_cracking.stego_strategies import FindingValidator
                combined_scores = FindingValidator.combine_strategy_scores(strategy_results)
                
                # Add the combined score to the result
                result.combined_assessment = combined_scores
                
                # Override the overall detection flag based on the combined confidence
                if combined_scores["combined_confidence"] > 0.6:
                    result.has_hidden_data = True
                elif combined_scores["combined_confidence"] < 0.4 and result.has_hidden_data:
                    # If individual strategies triggered but combined confidence is low,
                    # mark as "potential false positive"
                    result.potential_false_positive = True
                    
                self.logger.info(f"Combined confidence for {image_path.name}: {combined_scores['combined_confidence']:.2f} - {combined_scores['conclusion']}")
            except Exception as e:
                self.logger.error(f"Error applying multi-factor scoring to {image_path.name}: {e}")
        
        return result
    
    def _save_extracted_data(self, image_path: Path, strategy_name: str, data: Dict[str, Any]) -> None:
        """
        Save extracted data to a file.
        
        Args:
            image_path: Path to the source image
            strategy_name: Name of the strategy that extracted the data
            data: Dictionary containing extracted data
        """
        if "extracted_data" not in data:
            return
            
        extracted_data = data["extracted_data"]
        
        # Create unique directory name using parent directory + image name
        # This prevents collisions when multiple images have the same filename (e.g., image_01.jpg)
        parent_dir = image_path.parent.name
        unique_image_id = f"{parent_dir}_{image_path.stem}" if parent_dir else image_path.stem
        
        output_dir = self.output_path / "extracted_data" / unique_image_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle different types of extracted data
        if isinstance(extracted_data, str):
            # Text data
            output_file = output_dir / f"{strategy_name}_text.txt"
            with open(output_file, "w") as f:
                f.write(extracted_data)
            self.logger.info(f"Saved extracted text to {output_file}")
            
            # Update data dict with text for JSON serialization
            data["extracted_data"] = extracted_data
            
        elif isinstance(extracted_data, dict):
            # JSON-serializable data
            output_file = output_dir / f"{strategy_name}_data.json"
            with open(output_file, "w") as f:
                json.dump(extracted_data, f, indent=2)
            self.logger.info(f"Saved extracted data to {output_file}")
            
        elif isinstance(extracted_data, (bytes, bytearray)):
            # Binary data - save to file and store base64 in JSON
            output_file = output_dir / f"{strategy_name}_data.bin"
            with open(output_file, "wb") as f:
                f.write(extracted_data)
            self.logger.info(f"Saved extracted binary data to {output_file}")
            
            # Convert to base64 for JSON serialization
            data["extracted_data"] = {
                "type": "binary",
                "encoding": "base64",
                "data": base64.b64encode(extracted_data).decode('ascii')
            }
            
        elif isinstance(extracted_data, np.ndarray):
            # Image data
            try:
                img = Image.fromarray(extracted_data)
                output_file = output_dir / f"{strategy_name}_image.png"
                img.save(output_file)
                self.logger.info(f"Saved extracted image to {output_file}")
                
                # Store image dimensions for JSON serialization
                data["extracted_data"] = {
                    "type": "image",
                    "dimensions": extracted_data.shape,
                    "saved_to": str(output_file)
                }
            except Exception as e:
                self.logger.error(f"Failed to save extracted image: {e}")
                data["extracted_data"] = {
                    "type": "image",
                    "error": str(e)
                }
        
    def run(self) -> Dict[str, Any]:
        """
        Run the steganography analysis pipeline on all images.
        
        Returns:
            Dictionary with analysis results and metadata
        """
        if not self.validate_input():
            return {"success": False, "error": "Input validation failed"}
        
        if not self.strategies:
            self.logger.warning("No strategies defined. Analysis will not produce results.")
            return {"success": False, "error": "No strategies defined"}
        
        # Record start time for tracking progress
        start_time = time.time()
        
        # Create necessary directories
        (self.output_path / "results").mkdir(parents=True, exist_ok=True)
        (self.output_path / "extracted_data").mkdir(parents=True, exist_ok=True)
        (self.output_path / "logs").mkdir(parents=True, exist_ok=True)
        
        image_files = self.find_images()
        self.logger.info(f"Found {len(image_files)} images to analyze")
        
        for i, image_path in enumerate(image_files):
            # Process the image with all strategies
            result = self.analyze_image(image_path)
            self.results.append(result)
            
            # Create unique identifier using parent directory + image name
            parent_dir = image_path.parent.name
            unique_image_id = f"{parent_dir}_{image_path.stem}" if parent_dir else image_path.stem
            
            # Save individual result
            result_path = self.output_path / "results" / f"{unique_image_id}_analysis.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                # First attempt to serialize the result
                result_dict = result.to_dict()
                # Verify the result can be serialized before writing
                json.dumps(result_dict)  # This will raise TypeError if not serializable
                
                with open(result_path, "w") as f:
                    json.dump(result_dict, f, indent=2)
                    
            except (TypeError, ValueError, json.JSONDecodeError) as e:
                self.logger.error(f"Error serializing result for {image_path.name}: {e}")
                # Create a simplified fallback result
                fallback_result = {
                    "image_path": str(image_path),
                    "image_name": image_path.name,
                    "has_hidden_data": bool(result.has_hidden_data),
                    "error": f"Data serialization error: {str(e)}",
                    "strategy_results": {}
                }
                
                # Try to salvage any strategy results that can be serialized
                for strategy_name, strategy_result in result.strategy_results.items():
                    try:
                        # Test if this individual strategy result can be serialized
                        test_json = json.dumps(make_json_serializable(strategy_result))
                        fallback_result["strategy_results"][strategy_name] = json.loads(test_json)
                    except Exception:
                        fallback_result["strategy_results"][strategy_name] = {
                            "error": "Could not serialize strategy result"
                        }
                
                with open(result_path, "w") as f:
                    json.dump(fallback_result, f, indent=2)
                
            # Calculate and display progress
            progress = (i + 1) / len(image_files) * 100
            elapsed_time = time.time() - start_time
            images_per_second = (i + 1) / elapsed_time if elapsed_time > 0 else 0
            
            # Estimate remaining time
            if i > 0 and images_per_second > 0:
                remaining_images = len(image_files) - (i + 1)
                eta_seconds = remaining_images / images_per_second
                eta_min = int(eta_seconds // 60)
                eta_sec = int(eta_seconds % 60)
                self.logger.debug(f"Progress: {progress:.1f}% - ETA: {eta_min}m {eta_sec}s")
        
        # Get high-confidence images
        high_confidence_images = [r for r in self.results if 
                                r.combined_assessment and 
                                r.combined_assessment.get("combined_confidence", 0) > 0.7]
        
        # Get potential false positive images
        potential_false_positives = [r for r in self.results if r.potential_false_positive]
        
        # Compile summary
        summary = {
            "total_images": len(self.results),
            "images_with_hidden_data": sum(1 for r in self.results if r.has_hidden_data),
            "high_confidence_detections": len(high_confidence_images),
            "potential_false_positives": len(potential_false_positives),
            "strategy_success_counts": {
                strategy_class.name: sum(
                    1 for r in self.results 
                    if strategy_class.name in r.strategy_results and r.strategy_results[strategy_class.name]["detected"]
                )
                for strategy_class in self.strategies
            },
            "timestamp": self.timestamp,
            "elapsed_time": time.time() - start_time,
            "run_dir": str(self.output_path)
        }
        
        # Save summary
        summary_path = self.output_path / "analysis_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
            
        # Save high confidence detections to a separate file
        if high_confidence_images:
            high_conf_path = self.output_path / "high_confidence_detections.json"
            with open(high_conf_path, "w") as f:
                json.dump({
                    "count": len(high_confidence_images),
                    "images": [
                        {
                            "image_name": r.image_name,
                            "image_path": str(r.image_path),
                            "confidence": r.combined_assessment.get("combined_confidence", 0),
                            "conclusion": r.combined_assessment.get("conclusion", ""),
                            "evidence_count": r.combined_assessment.get("evidence_count", 0)
                        }
                        for r in high_confidence_images
                    ]
                }, f, indent=2)
            
        # Update the summary in the base directory
        base_dir = self.output_path.parent
        with open(base_dir / f"latest_run_summary_{self.timestamp}.txt", "a") as f:
            f.write(f"\nAnalysis completed in {summary['elapsed_time']:.2f} seconds\n")
            f.write(f"Total images: {summary['total_images']}\n")
            f.write(f"Images with hidden data: {summary['images_with_hidden_data']}\n")
            f.write(f"High confidence detections: {summary['high_confidence_detections']}\n")
            f.write(f"Potential false positives: {summary['potential_false_positives']}\n")
            strategy_counts = "\n".join([f"  - {name}: {count}" for name, count in summary['strategy_success_counts'].items()])
            f.write(f"Strategy success counts:\n{strategy_counts}\n")
            
            # Add information about high confidence detections
            if high_confidence_images:
                f.write("\nHigh confidence detections:\n")
                for r in high_confidence_images:
                    f.write(f"  - {r.image_name}: {r.combined_assessment.get('combined_confidence', 0):.2f} - {r.combined_assessment.get('conclusion', '')}\n")
        
        self.logger.info(f"Analysis completed. Summary: {summary}")
        self.logger.info(f"Results saved to: {self.output_path}")
        
        return {
            "success": True,
            "summary": summary,
            "results_directory": str(self.output_path)
        } 