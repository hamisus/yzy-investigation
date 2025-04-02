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
logger = logging.getLogger("yzy_investigation.puzzle_cracking.stego_analysis")


def make_json_serializable(data: Any) -> Any:
    """
    Recursively convert data to JSON serializable format.
    
    Args:
        data: Any data structure to convert
        
    Returns:
        JSON serializable version of the data
    """
    if isinstance(data, dict):
        return {k: make_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [make_json_serializable(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.int32, np.int64, np.uint32, np.uint64)):
        return int(data)
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (bytes, bytearray)):
        return {
            "type": "binary",
            "encoding": "base64",
            "data": base64.b64encode(data).decode('ascii')
        }
    elif hasattr(data, '__class__') and data.__class__.__name__ == 'IFDRational':
        return float(data)
    elif hasattr(data, 'isoformat'):  # Handle date/datetime
        return data.isoformat()
    else:
        return data


class StegStrategy(ABC):
    """Base class for all steganography analysis strategies."""
    
    name: str = "base_strategy"
    description: str = "Base steganography analysis strategy"
    
    def __init__(self) -> None:
        """Initialize the strategy."""
        self.logger = logging.getLogger(f"yzy_investigation.puzzle_cracking.stego_analysis.{self.name}")
    
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
            "detected": detected,
            "data": data or {}
        }
    
    def _make_json_serializable(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a dictionary to be JSON serializable by handling binary data.
        
        Args:
            data: Dictionary to convert
            
        Returns:
            JSON serializable dictionary
        """
        return make_json_serializable(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary for serialization.
        
        Returns:
            Dictionary representation of the result
        """
        # Ensure all data is JSON serializable, including nested data
        return make_json_serializable({
            "image_path": str(self.image_path),
            "image_name": self.image_name,
            "has_hidden_data": self.has_hidden_data,
            "strategy_results": self.strategy_results
        })


class StegAnalysisPipeline(BasePipeline):
    """Pipeline for analyzing images for steganography."""
    
    def __init__(
        self, 
        input_path: Path, 
        output_path: Optional[Path] = None,
        strategies: Optional[List[Type[StegStrategy]]] = None
    ) -> None:
        """
        Initialize the steganography analysis pipeline.
        
        Args:
            input_path: Path to directory containing images to analyze
            output_path: Optional path for output data. If None, uses results/stego_analysis/
            strategies: Optional list of StegStrategy classes to use
        """
        super().__init__(
            input_path, 
            output_path or Path("results/stego_analysis")
        )
        
        self.strategies = strategies or []
        self.results: List[StegAnalysisResult] = []
        self.data_manager = DataManager()
        self.logger = logger
    
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
        
        for strategy_class in self.strategies:
            strategy = strategy_class()
            self.logger.info(f"Applying strategy {strategy.name} to {image_path.name}")
            
            try:
                detected, data = strategy.analyze(image_path)
                result.add_strategy_result(strategy.name, detected, data)
                
                if detected:
                    self.logger.info(f"Strategy {strategy.name} detected hidden data in {image_path.name}")
                    # Save extracted data if available
                    if data and "extracted_data" in data:
                        self._save_extracted_data(image_path, strategy.name, data)
            except Exception as e:
                self.logger.error(f"Error applying strategy {strategy.name} to {image_path.name}: {e}")
                result.add_strategy_result(strategy.name, False, {"error": str(e)})
        
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
        output_dir = self.output_path / "extracted_data" / image_path.stem
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
        
        image_files = self.find_images()
        self.logger.info(f"Found {len(image_files)} images to analyze")
        
        for image_path in image_files:
            result = self.analyze_image(image_path)
            self.results.append(result)
            
            # Save individual result
            result_path = self.output_path / "results" / f"{image_path.stem}_analysis.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                with open(result_path, "w") as f:
                    json.dump(result.to_dict(), f, indent=2)
            except TypeError as e:
                self.logger.error(f"Error serializing result for {image_path.name}: {e}")
                # Try a fallback method with more aggressive serialization
                with open(result_path, "w") as f:
                    simplified_result = {
                        "image_path": str(image_path),
                        "image_name": image_path.name,
                        "has_hidden_data": result.has_hidden_data,
                        "error": "Some data could not be serialized to JSON"
                    }
                    json.dump(simplified_result, f, indent=2)
        
        # Compile summary
        summary = {
            "total_images": len(self.results),
            "images_with_hidden_data": sum(1 for r in self.results if r.has_hidden_data),
            "strategy_success_counts": {
                strategy_class.name: sum(
                    1 for r in self.results 
                    if strategy_class.name in r.strategy_results and r.strategy_results[strategy_class.name]["detected"]
                )
                for strategy_class in self.strategies
            }
        }
        
        # Save summary
        summary_path = self.output_path / "analysis_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        return {
            "success": True,
            "summary": summary,
            "results_directory": str(self.output_path)
        } 