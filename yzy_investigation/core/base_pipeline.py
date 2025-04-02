"""
Base pipeline class that all sub-project pipelines should inherit from.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path


class BasePipeline(ABC):
    """Abstract base class for all investigation pipelines."""
    
    def __init__(self, input_path: Path, output_path: Optional[Path] = None) -> None:
        """
        Initialize the pipeline.
        
        Args:
            input_path: Path to input data
            output_path: Optional path for output data. If None, uses results/
        """
        self.input_path = Path(input_path)
        self.output_path = output_path or Path("results")
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        Run the pipeline's main logic.
        
        Returns:
            Dict containing results and metadata
        """
        pass
    
    @abstractmethod
    def validate_input(self) -> bool:
        """
        Validate input data before processing.
        
        Returns:
            bool indicating if input is valid
        """
        pass 