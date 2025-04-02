"""
Data management utilities for the YzY investigation project.
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union


class DataManager:
    """Manages data operations across all sub-projects."""
    
    def __init__(self, data_dir: Optional[Path] = None) -> None:
        """
        Initialize the data manager.
        
        Args:
            data_dir: Optional root data directory. If None, uses data/
        """
        self.data_dir = data_dir or Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create necessary directories
        for directory in [self.raw_dir, self.processed_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
    def save_json(self, data: Dict[str, Any], filename: str, subdir: str = "processed") -> None:
        """
        Save data as JSON.
        
        Args:
            data: Dictionary to save
            filename: Name of the file (with or without .json extension)
            subdir: Subdirectory within data/ to save to
        """
        if not filename.endswith(".json"):
            filename += ".json"
            
        save_dir = self.data_dir / subdir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(save_dir / filename, "w") as f:
            json.dump(data, f, indent=2)
            
    def load_json(self, filename: str, subdir: str = "processed") -> Dict[str, Any]:
        """
        Load data from JSON file.
        
        Args:
            filename: Name of the file (with or without .json extension)
            subdir: Subdirectory within data/ to load from
            
        Returns:
            Dictionary containing the loaded data
        """
        if not filename.endswith(".json"):
            filename += ".json"
            
        load_path = self.data_dir / subdir / filename
        with open(load_path, "r") as f:
            return json.load(f) 