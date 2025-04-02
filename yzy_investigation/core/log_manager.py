"""
Centralized logging utility for the YzY investigation project.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class LogManager:
    """Manages logging across all sub-projects."""
    
    def __init__(self, log_dir: Optional[Path] = None) -> None:
        """
        Initialize the log manager.
        
        Args:
            log_dir: Optional directory for log files. If None, uses results/logs/
        """
        self.log_dir = log_dir or Path("results/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a date-specific subdirectory for structured logs
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.structured_logs_dir = self.log_dir / "structured" / self.today
        self.structured_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up a logger for this class
        self.logger = logging.getLogger("yzy_investigation.log_manager")
        
    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Log an event with associated data to a structured JSON file.
        
        Args:
            event_type: Type of event (e.g., "puzzle_solved", "scrape_completed")
            data: Dictionary containing event data
        """
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "event_type": event_type,
            "data": data
        }
        
        # Write to JSON log file in a structured format
        log_file = self.structured_logs_dir / f"{event_type}_{datetime.now().strftime('%H%M%S')}.json"
        with open(log_file, "a") as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)
            f.write("\n")
            
        # Also log to the Python logger
        self.logger.info(f"Event logged: {event_type} [{log_file}]")
        
    def get_log_file_path(self, name: str, extension: str = "json") -> Path:
        """
        Get a path for a log file with a timestamp.
        
        Args:
            name: Base name for the log file
            extension: File extension (default: json)
            
        Returns:
            Path object for the log file
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        return self.structured_logs_dir / f"{name}_{timestamp}.{extension}" 