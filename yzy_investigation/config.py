"""
Global configuration settings for the YzY investigation project.
"""
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Sub-project specific settings
IMAGE_CRACKING = {
    "supported_formats": [".txt", ".png", ".jpg", ".jpeg"],
    "max_file_size": 10 * 1024 * 1024,  # 10MB
}

WEB_SCRAPER = {
    "user_agent": "YzY Investigation Bot/1.0",
    "request_delay": 2,  # seconds between requests
    "max_retries": 3,
}

# Logging settings
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO" 