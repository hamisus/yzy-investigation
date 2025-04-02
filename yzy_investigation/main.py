"""
Main CLI interface for the YzY investigation project.
"""
import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Type

from yzy_investigation.projects.web_scraper import YewsScraper
from yzy_investigation.projects.puzzle_cracking import (
    StegAnalysisPipeline,
    StegStrategy,
    LsbStrategy,
    ColorHistogramStrategy,
    FileSignatureStrategy,
    MetadataAnalysisStrategy
)
# Import other project modules as they are implemented


def setup_logging(verbose: bool = False) -> None:
    """
    Set up logging configuration.
    
    Args:
        verbose: Whether to enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create logs directory
    logs_dir = Path("results/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = logs_dir / f"yzy_investigation_{timestamp}.log"
    
    # Clear any existing handlers to avoid duplicate logs
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # Configure the root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Output to console
            logging.FileHandler(log_file)  # Output to file
        ]
    )
    
    # Explicitly set the level to make sure it takes effect
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Set level for common noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.WARNING)
    
    logging.info(f"Logging to file: {log_file}")


def scrape_yews(output_dir: Optional[Path] = None, verbose: bool = False) -> None:
    """
    Scrape images and content from YEWS.news.
    
    Args:
        output_dir: Optional custom output directory
        verbose: Whether to enable verbose logging
    """
    setup_logging(verbose)
    logging.info("Starting YEWS scraper...")
    
    scraper = YewsScraper(output_path=output_dir)
    results = scraper.run()
    
    logging.info(f"Scraping completed. Results: {results}")


def analyze_stego(
    input_dir: Path,
    output_dir: Optional[Path] = None, 
    strategies: Optional[List[str]] = None,
    verbose: bool = False
) -> None:
    """
    Analyze images for steganography.
    
    Args:
        input_dir: Directory containing images to analyze
        output_dir: Optional output directory for results
        strategies: Optional list of strategy names to use
        verbose: Whether to enable verbose logging
    """
    setup_logging(verbose)
    logging.info("Starting steganography analysis...")
    
    # Map strategy names to classes
    strategy_map = {
        "lsb": LsbStrategy,
        "color_histogram": ColorHistogramStrategy,
        "file_signature": FileSignatureStrategy,
        "metadata": MetadataAnalysisStrategy
    }
    
    # Select strategies to use
    selected_strategies: List[Type[StegStrategy]] = []
    if strategies:
        for strategy_name in strategies:
            if strategy_name in strategy_map:
                selected_strategies.append(strategy_map[strategy_name])
            else:
                logging.warning(f"Unknown strategy: {strategy_name}")
    else:
        # Use all strategies by default
        selected_strategies = list(strategy_map.values())
    
    # Create and run the pipeline
    pipeline = StegAnalysisPipeline(
        input_path=input_dir,
        output_path=output_dir,
        strategies=selected_strategies
    )
    results = pipeline.run()
    
    if results["success"]:
        logging.info(f"Analysis completed. Summary: {results['summary']}")
        logging.info(f"Results saved to: {results['results_directory']}")
    else:
        logging.error(f"Analysis failed: {results.get('error', 'Unknown error')}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="YzY Investigation Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape YEWS.news:
  python -m yzy_investigation.main scrape-yews
  
  # Scrape with verbose logging:
  python -m yzy_investigation.main scrape-yews -v
  
  # Scrape to custom directory:
  python -m yzy_investigation.main scrape-yews --output-dir ./custom_output
  
  # Analyze images for steganography:
  python -m yzy_investigation.main stego-analyze --input-dir ./data/raw/yews
  
  # Use specific stego analysis strategies:
  python -m yzy_investigation.main stego-analyze --input-dir ./data/raw/yews --strategies lsb file_signature
"""
    )
    
    # Global options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # YEWS scraper command
    yews_parser = subparsers.add_parser(
        "scrape-yews",
        help="Scrape images and content from YEWS.news"
    )
    yews_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Custom output directory for scraped content"
    )
    
    # Stego analysis command
    stego_parser = subparsers.add_parser(
        "stego-analyze",
        help="Analyze images for steganography"
    )
    stego_parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing images to analyze"
    )
    stego_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Custom output directory for analysis results"
    )
    stego_parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["lsb", "color_histogram", "file_signature", "metadata"],
        help="Specific strategies to use (default: all)"
    )
    
    # Add more subcommands as we implement them
    
    args = parser.parse_args()
    
    if args.command == "scrape-yews":
        scrape_yews(args.output_dir, args.verbose)
    elif args.command == "stego-analyze":
        analyze_stego(args.input_dir, args.output_dir, args.strategies, args.verbose)
    elif args.command is None:
        parser.print_help()
    else:
        logging.error(f"Unknown command: {args.command}")
        parser.print_help()


if __name__ == "__main__":
    main() 