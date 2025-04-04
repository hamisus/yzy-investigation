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
from yzy_investigation.projects.image_cracking import (
    StegAnalysisPipeline,
    StegStrategy,
    LsbStrategy,
    ColorHistogramStrategy,
    FileSignatureStrategy,
    MetadataAnalysisStrategy,
    KeywordXorStrategy,
    ShiftCipherStrategy,
    BitSequenceStrategy,
    BlakeHashStrategy,
    CustomRgbEncodingStrategy
)
from yzy_investigation.projects.image_cracking.scripts.image_crack_cli import ImageCrackingPipeline
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
        "metadata": MetadataAnalysisStrategy,
        "keyword_xor": KeywordXorStrategy,
        "shift_cipher": ShiftCipherStrategy,
        "bit_sequence": BitSequenceStrategy,
        "blake_hash": BlakeHashStrategy
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
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create and run the pipeline with timestamped output
    pipeline = StegAnalysisPipeline(
        input_path=input_dir,
        output_path=output_dir,  # This is now treated as the base directory
        strategies=selected_strategies,
        timestamp=timestamp
    )
    results = pipeline.run()
    
    if results["success"]:
        logging.info(f"Analysis completed. Summary: {results['summary']}")
        logging.info(f"Results saved to: {results['results_directory']}")
    else:
        logging.error(f"Analysis failed: {results.get('error', 'Unknown error')}")


def image_crack(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    keywords: Optional[List[str]] = None,
    key_numbers: Optional[List[int]] = None,
    config_file: Optional[Path] = None,
    verbose: bool = False
) -> None:
    """
    Run the end-to-end image cracking pipeline.
    
    Args:
        input_dir: Directory containing images to analyze
        output_dir: Optional output directory for results
        keywords: Optional list of additional keywords to use in analysis
        key_numbers: Optional list of additional key numbers to use in analysis
        config_file: Optional path to a configuration file with keywords and key numbers
        verbose: Whether to enable verbose logging
    """
    setup_logging(verbose)
    logging.info("Starting end-to-end image cracking pipeline...")
    
    pipeline = ImageCrackingPipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        keywords=keywords,
        key_numbers=key_numbers,
        config_file=config_file,
        verbose=verbose
    )
    
    results = pipeline.run()
    
    # Print summary
    print("\nIMAGE CRACKING SUMMARY:")
    print("========================")
    print(f"Images analyzed: {results['images_analyzed']}")
    print(f"Images with hidden data: {results['hidden_data_detected']}")
    
    if results['target_string_found']:
        print("\n!!! TARGET STRING FOUND !!!")
        print("See TARGET_FOUND.txt for details")
        
    print(f"\nHigh significance findings: {results['high_significance_findings']}")
    print(f"Medium significance findings: {results['medium_significance_findings']}")
    print(f"Files discovered: {results['files_discovered']}")
    
    print(f"\nFull results saved to: {output_dir or 'results/image_cracking'}")
    
    logging.info("Image cracking completed.")


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
  
  # Run end-to-end image cracking:
  python -m yzy_investigation.main image-crack --input-dir ./data/raw/yews
  
  # Run image cracking with custom keywords and key numbers:
  python -m yzy_investigation.main image-crack --input-dir ./data/raw/yews --keywords silver YZY --key-numbers 4 333 353
  
  # Run image cracking with a keywords configuration file:
  python -m yzy_investigation.main image-crack --input-dir ./data/raw/yews --config ./yzy_investigation/projects/image_cracking/config/keywords.json
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
        choices=["lsb", "color_histogram", "file_signature", "metadata", 
                 "keyword_xor", "shift_cipher", "bit_sequence", "blake_hash"],
        help="Specific strategies to use (default: all)"
    )
    
    # Puzzle cracking command
    puzzle_parser = subparsers.add_parser(
        "image-crack",
        help="Run end-to-end image cracking pipeline"
    )
    puzzle_parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing images to analyze"
    )
    puzzle_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Custom output directory for results"
    )
    puzzle_parser.add_argument(
        "--keywords",
        nargs="+",
        help="Additional keywords to search for in analysis"
    )
    puzzle_parser.add_argument(
        "--key-numbers",
        type=int,
        nargs="+",
        help="Additional key numbers to use in analysis"
    )
    puzzle_parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file with keywords and key numbers"
    )
    
    # Add more subcommands as we implement them
    
    args = parser.parse_args()
    
    if args.command == "scrape-yews":
        scrape_yews(args.output_dir, args.verbose)
    elif args.command == "stego-analyze":
        analyze_stego(args.input_dir, args.output_dir, args.strategies, args.verbose)
    elif args.command == "image-crack":
        image_crack(args.input_dir, args.output_dir, args.keywords, args.key_numbers, args.config, args.verbose)
    elif args.command is None:
        parser.print_help()
    else:
        logging.error(f"Unknown command: {args.command}")
        parser.print_help()


if __name__ == "__main__":
    main() 