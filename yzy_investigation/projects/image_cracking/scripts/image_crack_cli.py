#!/usr/bin/env python3
"""
End-to-end puzzle cracking CLI for steganography analysis.

This script provides a unified interface to:
1. Run multiple steganography analysis strategies on images
2. Process the results to find hidden data
3. Generate clear output of findings with significance ratings

The pipeline can handle a variety of steganographic techniques
and automatically identifies the most promising results.
"""

import argparse
import json
import logging
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Type, Tuple, Union

from tqdm import tqdm

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
from yzy_investigation.projects.image_cracking.process_stego_results import (
    StegoResultProcessor,
    ResultSignificanceChecker
)
from yzy_investigation.core.log_manager import setup_logging


class ProgressTracker:
    """
    Track progress and estimate time remaining for long-running tasks.
    
    This class provides terminal-based progress updates with ETA information.
    """
    
    def __init__(
        self, 
        total: int, 
        desc: str = "Processing", 
        unit: str = "item",
        log_interval: int = 1
    ) -> None:
        """
        Initialize the progress tracker.
        
        Args:
            total: Total number of items to process
            desc: Description of the task
            unit: Unit of items being processed
            log_interval: Interval in seconds between progress updates
        """
        self.total = total
        self.desc = desc
        self.unit = unit
        self.log_interval = log_interval
        
        self.current = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_update_count = 0
        
        # Create a progress bar
        self.pbar = tqdm(total=total, desc=desc, unit=unit)
        
    def update(self, increment: int = 1) -> None:
        """
        Update the progress tracker.
        
        Args:
            increment: Number of items to increment the counter by
        """
        self.current += increment
        self.pbar.update(increment)
        
        current_time = time.time()
        time_diff = current_time - self.last_update_time
        
        # Update rate stats periodically for more accurate ETA
        if time_diff >= self.log_interval:
            items_since_last = self.current - self.last_update_count
            rate = items_since_last / time_diff if time_diff > 0 else 0
            
            self.last_update_time = current_time
            self.last_update_count = self.current
    
    def finish(self) -> None:
        """Complete the progress tracking."""
        self.pbar.close()
        elapsed = time.time() - self.start_time
        print(f"\n{self.desc} completed in {self._format_time(elapsed)}")
    
    def _format_time(self, seconds: float) -> str:
        """
        Format time in a human-readable format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{int(minutes)} minutes {int(seconds % 60)} seconds"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{int(hours)} hours {int(minutes)} minutes"


class ImageCrackingPipeline:
    """
    Comprehensive pipeline for cracking steganography puzzles.
    
    This class orchestrates the end-to-end process of analyzing images 
    for hidden data, processing the results, and generating reports.
    """
    
    def __init__(
        self, 
        input_dir: Path, 
        output_dir: Optional[Path] = None,
        keywords: Optional[List[str]] = None,
        key_numbers: Optional[List[int]] = None,
        config_file: Optional[Path] = None,
        verbose: bool = False
    ) -> None:
        """
        Initialize the puzzle cracking pipeline.
        
        Args:
            input_dir: Directory containing images to analyze
            output_dir: Optional output directory for results
            keywords: Optional list of keywords to use in analysis
            key_numbers: Optional list of key numbers to use in analysis
            config_file: Optional path to a configuration file with keywords and key numbers
            verbose: Whether to enable verbose logging
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else Path('results/image_cracking')
        self.verbose = verbose
        
        # Set up logging
        self.logger = setup_logging(
            "image_cracking", 
            log_level=logging.DEBUG if verbose else logging.INFO,
            log_dir=self.output_dir / "logs"
        )
        
        # Load keywords and key numbers from config file if provided
        config_keywords, config_key_numbers = self._load_config(config_file)
        
        # Merge keywords and key numbers from arguments and config file
        self.custom_keywords = list(set((keywords or []) + config_keywords))
        self.custom_key_numbers = list(set((key_numbers or []) + config_key_numbers))
        
        # Create output directories
        self.stego_output_dir = self.output_dir / "stego_analysis"
        self.processing_output_dir = self.output_dir / "processed_results"
        
        for directory in [self.output_dir, self.stego_output_dir, self.processing_output_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        self.logger.info(f"Initialized pipeline: Input={input_dir}, Output={self.output_dir}")
        if self.custom_keywords:
            self.logger.info(f"Using keywords: {', '.join(self.custom_keywords)}")
        if self.custom_key_numbers:
            self.logger.info(f"Using key numbers: {', '.join(map(str, self.custom_key_numbers))}")
            
    def _load_config(self, config_file: Optional[Path]) -> Tuple[List[str], List[int]]:
        """
        Load keywords and key numbers from a configuration file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Tuple containing:
                - List of keywords
                - List of key numbers
        """
        keywords = []
        key_numbers = []
        
        # If no config file provided, try the default location
        if config_file is None:
            default_config = Path(__file__).parent.parent / "config" / "keywords.json"
            if default_config.exists():
                config_file = default_config
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                keywords = config.get('keywords', [])
                key_numbers = config.get('key_numbers', [])
                
                self.logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                self.logger.error(f"Error loading config file {config_file}: {e}")
        
        return keywords, key_numbers
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete puzzle cracking pipeline.
        
        Returns:
            Dictionary with pipeline results summary
        """
        start_time = time.time()
        self.logger.info("Starting puzzle cracking pipeline")
        
        print("\n🔍 Starting Puzzle Cracking Pipeline")
        print("------------------------------------")
        
        # Count images to process
        image_files = self._count_images()
        print(f"Found {image_files} images to analyze")
        
        try:
            # Step 1: Run stego analysis with all strategies
            print(f"\n⚙️  Step 1/3: Running steganography analysis strategies")
            stego_results = self._run_stego_analysis()
            
            if not stego_results.get("success", False):
                self.logger.error("Stego analysis failed. Aborting pipeline.")
                raise RuntimeError(f"Steganography analysis failed: {stego_results.get('error', 'Unknown error')}")
            
            # Step 2: Process and analyze the results
            print(f"\n🔄 Step 2/3: Processing analysis results")
            stego_results_dir = Path(stego_results["results_directory"])
            
            # Ensure the results directory exists and contains expected files
            if not stego_results_dir.exists():
                self.logger.error(f"Results directory not found: {stego_results_dir}")
                raise FileNotFoundError(f"Results directory not found: {stego_results_dir}")
                
            # Check for the analysis_summary.json file
            summary_file = stego_results_dir / "analysis_summary.json"
            if not summary_file.exists():
                self.logger.error(f"Analysis summary file not found: {summary_file}")
                raise FileNotFoundError(f"Analysis summary file not found: {summary_file}")
                
            # Check for the extracted_data directory
            extracted_data_dir = stego_results_dir / "extracted_data"
            if not extracted_data_dir.exists():
                self.logger.error(f"Extracted data directory not found: {extracted_data_dir}")
                raise FileNotFoundError(f"Extracted data directory not found: {extracted_data_dir}")
                
            # Process the results
            processing_results = self._process_results(stego_results_dir)
            
            # Step 3: Generate final report
            print(f"\n📊 Step 3/3: Generating final report")
            report = self._generate_report(stego_results, processing_results)
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Pipeline complete in {self._format_time(elapsed_time)}")
            
            print(f"\n✅ Pipeline complete in {self._format_time(elapsed_time)}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error in pipeline: {str(e)}")
            print(f"\n❌ Error in pipeline: {str(e)}")
            
            # Try to recover and return partial results if possible
            partial_report = {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "elapsed_time": time.time() - start_time,
                "images_analyzed": image_files,
                "hidden_data_detected": 0,
                "target_string_found": False,
                "high_significance_findings": 0,
                "medium_significance_findings": 0,
                "files_discovered": 0
            }
            
            print("\n⚠️ Pipeline encountered errors but attempted to recover.")
            print("   Check logs for details.")
            
            return partial_report
    
    def _count_images(self) -> int:
        """
        Count the number of images in the input directory.
        
        Returns:
            Number of image files
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
        count = sum(1 for f in self.input_dir.glob('**/*') 
                   if f.is_file() and f.suffix.lower() in image_extensions)
        return count
    
    def _format_time(self, seconds: float) -> str:
        """
        Format time in a human-readable format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{int(minutes)} minutes {int(seconds % 60)} seconds"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{int(hours)} hours {int(minutes)} minutes"
    
    def _run_stego_analysis(self) -> Dict[str, Any]:
        """
        Run steganography analysis with all available strategies.
        
        Returns:
            Dictionary with analysis results
        """
        self.logger.info("Running steganography analysis")
        
        # Configure all strategies
        strategies = [
            LsbStrategy,
            ColorHistogramStrategy,
            FileSignatureStrategy,
            MetadataAnalysisStrategy,
            KeywordXorStrategy,
            ShiftCipherStrategy,
            BitSequenceStrategy,
            BlakeHashStrategy,
            CustomRgbEncodingStrategy
        ]
        
        print(f"Using {len(strategies)} analysis strategies")
        
        # Create and run the pipeline
        pipeline = StegAnalysisPipeline(
            input_path=self.input_dir,
            output_path=self.stego_output_dir,
            strategies=strategies
        )
        
        # Configure strategy keywords and numbers if needed
        if self.custom_keywords or self.custom_key_numbers:
            for strategy_class in strategies:
                if hasattr(strategy_class, 'KEY_TERMS') and self.custom_keywords:
                    # Make sure we don't have duplicates
                    existing_terms = set(strategy_class.KEY_TERMS)
                    new_terms = [term for term in self.custom_keywords if term not in existing_terms]
                    if new_terms:
                        strategy_class.KEY_TERMS = list(strategy_class.KEY_TERMS) + new_terms
                    
                if hasattr(strategy_class, 'KEY_NUMBERS') and self.custom_key_numbers:
                    # Make sure we don't have duplicates
                    existing_numbers = set(strategy_class.KEY_NUMBERS)
                    new_numbers = [num for num in self.custom_key_numbers if num not in existing_numbers]
                    if new_numbers:
                        strategy_class.KEY_NUMBERS = list(strategy_class.KEY_NUMBERS) + new_numbers
        
        # Replace pipeline's run method with our progress tracking version
        original_run = pipeline.run
        
        def run_with_progress() -> Dict[str, Any]:
            image_files = list(pipeline.find_images())
            progress = ProgressTracker(len(image_files), "Analyzing images", "image")
            
            # Create a wrapper for analyze_image that updates progress
            original_analyze = pipeline.analyze_image
            
            def analyze_with_progress(image_path):
                result = original_analyze(image_path)
                progress.update()
                return result
            
            pipeline.analyze_image = analyze_with_progress
            
            # Run the pipeline
            result = original_run()
            progress.finish()
            return result
        
        pipeline.run = run_with_progress
        
        results = pipeline.run()
        
        if results["success"]:
            self.logger.info(f"Stego analysis completed with {results['summary']['total_images']} images processed")
            self.logger.info(f"Results saved to: {results['results_directory']}")
        else:
            self.logger.error(f"Stego analysis failed: {results.get('error', 'Unknown error')}")
            
        return results
    
    def _process_results(self, stego_results_dir: Path) -> Dict[str, Any]:
        """
        Process steganography results to find patterns and hidden data.
        
        Args:
            stego_results_dir: Directory containing stego analysis results
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info("Processing steganography results")
        
        processor = StegoResultProcessor(
            results_dir=stego_results_dir,
            output_dir=self.processing_output_dir
        )
        
        # Replace process_all_images with progress tracking version
        original_process = processor.process_all_images
        
        def process_with_progress() -> Dict[str, Any]:
            # First, count how many images we need to process
            extracted_data_dir = Path(stego_results_dir) / "extracted_data"
            image_dirs = [d for d in extracted_data_dir.iterdir() if d.is_dir()]
            
            progress = ProgressTracker(len(image_dirs), "Processing results", "image")
            
            # Create a wrapper for _process_image_data that updates progress
            original_process_image = processor._process_image_data
            
            def process_image_with_progress(image_dir):
                result = original_process_image(image_dir)
                progress.update()
                return result
            
            processor._process_image_data = process_image_with_progress
            
            # Run the processor
            result = original_process()
            progress.finish()
            return result
        
        processor.process_all_images = process_with_progress
        
        results = processor.process_all_images()
        
        self.logger.info(f"Results processing complete. Found {results['files_discovered']} potential files.")
        if results.get("significance_analysis", {}).get("found_target", False):
            self.logger.warning("TARGET STRING FOUND in analysis results!")
            
        return results
    
    def _generate_report(
        self, 
        stego_results: Dict[str, Any], 
        processing_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive report of findings.
        
        Args:
            stego_results: Results from steganography analysis
            processing_results: Results from processing
            
        Returns:
            Dictionary with final report
        """
        self.logger.info("Generating final report")
        
        # Extract key information
        sign_analysis = processing_results.get("significance_analysis", {})
        high_sign_results = sign_analysis.get("high_significance_results", [])
        medium_sign_results = sign_analysis.get("medium_significance_results", [])
        low_sign_results = sign_analysis.get("low_significance_results", [])
        high_sign_count = len(high_sign_results)
        medium_sign_count = len(medium_sign_results)
        
        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "input_directory": str(self.input_dir),
            "output_directory": str(self.output_dir),
            "images_analyzed": stego_results.get("summary", {}).get("total_images", 0),
            "hidden_data_detected": stego_results.get("summary", {}).get("images_with_hidden_data", 0),
            "target_string_found": sign_analysis.get("found_target", False),
            "high_significance_findings": high_sign_count,
            "medium_significance_findings": medium_sign_count,
            "low_significance_findings": len(low_sign_results),
            "files_discovered": processing_results.get("files_discovered", 0),
            "file_types": processing_results.get("file_types_found", {}),
            "strategies": stego_results.get("summary", {}).get("strategies", []),
            "keywords_used": self.custom_keywords,
            "key_numbers_used": self.custom_key_numbers,
            "high_significance_results": high_sign_results,
            "medium_significance_results": medium_sign_results[:5]  # Limit to top 5
        }
        
        # Save report to file
        report_path = self.output_dir / "final_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
            
        # Create a human-readable detailed summary
        summary_path = self.output_dir / "summary.txt"
        with open(summary_path, "w") as f:
            f.write("PUZZLE CRACKING SUMMARY\n")
            f.write("======================\n\n")
            f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Images analyzed: {report['images_analyzed']}\n")
            f.write(f"Images with hidden data: {report['hidden_data_detected']}\n")
            f.write(f"Files discovered: {report['files_discovered']}\n\n")
            
            if report['target_string_found']:
                f.write("!!! TARGET STRING FOUND !!!\n")
                f.write("See TARGET_FOUND.txt for details\n\n")
                
            f.write(f"Significance findings:\n")
            f.write(f"  High significance findings: {high_sign_count}\n")
            f.write(f"  Medium significance findings: {medium_sign_count}\n")
            f.write(f"  Low significance findings: {len(low_sign_results)}\n\n")
            
            # List file types found
            if report['file_types']:
                f.write("File types discovered:\n")
                for file_type, count in report['file_types'].items():
                    f.write(f"  - {file_type}: {count}\n")
                f.write("\n")
            
            # List high significance findings
            if high_sign_results:
                f.write("HIGH SIGNIFICANCE FINDINGS:\n")
                f.write("-------------------------\n")
                for i, result in enumerate(high_sign_results):
                    f.write(f"[{i+1}] Source: {result.get('source', 'Unknown')}\n")
                    for finding in result.get('findings', []):
                        if finding.get('type') == 'target_string':
                            f.write(f"    Target string found: {finding.get('term', '')}\n")
                            f.write(f"    Context: {finding.get('context', '')}\n")
                        elif finding.get('type') == 'high_value_term':
                            f.write(f"    High-value term: {finding.get('term', '')}\n")
                            f.write(f"    Context: {finding.get('context', '')}\n")
                    f.write("\n")
                f.write("\n")
                
            # List medium significance findings (limited)
            if medium_sign_results:
                f.write("MEDIUM SIGNIFICANCE FINDINGS (top 5):\n")
                f.write("----------------------------------\n")
                for i, result in enumerate(medium_sign_results[:5]):
                    f.write(f"[{i+1}] Source: {result.get('source', 'Unknown')}\n")
                    for finding in result.get('findings', []):
                        if finding.get('type') == 'high_value_term':
                            f.write(f"    High-value term: {finding.get('term', '')}\n")
                            f.write(f"    Context: {finding.get('context', '')}\n")
                    f.write("\n")
                f.write("\n")
            
            f.write(f"Keywords used: {', '.join(self.custom_keywords)}\n")
            f.write(f"Key numbers used: {', '.join(map(str, self.custom_key_numbers))}\n\n")
            
            f.write("Detailed results:\n")
            f.write(f"  - Stego analysis: {self.stego_output_dir}\n")
            f.write(f"  - Processed results: {self.processing_output_dir}\n")
            f.write(f"  - Extracted data: {self.processing_output_dir / 'reconstructed_files'}\n")
            f.write(f"  - Combined LSB data: {self.processing_output_dir / 'combined_lsb_data'}\n")
        
        # Create a high-value findings summary
        findings_path = self.output_dir / "significant_findings.txt"
        with open(findings_path, "w") as f:
            f.write("SIGNIFICANT FINDINGS SUMMARY\n")
            f.write("===========================\n\n")
            
            if report['target_string_found']:
                f.write("!!! TARGET STRING FOUND !!!\n\n")
                
            # High significance findings
            if high_sign_results:
                f.write("HIGH SIGNIFICANCE FINDINGS:\n")
                f.write("-------------------------\n")
                for i, result in enumerate(high_sign_results):
                    f.write(f"[{i+1}] Source: {result.get('source', 'Unknown')}\n")
                    for finding in result.get('findings', []):
                        if finding.get('type') == 'target_string':
                            f.write(f"    Target string found: {finding.get('term', '')}\n")
                            f.write(f"    Context: {finding.get('context', '')}\n")
                        elif finding.get('type') == 'high_value_term':
                            f.write(f"    High-value term: {finding.get('term', '')}\n")
                            f.write(f"    Context: {finding.get('context', '')}\n")
                        elif finding.get('type') == 'readable_text':
                            f.write(f"    Readable text: {finding.get('sample', '')}\n")
                    f.write("\n")
                f.write("\n")
                
            # Medium significance findings
            if medium_sign_results:
                f.write("MEDIUM SIGNIFICANCE FINDINGS:\n")
                f.write("---------------------------\n")
                for i, result in enumerate(medium_sign_results):
                    f.write(f"[{i+1}] Source: {result.get('source', 'Unknown')}\n")
                    for finding in result.get('findings', []):
                        if finding.get('type') == 'high_value_term':
                            f.write(f"    High-value term: {finding.get('term', '')}\n")
                            f.write(f"    Context: {finding.get('context', '')}\n")
                        elif finding.get('type') == 'readable_text':
                            f.write(f"    Readable text: {finding.get('sample', '')}\n")
                    f.write("\n")
                    # Limit to 10 medium significance results to avoid too much output
                    if i >= 9:
                        f.write(f"... and {len(medium_sign_results) - 10} more medium significance findings\n\n")
                        break
                f.write("\n")
            
            # File discoveries
            if report['files_discovered'] > 0:
                f.write("DISCOVERED FILES:\n")
                f.write("----------------\n")
                for file_type, count in report['file_types'].items():
                    f.write(f"  - {file_type}: {count}\n")
                f.write(f"\nSee reconstructed_files directory for extracted files.\n\n")
        
        self.logger.info(f"Report generated: {report_path}")
        self.logger.info(f"Summary generated: {summary_path}")
        self.logger.info(f"Findings summary generated: {findings_path}")
        
        return report


def main() -> None:
    """Main entry point for the puzzle cracking CLI."""
    parser = argparse.ArgumentParser(
        description="End-to-end puzzle cracking pipeline for steganography analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run puzzle cracking on a directory of images:
  python -m yzy_investigation.projects.image_cracking.scripts.image_crack_cli --input-dir ./data/raw/yews
  
  # Specify custom output directory:
  python -m yzy_investigation.projects.image_cracking.scripts.image_crack_cli --input-dir ./data/raw/yews --output-dir ./my_results
  
  # Add custom keywords to search for:
  python -m yzy_investigation.projects.image_cracking.scripts.image_crack_cli --input-dir ./data/raw/yews --keywords code secret hidden
  
  # Add custom key numbers:
  python -m yzy_investigation.projects.image_cracking.scripts.image_crack_cli --input-dir ./data/raw/yews --key-numbers 42 123 456
  
  # Use a custom configuration file:
  python -m yzy_investigation.projects.image_cracking.scripts.image_crack_cli --input-dir ./data/raw/yews --config ./my_keywords.json
"""
    )
    
    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        required=True,
        help="Directory containing images to analyze"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Directory to store results (default: results/image_cracking)"
    )
    
    parser.add_argument(
        "--keywords", "-k",
        nargs="+",
        help="Additional keywords to search for in the analysis"
    )
    
    parser.add_argument(
        "--key-numbers", "-n",
        type=int,
        nargs="+",
        help="Additional key numbers to use in the analysis"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to configuration file with keywords and key numbers"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    try:
        pipeline = ImageCrackingPipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            keywords=args.keywords,
            key_numbers=args.key_numbers,
            config_file=args.config,
            verbose=args.verbose
        )
        
        results = pipeline.run()
        
        # Print summary
        print("\n📋 PUZZLE CRACKING SUMMARY:")
        print("========================")
        print(f"🖼️  Images analyzed: {results['images_analyzed']}")
        print(f"🔍 Images with hidden data: {results['hidden_data_detected']}")
        print(f"📁 Files discovered: {results['files_discovered']}")
        
        if results['target_string_found']:
            print("\n🚨 !!! TARGET STRING FOUND !!!")
            print("   See TARGET_FOUND.txt for details")
            
        print(f"\n⭐ High significance findings: {results['high_significance_findings']}")
        print(f"✳️  Medium significance findings: {results['medium_significance_findings']}")
        
        # Print file types discovered
        if results.get('file_types'):
            print("\n📄 File types discovered:")
            for file_type, count in results['file_types'].items():
                print(f"   - {file_type}: {count}")
        
        # Show top high significance findings
        if results.get('high_significance_results'):
            print("\n🔴 TOP HIGH SIGNIFICANCE FINDINGS:")
            for i, result in enumerate(results['high_significance_results'][:3]):  # Show top 3
                print(f"   [{i+1}] Source: {result.get('source', 'Unknown')}")
                for finding in result.get('findings', []):
                    if finding.get('type') == 'target_string':
                        print(f"      • Target string found: {finding.get('term', '')}")
                        print(f"        Context: {finding.get('context', '')[:100]}...")
                    elif finding.get('type') == 'high_value_term':
                        print(f"      • High-value term: {finding.get('term', '')}")
                        print(f"        Context: {finding.get('context', '')[:100]}...")
            if len(results['high_significance_results']) > 3:
                print(f"   ... and {len(results['high_significance_results']) - 3} more (see significant_findings.txt)")
        
        # Show top medium significance findings
        if results.get('medium_significance_results'):
            print("\n🟠 TOP MEDIUM SIGNIFICANCE FINDINGS:")
            for i, result in enumerate(results['medium_significance_results'][:2]):  # Show top 2
                print(f"   [{i+1}] Source: {result.get('source', 'Unknown')}")
                for finding in result.get('findings', []):
                    if finding.get('type') == 'high_value_term':
                        print(f"      • High-value term: {finding.get('term', '')}")
                        print(f"        Context: {finding.get('context', '')[:100]}...")
            print(f"   ... see significant_findings.txt for more")
                
        print(f"\n📂 Full results saved to: {args.output_dir or 'results/image_cracking'}")
        print(f"📝 Detailed findings in: {(args.output_dir or Path('results/image_cracking')) / 'significant_findings.txt'}")
        print(f"📊 Analysis summary in: {(args.output_dir or Path('results/image_cracking')) / 'summary.txt'}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 