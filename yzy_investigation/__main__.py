#!/usr/bin/env python3
"""Main entry point for the YzY Investigation toolkit."""

import argparse
from pathlib import Path
import sys

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YzY Investigation Toolkit")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # X Spaces commands
    spaces_parser = subparsers.add_parser("x-spaces", help="X Spaces downloader")
    spaces_parser.add_argument("url", help="URL of the Space to download")
    spaces_parser.add_argument("--output-dir", help="Output directory for downloads")
    
    # Web scraper commands
    scraper_parser = subparsers.add_parser("scrape-yews", help="YEWS.news scraper")
    scraper_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    scraper_parser.add_argument("--output-dir", help="Output directory for scraped data")
    
    # Image cracking commands
    image_parser = subparsers.add_parser("image-crack", help="Image cracking tools")
    image_parser.add_argument("--input-dir", required=True, help="Input directory containing images")
    image_parser.add_argument("--output-dir", help="Output directory for results")
    
    # Stego analysis commands
    stego_parser = subparsers.add_parser("stego-analyze", help="Steganography analysis")
    stego_parser.add_argument("path", help="Path to image or directory")
    stego_parser.add_argument("--use-keywords", action="store_true", help="Use image cracking keywords")
    stego_parser.add_argument("-o", "--output", help="Output file for results")
    stego_parser.add_argument("-e", "--extensions", nargs="+", help="File extensions to analyze")
    
    # Discord manager commands
    discord_parser = subparsers.add_parser("discord", help="Discord manager")
    discord_subparsers = discord_parser.add_subparsers(dest="discord_command", help="Discord command to run")
    
    # Daily recap command
    recap_parser = discord_subparsers.add_parser("daily-recap", help="Generate daily recap")
    recap_parser.add_argument(
        "--start-time",
        type=str,
        help="Start time for message filtering (YYYY-MM-DD HH:MM:SS)"
    )
    recap_parser.add_argument(
        "--end-time",
        type=str,
        help="End time for message filtering (YYYY-MM-DD HH:MM:SS)"
    )
    recap_parser.add_argument(
        "--backup-dir",
        type=str,
        help="Directory to store backups"
    )
    recap_parser.add_argument(
        "--recap-dir",
        type=str,
        help="Directory to store recaps"
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    if not args.command:
        print("Please specify a command. Use --help for more information.")
        return
        
    if args.command == "x-spaces":
        from yzy_investigation.projects.x_spaces import SpaceDownloader
        downloader = SpaceDownloader(output_dir=args.output_dir)
        downloader.download_space(args.url)
        
    elif args.command == "scrape-yews":
        from yzy_investigation.projects.web_scraper import YewsScraper
        scraper = YewsScraper(verbose=args.verbose, output_dir=args.output_dir)
        scraper.scrape()
        
    elif args.command == "image-crack":
        from yzy_investigation.projects.image_cracking import ImageCracker
        cracker = ImageCracker(input_dir=args.input_dir, output_dir=args.output_dir)
        cracker.process_directory()
        
    elif args.command == "stego-analyze":
        from yzy_investigation.projects.stego_analysis import StegoAnalyzer
        analyzer = StegoAnalyzer(
            use_keywords=args.use_keywords,
            output_file=args.output,
            extensions=args.extensions
        )
        if Path(args.path).is_dir():
            analyzer.batch_analyze(args.path)
        else:
            analyzer.analyze(args.path)
            
    elif args.command == "discord":
        if not args.discord_command:
            print("Please specify a Discord command. Use --help for more information.")
            return
            
        if args.discord_command == "daily-recap":
            from yzy_investigation.projects.discord_manager.src.cli import run_daily_recap
            import asyncio
            asyncio.run(run_daily_recap(args))

if __name__ == "__main__":
    main() 