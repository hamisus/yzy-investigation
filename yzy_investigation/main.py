"""
Main CLI interface for the YzY investigation project.
"""
import argparse
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Type
from dotenv import load_dotenv
import asyncio

# Load environment variables from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / '.env')

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
from yzy_investigation.projects.discord_manager.src.discord_manager import DiscordManager
from yzy_investigation.projects.discord_manager.src.message_summarizer import DiscordMessageSummarizer, SummaryFormat
from yzy_investigation.projects.discord_manager.src.daily_recap import DailyRecapGenerator
from yzy_investigation.projects.discord_manager.src.summary_publisher import SummaryPublisher
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


def discord_backup(
    server_id: Optional[int] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    output_dir: Optional[Path] = None,
    all_messages: bool = False,
    verbose: bool = False
) -> None:
    """Backup Discord server messages.
    
    Args:
        server_id: Optional ID of the server to backup (defaults to DISCORD_SERVER_ID from .env)
        start_time: Optional start time for message filtering
        end_time: Optional end time for message filtering
        output_dir: Optional output directory for backups
        all_messages: Whether to backup all messages regardless of date
        verbose: Whether to enable verbose logging
    """
    setup_logging(verbose)
    logging.info("Starting Discord backup...")
    
    # Get server ID from environment if not provided
    if server_id is None:
        server_id = int(os.getenv("DISCORD_SERVER_ID", "0"))
        if server_id == 0:
            logging.error("Please set DISCORD_SERVER_ID in your .env file")
            return
    
    # Handle time range
    if all_messages:
        print("Backing up ALL messages (this might take a while)...")
        start_time = None
        end_time = None
    else:
        # Default to last 24 hours if no time range specified
        if not start_time and not end_time:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=1)
            print(f"Using default time range (last 24 hours):")
            print(f"  Start: {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"  End: {end_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        else:
            # Make timestamps timezone-aware if they aren't already
            if start_time and not start_time.tzinfo:
                start_time = start_time.replace(tzinfo=timezone.utc)
            if end_time and not end_time.tzinfo:
                end_time = end_time.replace(tzinfo=timezone.utc)
            print(f"Using specified time range:")
            print(f"  Start: {start_time.strftime('%Y-%m-%d %H:%M:%S %Z') if start_time else 'Beginning'}")
            print(f"  End: {end_time.strftime('%Y-%m-%d %H:%M:%S %Z') if end_time else 'Now'}")
    
    # Set up output directory
    if not output_dir:
        output_dir = Path("data/discord/backups")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Backup directory: {output_dir}")
    
    # Initialize Discord manager
    manager = DiscordManager(os.getenv("DISCORD_BOT_TOKEN"), str(output_dir))
    
    async def run_backup():
        try:
            print("Connecting to Discord...")
            stats = await manager.start_backup(server_id, start_time, end_time)
            
            print("\nBackup completed successfully!")
            print(f"Messages backed up: {stats['message_count']}")
            print(f"Attachments backed up: {stats['attachment_count']}")
            print(f"Channels backed up: {stats['channels_backed_up']}")
            
            if 'error_count' in stats and stats['error_count'] > 0:
                print(f"Errors encountered: {stats['error_count']}")
                
            print(f"Backup saved to: {output_dir}")
            
            logging.info(f"Backup completed. Stats: {stats}")
        except Exception as e:
            logging.error(f"Error during backup: {str(e)}")
            raise
    
    # Run the async function
    asyncio.run(run_backup())


def discord_summarize(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    channels: Optional[List[str]] = None,
    server_id: Optional[str] = None,
    verbose: bool = False
) -> None:
    """Summarize Discord messages from a backup.
    
    Args:
        input_dir: Directory containing message backups
        output_dir: Optional output directory for summaries
        channels: Optional list of channel names to summarize (defaults to ['game-building', 'irl', 'tech-analysis', 'sanctuary'])
        server_id: Optional Discord server ID for message links
        verbose: Whether to enable verbose logging
    """
    setup_logging(verbose)
    logging.info("Starting Discord message summarization...")
    
    # Set up output directory
    if not output_dir:
        output_dir = Path("data/discord/summaries")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set default channels if none provided
    if not channels:
        channels = ['game-building', 'irl', 'tech-analysis', 'sanctuary']
        logging.info(f"Using default channels: {', '.join(channels)}")
        print(f"Using default channels: {', '.join(channels)}")
    
    # Get server ID from environment if not provided
    if not server_id:
        server_id = os.getenv("DISCORD_SERVER_ID", "SERVER_ID")
    
    # Initialize summarizer
    summarizer = DiscordMessageSummarizer(
        model="gpt-4o",
        # format_type=SummaryFormat.MARKDOWN,
        format_type=SummaryFormat.TIMELINE,
        output_dir=str(output_dir)
    )
    
    # Process each channel's messages
    total_channels = 0
    processed_channels = 0
    
    for channel_dir in input_dir.iterdir():
        if not channel_dir.is_dir():
            continue
            
        # Skip channels not in the filter list
        if channel_dir.name not in channels:
            logging.info(f"Skipping channel: {channel_dir.name} (not in channel list)")
            continue
            
        messages_file = channel_dir / "messages.json"
        if not messages_file.exists():
            logging.warning(f"No messages file found for channel: {channel_dir.name}")
            continue
            
        total_channels += 1
        logging.info(f"Processing channel: {channel_dir.name}")
        print(f"Summarizing messages from #{channel_dir.name}...")
        
        # Load messages
        with open(messages_file, 'r', encoding='utf-8') as f:
            messages = json.load(f)
            
        if not messages:
            logging.info(f"No messages found in channel: {channel_dir.name}")
            continue
            
        # Get channel ID from the first message if available
        channel_id = None
        try:
            # Load server_info.json to find channel ID by name
            server_info_path = input_dir / "server_info.json"
            if server_info_path.exists():
                with open(server_info_path, 'r', encoding='utf-8') as f:
                    server_info = json.load(f)
                    for channel_info in server_info.get('channels', []):
                        if channel_info.get('name') == channel_dir.name:
                            channel_id = str(channel_info.get('id'))
                            break
        except Exception as e:
            logging.warning(f"Error getting channel ID: {str(e)}")
        
        # Generate summary
        summary_path = output_dir / f"{channel_dir.name}_summary.md"
        summarizer.summarize_messages(
            messages, 
            summary_path,
            server_id=server_id,
            channel_id=channel_id
        )
        
        processed_channels += 1
        print(f"✓ Summary for #{channel_dir.name} saved to {summary_path}")
        
    logging.info(f"Summarization completed. Processed {processed_channels} of {total_channels} channels.")
    print(f"\nSummarization complete! Processed {processed_channels} channels.")
    print(f"Summaries saved to: {output_dir}")


def discord_daily_recap(
    server_id: Optional[int] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    channels: Optional[List[str]] = None,
    verbose: bool = False
) -> None:
    """Generate a daily recap of Discord server activity.
    
    This command first performs a backup of recent messages and then generates 
    summaries for each channel. Unlike the discord-summarize command (which works
    on existing backups), this command performs the backup and summarization in one step.
    
    Args:
        server_id: Optional ID of the server to generate recap for (defaults to DISCORD_SERVER_ID from .env)
        start_time: Optional start time for message filtering
        end_time: Optional end time for message filtering
        channels: Optional list of channel names to process (defaults to ['game-building', 'irl', 'tech-analysis', 'sanctuary'])
        verbose: Whether to enable verbose logging
    """
    setup_logging(verbose)
    logging.info("Starting Discord daily recap generation...")
    
    # Get server ID from environment if not provided
    if server_id is None:
        server_id = int(os.getenv("DISCORD_SERVER_ID", "0"))
        if server_id == 0:
            logging.error("Please set DISCORD_SERVER_ID in your .env file")
            return
    
    # Set up directories
    backup_dir = Path("data/discord/backups")
    recap_dir = Path("data/discord/recaps")
    backup_dir.mkdir(parents=True, exist_ok=True)
    recap_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = DailyRecapGenerator(
        backup_dir=str(backup_dir),
        recap_dir=str(recap_dir)
    )
    
    async def run_recap():
        try:
            print("Connecting to Discord and backing up messages...")
            await generator.manager.connect()
            stats = await generator.generate_recap(server_id, start_time, end_time, channels)
            
            print("\nDaily Recap Generation Complete!")
            print(f"Channels processed: {stats['channels_processed']}")
            print(f"Total messages summarized: {stats['total_messages']}")
            print("\nRecap files generated:")
            for file_path in stats['recap_files']:
                print(f"  - {file_path}")
                
            logging.info(f"Daily recap generation completed. Stats: {stats}")
        finally:
            await generator.manager.disconnect()
    
    # Run the async function
    asyncio.run(run_recap())


def discord_publish_summary(
    summary_path: Path,
    test_mode: bool = False,
    include_overview: bool = True,
    delay: float = 1.0,
    verbose: bool = False
) -> None:
    """Publish a summary to a Discord channel.
    
    Args:
        summary_path: Path to the summary markdown file
        test_mode: Whether to use test server and channel IDs
        include_overview: Whether to include the overview paragraph
        delay: Delay between messages in seconds
        verbose: Whether to enable verbose logging
    """
    setup_logging(verbose)
    logging.info(f"Starting Discord summary publishing for: {summary_path}")
    
    # Verify the summary file exists
    if not summary_path.exists():
        logging.error(f"Summary file not found: {summary_path}")
        print(f"Error: Summary file not found: {summary_path}")
        return
    
    # Initialize publisher
    publisher = SummaryPublisher(test_mode=test_mode)
    
    # Which mode we're using for logging purposes
    mode = "TEST" if test_mode else "PRODUCTION"
    logging.info(f"Using {mode} mode")
    print(f"Using {mode} mode")
    
    async def run_publisher():
        try:
            print(f"Publishing summary from: {summary_path}")
            print(f"Connecting to Discord...")
            
            stats = await publisher.start(
                str(summary_path),
                include_overview=include_overview,
                delay=delay
            )
            
            print("\nSummary Publishing Complete!")
            print(f"Messages sent: {stats['messages_sent']}")
            print(f"Bullet points sent: {stats['bullet_points_sent']}")
            if stats['errors'] > 0:
                print(f"Errors encountered: {stats['errors']}")
                
            logging.info(f"Publishing completed. Stats: {stats}")
            
        except Exception as e:
            error_msg = f"Error publishing summary: {str(e)}"
            logging.error(error_msg)
            print(f"Error: {error_msg}")
    
    # Run the async function
    asyncio.run(run_publisher())


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
  
  # Backup Discord server (past 24 hours only):
  python -m yzy_investigation.main discord-backup
  
  # Backup all Discord messages:
  python -m yzy_investigation.main discord-backup --all
  
  # Backup with time range:
  python -m yzy_investigation.main discord-backup --start-time "2024-03-20 00:00:00" --end-time "2024-03-21 00:00:00"
  
  # OPTION 1: Summarize messages from an existing backup:
  # Step 1: Create backup first
  python -m yzy_investigation.main discord-backup --output-dir ./data/discord/backups/my_backup
  # Step 2: Summarize the backup (uses default channels: game-building, irl, tech-analysis, sanctuary)
  python -m yzy_investigation.main discord-summarize --input-dir ./data/discord/backups/my_backup
  
  # Summarize specific channels with clickable Discord message links:
  python -m yzy_investigation.main discord-summarize --input-dir ./data/discord/backups/my_backup --channels game-building irl --server-id 1234567890
  
  # OPTION 2: Backup AND summarize in one step:
  # Generate daily recap for the last 24 hours (performs backup and summarization together)
  python -m yzy_investigation.main discord-daily-recap
  
  # Generate recap for specific time range:
  python -m yzy_investigation.main discord-daily-recap --start-time "2024-03-20 00:00:00" --end-time "2024-03-21 00:00:00"
  
  # Publish a summary to Discord:
  python -m yzy_investigation.main discord-publish-summary --summary-path ./data/discord/summaries/game-building_summary.md
  
  # Publish a summary to Discord test environment:
  python -m yzy_investigation.main discord-publish-summary --summary-path ./data/discord/summaries/game-building_summary.md --test-mode
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
    
    # Discord backup command
    discord_backup_parser = subparsers.add_parser(
        "discord-backup",
        help="Backup Discord server messages"
    )
    discord_backup_parser.add_argument(
        "--server-id",
        type=int,
        help="ID of the server to backup (defaults to DISCORD_SERVER_ID from .env)"
    )
    discord_backup_parser.add_argument(
        "--start-time",
        type=str,
        help="Start time for message filtering (YYYY-MM-DD HH:MM:SS)"
    )
    discord_backup_parser.add_argument(
        "--end-time",
        type=str,
        help="End time for message filtering (YYYY-MM-DD HH:MM:SS)"
    )
    discord_backup_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Custom output directory for backups"
    )
    discord_backup_parser.add_argument(
        "--all",
        dest="all_messages",
        action="store_true",
        help="Backup all messages regardless of date"
    )
    
    # Discord summarize command
    discord_summarize_parser = subparsers.add_parser(
        "discord-summarize",
        help="Summarize Discord messages from an existing backup directory"
    )
    discord_summarize_parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing message backups"
    )
    discord_summarize_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Custom output directory for summaries"
    )
    discord_summarize_parser.add_argument(
        "--channels",
        nargs="+",
        help="List of channels to summarize (defaults to game-building, irl, tech-analysis, and sanctuary)"
    )
    discord_summarize_parser.add_argument(
        "--server-id",
        type=str,
        help="Discord server ID for message links (defaults to DISCORD_SERVER_ID from .env)"
    )
    
    # Discord daily recap command
    discord_recap_parser = subparsers.add_parser(
        "discord-daily-recap",
        help="Backup recent messages AND generate summaries in one operation"
    )
    discord_recap_parser.add_argument(
        "--server-id",
        type=int,
        help="ID of the server to generate recap for (defaults to DISCORD_SERVER_ID from .env)"
    )
    discord_recap_parser.add_argument(
        "--start-time",
        type=str,
        help="Start time for message filtering (YYYY-MM-DD HH:MM:SS)"
    )
    discord_recap_parser.add_argument(
        "--end-time",
        type=str,
        help="End time for message filtering (YYYY-MM-DD HH:MM:SS)"
    )
    discord_recap_parser.add_argument(
        "--channels",
        nargs="+",
        help="List of channels to process (defaults to game-building, irl, tech-analysis, and sanctuary)"
    )
    
    # Discord publish summary command
    discord_publish_parser = subparsers.add_parser(
        "discord-publish-summary",
        help="Publish a summary to a Discord channel"
    )
    discord_publish_parser.add_argument(
        "--summary-path",
        type=Path,
        required=True,
        help="Path to the summary markdown file to publish"
    )
    discord_publish_parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Use test server and channel IDs from environment variables"
    )
    discord_publish_parser.add_argument(
        "--include-overview",
        action="store_true",
        default=True,
        help="Include the overview paragraph in the published messages"
    )
    discord_publish_parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between messages in seconds to avoid rate limiting"
    )
    
    args = parser.parse_args()
    
    if args.command == "scrape-yews":
        scrape_yews(args.output_dir, args.verbose)
    elif args.command == "stego-analyze":
        analyze_stego(args.input_dir, args.output_dir, args.strategies, args.verbose)
    elif args.command == "image-crack":
        image_crack(args.input_dir, args.output_dir, args.keywords, args.key_numbers, args.config, args.verbose)
    elif args.command == "discord-backup":
        start_time = datetime.strptime(args.start_time, "%Y-%m-%d %H:%M:%S") if args.start_time else None
        end_time = datetime.strptime(args.end_time, "%Y-%m-%d %H:%M:%S") if args.end_time else None
        discord_backup(args.server_id, start_time, end_time, args.output_dir, args.all_messages, args.verbose)
    elif args.command == "discord-summarize":
        discord_summarize(args.input_dir, args.output_dir, args.channels, args.server_id, args.verbose)
    elif args.command == "discord-daily-recap":
        start_time = datetime.strptime(args.start_time, "%Y-%m-%d %H:%M:%S") if args.start_time else None
        end_time = datetime.strptime(args.end_time, "%Y-%m-%d %H:%M:%S") if args.end_time else None
        discord_daily_recap(args.server_id, start_time, end_time, args.channels, args.verbose)
    elif args.command == "discord-publish-summary":
        discord_publish_summary(args.summary_path, args.test_mode, args.include_overview, args.delay, args.verbose)
    elif args.command is None:
        parser.print_help()
    else:
        logging.error(f"Unknown command: {args.command}")
        parser.print_help()


if __name__ == "__main__":
    main() 