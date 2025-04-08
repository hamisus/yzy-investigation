#!/usr/bin/env python3
"""Command-line interface for the Discord manager project."""

import asyncio
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import os
from dotenv import load_dotenv

from yzy_investigation.projects.discord_manager.src.daily_recap import DailyRecapGenerator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Discord Manager CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Daily recap command
    recap_parser = subparsers.add_parser("daily-recap", help="Generate daily recap")
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

def parse_datetime(dt_str: str) -> datetime:
    """Parse datetime string in YYYY-MM-DD HH:MM:SS format."""
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")

async def run_daily_recap(args):
    """Run the daily recap generator."""
    # Load environment variables
    load_dotenv()
    
    # Get server ID from environment
    server_id = int(os.getenv("DISCORD_SERVER_ID", "0"))
    if server_id == 0:
        print("Please set DISCORD_SERVER_ID in your .env file")
        return
    
    # Parse datetime arguments
    start_time = parse_datetime(args.start_time) if args.start_time else None
    end_time = parse_datetime(args.end_time) if args.end_time else None
    
    # Initialize generator
    generator = DailyRecapGenerator(
        backup_dir=args.backup_dir,
        recap_dir=args.recap_dir
    )
    
    try:
        # Connect to Discord
        await generator.manager.connect()
        
        # Generate recap
        stats = await generator.generate_recap(server_id, start_time, end_time)
        
        print("\nDaily Recap Generation Complete!")
        print(f"Channels processed: {stats['channels_processed']}")
        print(f"Total messages summarized: {stats['total_messages']}")
        print("\nRecap files generated:")
        for file_path in stats['recap_files']:
            print(f"  - {file_path}")
            
    finally:
        await generator.manager.disconnect()

def main():
    """Main entry point."""
    args = parse_args()
    
    if args.command == "daily-recap":
        asyncio.run(run_daily_recap(args))
    else:
        print("Please specify a command. Use --help for more information.")

if __name__ == "__main__":
    main() 