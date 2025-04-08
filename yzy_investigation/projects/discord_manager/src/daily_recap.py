"""Daily recap generator for Discord channels.

This script generates daily recaps of Discord channel activity by:
1. Fetching messages from the past 24 hours
2. Processing them through the message summarizer
3. Generating a formatted recap with links to important messages
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import os
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from yzy_investigation.projects.discord_manager.src.discord_manager import DiscordManager
from yzy_investigation.projects.discord_manager.src.message_summarizer import DiscordMessageSummarizer, SummaryFormat

class DailyRecapGenerator:
    """Generates daily recaps of Discord channel activity."""
    
    def __init__(self, backup_dir: Optional[str] = None, recap_dir: Optional[str] = None):
        """Initialize the daily recap generator.
        
        Args:
            backup_dir: Directory where backups are stored (defaults to project's data directory)
            recap_dir: Directory where recaps will be saved (defaults to project's data directory)
        """
        # Set up directories relative to project root
        self.project_root = project_root
        self.data_dir = self.project_root / "data" / "discord"
        
        self.backup_dir = Path(backup_dir) if backup_dir else self.data_dir / "backups"
        self.recap_dir = Path(recap_dir) if recap_dir else self.data_dir / "recaps"
        
        # Create directories if they don't exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.recap_dir.mkdir(parents=True, exist_ok=True)
        
        # Default channels to process
        self.default_channels = ['game-building', 'irl', 'tech-analysis', 'sanctuary']
        
        # Initialize Discord manager
        self.manager = DiscordManager(os.getenv("DISCORD_BOT_TOKEN"), str(self.backup_dir))
        
        # Initialize message summarizer
        self.summarizer = DiscordMessageSummarizer(
            model="gpt-4",
            format_type=SummaryFormat.MARKDOWN,
            output_dir=str(self.recap_dir)
        )

    async def generate_recap(
        self,
        server_id: int,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        channels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate a daily recap for a server.
        
        Args:
            server_id: ID of the server to generate recap for
            start_time: Optional start time for message filtering
            end_time: Optional end time for message filtering
            channels: Optional list of channel names to process (defaults to default_channels)
            
        Returns:
            Dict containing recap statistics and paths
        """
        if not start_time:
            start_time = datetime.now() - timedelta(days=1)
        if not end_time:
            end_time = datetime.now()
            
        # Use default channels if none specified
        if channels is None:
            channels = self.default_channels
            print(f"Using default channels: {', '.join(channels)}")
            
        # First, backup the messages
        print(f"\nBacking up messages from {start_time} to {end_time}...")
        backup_stats = await self.manager.backup_server(server_id, start_time, end_time)
        
        # Get the most recent backup directory
        backup_dirs = sorted(self.backup_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not backup_dirs:
            raise ValueError("No backup directories found")
            
        latest_backup = backup_dirs[0]
        print(f"Using backup from: {latest_backup}")
        
        # Process each channel
        recap_stats = {
            'channels_processed': 0,
            'total_messages': 0,
            'recap_files': []
        }
        
        # Get server info
        with open(latest_backup / 'server_info.json', 'r', encoding='utf-8') as f:
            server_info = json.load(f)
            
        # Process each text channel
        for channel_info in server_info['channels']:
            if channel_info['type'] != 'text':
                continue
                
            channel_name = channel_info['name']
            
            # Skip channels not in the specified list
            if channel_name not in channels:
                print(f"Skipping #{channel_name} (not in specified channels)")
                continue
                
            channel_dir = latest_backup / channel_name
            
            if not (channel_dir / 'messages.json').exists():
                print(f"No messages found for #{channel_name}")
                continue
                
            # Load messages
            with open(channel_dir / 'messages.json', 'r', encoding='utf-8') as f:
                messages = json.load(f)
                
            if not messages:
                print(f"No messages in #{channel_name} for the specified time period")
                continue
                
            print(f"\nProcessing #{channel_name} ({len(messages)} messages)...")
            
            # Generate summary
            recap_filename = f"recap_{channel_name}_{start_time.strftime('%Y%m%d')}.md"
            recap_path = self.recap_dir / recap_filename
            
            summary = self.summarizer.summarize_messages(messages, recap_path)
            
            recap_stats['channels_processed'] += 1
            recap_stats['total_messages'] += len(messages)
            recap_stats['recap_files'].append(str(recap_path))
            
        return recap_stats

async def main():
    """Main entry point for the daily recap generator."""
    # Get server ID from environment
    server_id = int(os.getenv("DISCORD_SERVER_ID", "0"))
    if server_id == 0:
        print("Please set DISCORD_SERVER_ID in your .env file")
        return
        
    # Initialize generator
    generator = DailyRecapGenerator()
    
    try:
        # Connect to Discord
        await generator.manager.connect()
        
        # Generate recap
        stats = await generator.generate_recap(server_id)
        
        print("\nDaily Recap Generation Complete!")
        print(f"Channels processed: {stats['channels_processed']}")
        print(f"Total messages summarized: {stats['total_messages']}")
        print("\nRecap files generated:")
        for file_path in stats['recap_files']:
            print(f"  - {file_path}")
            
    finally:
        await generator.manager.disconnect()

if __name__ == "__main__":
    asyncio.run(main()) 