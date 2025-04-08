"""Discord Manager - A comprehensive Discord server management tool.

This module provides functionality for managing Discord servers including:
- Server backups with message history
- Channel management
- Message summarization
- User management
"""

import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone, timedelta

import discord
from discord.ext import commands
from discord import Guild, TextChannel, Message
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DiscordManager:
    """Main class for managing Discord servers and their content."""
    
    def __init__(self, token: str, backup_dir: str = "discord_backups"):
        """Initialize the Discord manager.
        
        Args:
            token: Discord bot token for authentication
            backup_dir: Directory where backups will be stored
        """
        self.token = token
        self.backup_dir = Path(backup_dir)
        
        # Configure intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.guilds = True
        
        self.bot = commands.Bot(command_prefix="!", intents=intents)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Store the backup parameters
        self._server_id = None
        self._start_time = None
        self._end_time = None
        self._backup_complete = asyncio.Future()
        
    async def start_backup(
        self,
        server_id: int,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Start the backup process.
        
        Args:
            server_id: ID of the server to backup
            start_time: Optional start time for filtering messages
            end_time: Optional end time for filtering messages
            
        Returns:
            Dict[str, Any]: Statistics about the backup
        """
        self._server_id = server_id
        self._start_time = start_time
        self._end_time = end_time
        self._backup_complete = asyncio.Future()
        
        @self.bot.event
        async def on_ready():
            logger.info(f"Logged in as {self.bot.user}")
            if not self._backup_complete.done():
                try:
                    stats = await self._do_backup()
                    self._backup_complete.set_result(stats)
                except Exception as e:
                    self._backup_complete.set_exception(e)
                finally:
                    await self.bot.close()
        
        try:
            await self.bot.start(self.token)
        except Exception as e:
            if not self._backup_complete.done():
                self._backup_complete.set_exception(e)
            raise
            
        return await self._backup_complete
        
    async def _do_backup(self) -> Dict[str, Any]:
        """Perform the actual backup operation."""
        server = self.bot.get_guild(self._server_id)
        if not server:
            raise ValueError(f"Server with ID {self._server_id} not found")
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{server.name}_{timestamp}"
        if self._start_time and self._end_time:
            backup_name += f"_from_{self._start_time.strftime('%Y%m%d')}_to_{self._end_time.strftime('%Y%m%d')}"
            
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        logger.info(f"Starting backup of server: {server.name}")
        if self._start_time and self._end_time:
            logger.info(f"Time range: {self._start_time} to {self._end_time}")
        
        # Save server information
        server_info = {
            'name': server.name,
            'id': server.id,
            'member_count': server.member_count,
            'created_at': server.created_at.isoformat(),
            'roles': [
                {
                    'name': role.name,
                    'id': role.id,
                    'color': role.color.value,
                    'permissions': role.permissions.value
                }
                for role in server.roles
            ],
            'channels': [
                {
                    'name': channel.name,
                    'id': channel.id,
                    'type': str(channel.type),
                    'category': channel.category.name if channel.category else None
                }
                for channel in server.channels
            ]
        }
        
        with open(backup_path / 'server_info.json', 'w', encoding='utf-8') as f:
            json.dump(server_info, f, indent=2, ensure_ascii=False)
        logger.info("Server information saved")
            
        # Backup each text channel
        total_stats = {
            'message_count': 0,
            'attachment_count': 0,
            'error_count': 0,
            'channels_backed_up': 0
        }
        
        for channel in server.text_channels:
            logger.info(f"Backing up channel: #{channel.name}")
            channel_stats = await self.backup_channel(
                channel, 
                backup_path, 
                self._start_time, 
                self._end_time
            )
            for key in total_stats:
                if key in channel_stats:
                    total_stats[key] += channel_stats[key]
            total_stats['channels_backed_up'] += 1
            logger.info(f"Channel #{channel.name} backup complete")
            
        logger.info("Server backup complete!")
        return total_stats

    async def backup_channel(
        self,
        channel: TextChannel,
        backup_path: Path,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Backup messages from a single channel.
        
        Args:
            channel: The channel to backup
            backup_path: Where to store the backup
            start_time: Optional start time for filtering messages
            end_time: Optional end time for filtering messages
            
        Returns:
            Dict[str, Any]: Statistics about the backup
        """
        stats = {
            'message_count': 0,
            'attachment_count': 0,
            'error_count': 0
        }
        
        try:
            channel_dir = backup_path / channel.name
            channel_dir.mkdir(exist_ok=True)
            
            # Ensure timestamps are timezone-aware
            if start_time and not start_time.tzinfo:
                start_time = start_time.replace(tzinfo=timezone.utc)
            if end_time and not end_time.tzinfo:
                end_time = end_time.replace(tzinfo=timezone.utc)
            
            messages = []
            async for message in channel.history(
                limit=None,
                oldest_first=True,
                after=start_time,
                before=end_time
            ):
                # Skip messages outside our time range
                message_time = message.created_at.replace(tzinfo=timezone.utc)
                if start_time and message_time < start_time:
                    continue
                if end_time and message_time > end_time:
                    continue
                
                message_data = {
                    'id': message.id,
                    'content': message.content,
                    'author': str(message.author),
                    'author_id': message.author.id,
                    'timestamp': message.created_at.isoformat(),
                    'attachments': [
                        {
                            'filename': attachment.filename,
                            'url': attachment.url,
                            'size': attachment.size
                        }
                        for attachment in message.attachments
                    ],
                    'embeds': [embed.to_dict() for embed in message.embeds],
                    'reactions': [
                        {
                            'emoji': str(reaction.emoji),
                            'count': reaction.count
                        }
                        for reaction in message.reactions
                    ]
                }
                messages.append(message_data)
                stats['message_count'] += 1
                stats['attachment_count'] += len(message.attachments)
                
                # Log progress every 100 messages
                if stats['message_count'] % 100 == 0:
                    logger.info(f"Retrieved {stats['message_count']} messages from #{channel.name}")
                
            # Save messages to JSON
            with open(channel_dir / "messages.json", 'w', encoding='utf-8') as f:
                json.dump(messages, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error backing up channel {channel.name}: {str(e)}")
            stats['error_count'] += 1
            
        return stats

    async def summarize_messages(
        self,
        channel: TextChannel,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> str:
        """Create a summary of messages in a channel.
        
        Args:
            channel: The channel to summarize
            start_time: Optional start time for filtering messages
            end_time: Optional end time for filtering messages
            
        Returns:
            str: A summary of the messages
        """
        messages = []
        async for message in channel.history(
            limit=None,
            oldest_first=True,
            after=start_time,
            before=end_time
        ):
            if message.content:
                messages.append(f"{message.author.name}: {message.content}")
                
        if not messages:
            return f"No messages found in #{channel.name} for the specified time period."
            
        # TODO: Implement actual summarization logic
        # For now, just return the first few messages
        return "\n".join(messages[:5]) + "\n...\n(Summary functionality to be implemented)"

def main():
    """Main entry point for the Discord manager."""
    load_dotenv()
    
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise ValueError("DISCORD_BOT_TOKEN not found in environment variables")
        
    manager = DiscordManager(token)
    
    async def run():
        try:
            await manager.start_backup(int(os.getenv("DISCORD_SERVER_ID")))
            # Example usage:
            # server_id = int(os.getenv("DISCORD_SERVER_ID"))
            # stats = await manager.backup_server(server_id)
            # print(f"Backup completed: {stats}")
        finally:
            await manager.bot.close()
            
    asyncio.run(run())

if __name__ == "__main__":
    main() 