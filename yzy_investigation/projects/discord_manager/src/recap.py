#!/usr/bin/env python3
"""
Simple Discord recap script:
1. Loads environment variables (Discord bot token, channel IDs, server ID, OpenAI key) from .env
2. Tracks the last time a summary was posted in recap_state.json
3. Fetches all new messages from a source channel since the last summary
4. Summarises them with GPT-4o
5. Posts the summary into a designated recap channel (test or production based on test_mode)
6. Updates the timestamp in recap_state.json

You can run this regularly with a cron job.
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple

import discord
from discord.ext import commands
from dotenv import load_dotenv
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path to our simple JSON file that stores the last summary time
STATE_FILE = "recap_state.json"


class RecapBot(commands.Bot):
    def __init__(self, test_mode: bool = False, **kwargs):
        # Give it a prefix (we won't actually use commands, but needed for Bot init)
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents, **kwargs)

        # Source channel is always from production
        self.source_channel_id = int(os.getenv("DISCORD_SOURCE_CHANNEL_ID", "0"))
        self.server_id = int(os.getenv("DISCORD_SERVER_ID", "0"))
        
        # Target channel and server depend on test_mode
        self.test_mode = test_mode
        if test_mode:
            self.target_server_id = int(os.getenv("DISCORD_SERVER_ID_TEST", "0"))
            self.target_channel_id = int(os.getenv("DISCORD_CHANNEL_ID_TEST", "0"))
        else:
            self.target_server_id = self.server_id
            self.target_channel_id = int(os.getenv("DISCORD_CHANNEL_ID", "0"))

        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.model_name = "gpt-4o"  # As requested
        self.state_file = Path(STATE_FILE)

        # Load or init the state
        self.last_summary_time = self._load_last_summary_time()
    
    def _load_last_summary_time(self) -> datetime:
        """
        Load the last summary time from the state file, or return epoch if missing.
        """
        if not self.state_file.is_file():
            # Default to the beginning of time if no state yet
            return datetime(1970, 1, 1, tzinfo=timezone.utc)
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "last_summary_time" in data:
                # Convert from ISO string
                return datetime.fromisoformat(data["last_summary_time"])
        except Exception:
            pass
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    
    def _save_last_summary_time(self, timestamp: datetime):
        """
        Save the provided timestamp to the state file in ISO format.
        """
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump({"last_summary_time": timestamp.isoformat()}, f, indent=2)
    
    async def on_ready(self):
        logger.info(f"Logged in as {self.user} (id: {self.user.id})")
        
        # Get source guild and channel (always production)
        source_guild = self.get_guild(self.server_id)
        if source_guild is None:
            logger.error(f"Could not find source server with ID {self.server_id}. Exiting.")
            await self.close()
            return
        
        source_channel = source_guild.get_channel(self.source_channel_id)
        if source_channel is None:
            logger.error(f"Could not find source channel with ID {self.source_channel_id}. Exiting.")
            await self.close()
            return

        # Get target guild and channel (test or production based on test_mode)
        target_guild = self.get_guild(self.target_server_id)
        if target_guild is None:
            logger.error(f"Could not find target server with ID {self.target_server_id}. Exiting.")
            await self.close()
            return

        target_channel = target_guild.get_channel(self.target_channel_id)
        if target_channel is None:
            logger.error(f"Could not find target channel with ID {self.target_channel_id}. Exiting.")
            await self.close()
            return

        mode_str = "TEST MODE" if self.test_mode else "PRODUCTION MODE"
        logger.info(f"Running in {mode_str}")
        logger.info(f"Source: #{source_channel.name} in {source_guild.name}")
        logger.info(f"Target: #{target_channel.name} in {target_guild.name}")
        logger.info(f"Fetching new messages since {self.last_summary_time}...")
        
        # Gather messages from source
        new_messages = await self._fetch_new_messages(source_channel, self.last_summary_time)
        
        if not new_messages:
            logger.info("No new messages found. Exiting.")
            # Update the last_summary_time to now so we don't keep picking up older messages
            self._save_last_summary_time(datetime.now(timezone.utc))
            await self.close()
            return
        
        # Summarise the messages
        summary_text = await self._summarise_messages(new_messages)

        # Post the summary to target channel
        if summary_text.strip():
            logger.info(f"Posting summary to #{target_channel.name} ...")
            await self._post_summary(target_channel, summary_text)
        else:
            logger.info("Summary is empty. Nothing to post.")

        # Update the last_summary_time to now
        self._save_last_summary_time(datetime.now(timezone.utc))

        # Done! Close the bot.
        await self.close()

    async def _fetch_new_messages(self, channel: discord.TextChannel, after_time: datetime):
        """
        Fetch all new messages in the channel since after_time, oldest first.
        In test mode, only fetches messages from the last 2 days regardless of after_time.

        Args:
            channel: The Discord channel to fetch messages from
            after_time: The timestamp to fetch messages after

        Returns:
            list[str]: List of formatted messages
        """
        new_messages = []

        # In test mode, limit to last 2 days
        if self.test_mode:
            two_days_ago = datetime.now(timezone.utc) - timedelta(days=2)
            effective_after = max(after_time, two_days_ago)
            logger.info(f"Test mode: Limiting message fetch to last 2 days (after {effective_after})")
        else:
            effective_after = after_time

        # We use oldest_first=True so we get them in ascending chronological order.
        # We'll filter by 'after=after_time' so we only get messages after the last summary.
        async for msg in channel.history(limit=None, oldest_first=True, after=effective_after):
            # Only summarise text content. You can also capture attachments or other logic if needed.
            if msg.content and not msg.author.bot:
                new_messages.append(f"{msg.author.display_name}: {msg.content}")
        return new_messages

    async def _summarise_messages(self, messages):
        """
        Summarise the provided list of text lines using GPT-4o.
        """
        openai.api_key = self.openai_api_key
        logger.info(f"Summarising {len(messages)} messages with model {self.model_name}...")
        
        # We'll just feed them in as a single big block of text for simplicity.
        # In production, consider chunking if it's huge or if there are token limits.
        conversation_text = "\n".join(messages)

        system_prompt = (
            "You are a helpful assistant that summarizes Discord conversations. "
            "Provide a clear, concise summary of the main points and any key actions or decisions. "
            "Keep it factual, with no speculation."
        )

        user_prompt = (
            "Here are the new messages:\n\n"
            f"{conversation_text}\n\n"
            "Summarise these messages. Aim for a short recap covering key points, topics discussed, questions/answers, and action items."
        )

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
            )
            summary = response.choices[0].message.content.strip()
            return summary

        except Exception as e:
            logger.error(f"OpenAI summarisation failed: {e}")
            return "Failed to summarise messages."

    async def _post_summary(self, channel: discord.TextChannel, summary_text: str):
        """
        Post the summary text to the given channel, splitting if necessary.
        """
        # Discord has a limit of ~2000 chars per message for regular bots. 
        # We'll do a simple chunking if it's too big:
        chunks = self._split_into_discord_chunks(summary_text, max_chars=2000)
        for chunk in chunks:
            await channel.send(chunk)

    def _split_into_discord_chunks(self, text: str, max_chars: int = 2000):
        """
        Split text into smaller strings that fit within Discord's message limit.
        """
        lines = text.split("\n")
        chunks = []
        current_chunk = ""
        
        for line in lines:
            if len(current_chunk) + len(line) + 1 > max_chars:
                # Push current chunk and reset
                chunks.append(current_chunk)
                current_chunk = line
            else:
                if current_chunk == "":
                    current_chunk = line
                else:
                    current_chunk += "\n" + line
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks


async def run_recap(test_mode: bool = False):
    """
    Main async entry point. Creates and runs the bot until it finishes.
    
    Args:
        test_mode: If True, summaries will be posted to test server/channel.
                  If False, summaries go to production server/channel.
                  Source messages are always pulled from production.
    """
    bot_token = os.getenv("DISCORD_BOT_TOKEN", "")
    if not bot_token:
        raise ValueError("DISCORD_BOT_TOKEN not found in .env")

    bot = RecapBot(test_mode=test_mode)
    await bot.start(bot_token)


def main():
    # Load environment variables
    load_dotenv()
    
    # Check for test mode flag
    test_mode = os.getenv("TEST_MODE", "").lower() == "true"
    
    # Run the bot
    asyncio.run(run_recap(test_mode=test_mode))


if __name__ == "__main__":
    main()