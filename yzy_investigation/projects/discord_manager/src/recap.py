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
from typing import Optional, Tuple, List, Dict, Any

import discord
from discord.ext import commands
from dotenv import load_dotenv
import aiohttp
import backoff

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path to our simple JSON file that stores the last summary time
STATE_FILE = "recap_state.json"

# Constants
OPENAI_TIMEOUT = 60  # seconds
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
MAX_MESSAGES_PER_CHUNK = 300  # Maximum messages to process in one API call
MAX_CONCURRENT_CHUNKS = 3  # Maximum number of chunks to process in parallel


class RecapBot(commands.Bot):
    def __init__(self, test_mode: bool = False, **kwargs):
        # Give it a prefix (we won't actually use commands, but needed for Bot init)
        intents = discord.Intents.default()
        intents.message_content = True
        
        # Increase the heartbeat timeout
        kwargs['heartbeat_timeout'] = 150.0  # Increase from default 60s
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
        
        # Create aiohttp session
        self.session = None

        # Load or init the state
        self.last_summary_time = self._load_last_summary_time()
        
        # Track token usage and cost
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0

    async def setup_hook(self) -> None:
        """Create aiohttp session when bot starts up."""
        self.session = aiohttp.ClientSession()

    async def close(self) -> None:
        """Clean up aiohttp session when bot shuts down."""
        if self.session:
            await self.session.close()
        await super().close()

    @backoff.on_exception(backoff.expo, 
                         (aiohttp.ClientError, asyncio.TimeoutError),
                         max_tries=3)
    async def _make_openai_request(self, messages: List[Dict[str, str]]) -> Dict:
        """Make an async request to OpenAI API with retry logic.
        
        Args:
            messages: List of message dictionaries for the chat completion
            
        Returns:
            API response as dictionary
            
        Raises:
            Exception: If all retries fail
        """
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        # Define the function schema for JSON response
        functions = [
            {
                "name": "create_summary",
                "description": "Create a summary of Discord messages organized by topics",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topics": {
                            "type": "array",
                            "description": "Array of topic summaries",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "t": {
                                        "type": "string",
                                        "description": "Topic title (max 100 chars)"
                                    },
                                    "d": {
                                        "type": "array",
                                        "description": "Array of 2-5 detail points about this topic",
                                        "items": {"type": "string"},
                                        "minItems": 2,
                                        "maxItems": 5
                                    },
                                    "s": {
                                        "type": "array",
                                        "description": "Array of Discord message URLs",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": ["t", "d", "s"]
                            },
                            "minItems": 2
                        }
                    },
                    "required": ["topics"]
                }
            }
        ]
        
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.3,
            "functions": functions,
            "function_call": {"name": "create_summary"}
        }
        
        timeout = aiohttp.ClientTimeout(total=OPENAI_TIMEOUT)
        async with self.session.post(
            OPENAI_API_URL,
            headers=headers,
            json=data,
            timeout=timeout
        ) as response:
            response.raise_for_status()
            return await response.json()

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

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost in USD for API usage based on token counts.
        
        Args:
            model: The model name being used
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            
        Returns:
            Total cost in USD
        """
        # Base prices in USD per token
        if model == "gpt-4o":
            prompt_cost = 2.50 / 1_000_000
            completion_cost = 10.00 / 1_000_000
        elif model == "gpt-4o-mini":
            prompt_cost = 0.150 / 1_000_000
            completion_cost = 0.600 / 1_000_000
        else:
            return 0.0
            
        # Calculate total cost
        total_cost = (prompt_tokens * prompt_cost) + (completion_tokens * completion_cost)
        return round(total_cost, 4)  # Round to 4 decimal places

    def _format_message(self, message: discord.Message) -> str:
        """Format a single message for summarization.
        
        Args:
            message: Discord message object
            
        Returns:
            Formatted message string
        """
        timestamp = message.created_at.strftime("%H:%M")
        author = message.author.display_name
        content = message.content
        
        # Handle attachments
        attachments = ""
        if message.attachments:
            attachments = " [Attachments: " + ", ".join(a.filename for a in message.attachments) + "]"
            
        # Handle embeds
        embeds = ""
        if message.embeds:
            embed_types = []
            for embed in message.embeds:
                if embed.type == 'rich':
                    if embed.title:
                        embed_types.append(f"Embed: {embed.title}")
                    else:
                        embed_types.append("Rich Embed")
                else:
                    embed_types.append(embed.type)
            embeds = " [Embeds: " + ", ".join(embed_types) + "]"
            
        # Add message ID as a hidden reference
        return f"[{timestamp}] {author}: {content}{attachments}{embeds} <!-- msg:{message.id} -->"
    
    def _split_messages_into_chunks(self, messages: List[discord.Message]) -> List[List[discord.Message]]:
        """Split messages into smaller chunks for processing.
        
        Args:
            messages: List of Discord messages
            
        Returns:
            List of message chunks
        """
        chunks = []
        for i in range(0, len(messages), MAX_MESSAGES_PER_CHUNK):
            chunk = messages[i:i + MAX_MESSAGES_PER_CHUNK]
            chunks.append(chunk)
        return chunks

    async def _summarise_chunk(self, chunk: List[discord.Message], chunk_number: int, total_chunks: int) -> str:
        """Summarise a single chunk of messages.
        
        Args:
            chunk: List of messages in the chunk
            chunk_number: Current chunk number
            total_chunks: Total number of chunks
            
        Returns:
            Summary in JSON format
        """
        logger.info(f"Summarising chunk {chunk_number}/{total_chunks} ({len(chunk)} messages)")
        
        # Format messages
        formatted_messages = [self._format_message(msg) for msg in chunk]
        conversation_text = "\n".join(formatted_messages)

        system_prompt = """You are a specialized JSON summarizer that creates concise, structured summaries of Discord conversations.
        You MUST create MULTIPLE topics for each chunk of messages - at least 2-3 distinct topics per chunk.
        
        You will be called with a function that requires:
        - 't': A brief title for the discussion (max 100 chars)
        - 'd': An array of 2-5 bullet points about this topic
        - 's': An array of message URLs for relevant messages
        
        You are summarizing part {chunk_number} of {total_chunks} from the conversation.
        
        CRITICAL REQUIREMENTS:
        1. You MUST return at least 2 distinct topics, even if the messages seem related
        2. You MUST split discussions into separate topics based on:
           - Different subjects being discussed
           - Different aspects of the same subject
           - Different time periods or phases of discussion
           - Different participants or viewpoints
        3. You MUST keep topics focused and concise
        4. You MUST avoid merging unrelated discussions
        5. You MUST include relevant message links
        
        Focus on identifying:
        - Main topics discussed
        - Important announcements or updates
        - Questions asked and their answers
        - Key decisions or action items
        - Significant debates or disagreements
        - Technical discussions
        - Community interactions
        
        IMPORTANT:
        - Return MULTIPLE distinct topics (MINIMUM 2 topics per chunk)
        - Include all important ideas
        - Avoid speculation or adding outside knowledge
        - Keep it factual and based only on what is stated
        - Include the author and usernames and entities of individuals involved or spoken about
        - For significant messages, include their message links
        """

        user_prompt = f"""Here are the messages to summarize (part {chunk_number} of {total_chunks}):
        ----------------
        {conversation_text}
        ----------------

        Create a summary with MULTIPLE distinct topics (at least 2-3).
        Extract message IDs from the <!-- msg:ID --> comments in the text.
        Use this format for message links:
        https://discord.com/channels/{self.server_id}/{self.source_channel_id}/MESSAGE_ID

        Remember: You MUST return at least 2 distinct topics, even if they are different aspects of the same discussion.
        """

        try:
            api_messages = [
                {"role": "system", "content": system_prompt.format(chunk_number=chunk_number, total_chunks=total_chunks)},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await self._make_openai_request(api_messages)
            
            # Track token usage
            self.total_prompt_tokens += response['usage']['prompt_tokens']
            self.total_completion_tokens += response['usage']['completion_tokens']
            
            # Calculate and track cost
            cost = self._calculate_cost(
                self.model_name,
                response['usage']['prompt_tokens'],
                response['usage']['completion_tokens']
            )
            self.total_cost += cost
            
            logger.info(f"Chunk {chunk_number} token usage - Prompt: {response['usage']['prompt_tokens']}, Completion: {response['usage']['completion_tokens']}")
            logger.info(f"Chunk {chunk_number} cost: ${cost:.4f}")

            # Extract the function call arguments
            function_call = response['choices'][0]['message']['function_call']
            if function_call['name'] != 'create_summary':
                raise ValueError(f"Unexpected function call: {function_call['name']}")
                
            # Parse the function arguments
            args = json.loads(function_call['arguments'])
            topics = args['topics']
            
            # Convert short field names back to full names for consistency
            full_topics = []
            for topic in topics:
                full_topics.append({
                    "topic": topic['t'],
                    "details": topic['d'],
                    "sources": topic['s']
                })
            
            logger.info(f"Chunk {chunk_number}: Generated {len(topics)} topics")
            return json.dumps(full_topics, indent=2)

        except Exception as e:
            logger.error(f"Failed to summarize chunk {chunk_number}: {e}")
            return json.dumps([{
                "topic": f"Error Summarizing Chunk {chunk_number}",
                "details": ["Failed to generate summary due to an error."],
                "sources": []
            }])

    async def _merge_summaries(self, chunk_summaries: List[str]) -> str:
        """Merge multiple chunk summaries into a cohesive final summary.
        
        Args:
            chunk_summaries: List of JSON summaries from chunks
            
        Returns:
            Final merged summary in JSON format
        """
        logger.info(f"Merging {len(chunk_summaries)} chunk summaries...")
        
        # First, parse all summaries and collect topics
        all_topics = []
        for i, summary in enumerate(chunk_summaries, 1):
            try:
                data = json.loads(summary)
                if isinstance(data, dict):
                    data = [data]
                all_topics.extend(data)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse chunk {i} summary")
                continue

        # If we have very few topics, no need to merge
        if len(all_topics) <= 5:
            return json.dumps(all_topics, indent=2)

        # Create a prompt to merge the topics
        topics_json = json.dumps(all_topics, indent=2)
        
        system_prompt = """You are merging multiple topic summaries into a cohesive final summary.
        Your task is to:
        1. Combine related topics
        2. Organize information chronologically where relevant
        3. Remove redundant information
        4. Ensure all important points are preserved
        5. Maintain all relevant source links
        
        The output must be a valid JSON array with multiple topic objects.
        Each topic must have:
        - 'topic': A clear, descriptive title
        - 'details': An array of bullet points
        - 'sources': An array of message URLs
        """

        user_prompt = f"""Here are the topics to merge:
        ----------------
        {topics_json}
        ----------------

        Create a final JSON array that combines related topics and organizes the information clearly.
        - Combine topics that discuss the same subject
        - Keep the topics distinct and focused
        - Preserve all important information
        - Maintain all source links
        - Aim for 5-10 well-organized topics
        """

        try:
            api_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await self._make_openai_request(api_messages)
            
            # Track token usage
            self.total_prompt_tokens += response['usage']['prompt_tokens']
            self.total_completion_tokens += response['usage']['completion_tokens']
            
            # Calculate and track cost
            cost = self._calculate_cost(
                self.model_name,
                response['usage']['prompt_tokens'],
                response['usage']['completion_tokens']
            )
            self.total_cost += cost
            
            logger.info(f"Merge token usage - Prompt: {response['usage']['prompt_tokens']}, Completion: {response['usage']['completion_tokens']}")
            logger.info(f"Merge cost: ${cost:.4f}")

            content = response['choices'][0]['message']['content'].strip()
            
            # Validate JSON and ensure it's an array
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    logger.warning("Merge: Received single object instead of array. Converting to array.")
                    data = [data]
                    content = json.dumps(data, indent=2)
                
                logger.info(f"Final summary contains {len(data)} topics")
            except json.JSONDecodeError:
                logger.error("Merge: Invalid JSON received. Using raw content.")
            
            return content

        except Exception as e:
            logger.error(f"Failed to merge summaries: {e}")
            # If merge fails, return all topics as-is
            return json.dumps(all_topics, indent=2)

    async def _summarise_messages(self, messages: List[discord.Message]) -> str:
        """
        Summarise the provided messages using GPT-4o.
        
        Args:
            messages: List of Discord messages
            
        Returns:
            Summary text in JSON format
        """
        if not messages:
            return json.dumps([{
                "topic": "No Messages",
                "details": ["No new messages to summarize."],
                "sources": []
            }])

        logger.info(f"Splitting {len(messages)} messages into chunks...")
        chunks = self._split_messages_into_chunks(messages)
        logger.info(f"Split into {len(chunks)} chunks")

        # Process chunks with concurrency limit
        chunk_summaries = []
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHUNKS)
        
        async def process_chunk(chunk: List[discord.Message], chunk_number: int) -> str:
            async with semaphore:
                return await self._summarise_chunk(chunk, chunk_number, len(chunks))

        # Create tasks for all chunks
        tasks = [
            process_chunk(chunk, i+1)
            for i, chunk in enumerate(chunks)
        ]
        
        # Wait for all chunks to be processed
        chunk_summaries = await asyncio.gather(*tasks)
        
        # If only one chunk, no need to merge
        if len(chunks) == 1:
            return chunk_summaries[0]
            
        # Merge all chunk summaries into a final summary
        return await self._merge_summaries(chunk_summaries)

    async def _post_summary(self, channel: discord.TextChannel, summary_text: str):
        """
        Post the summary text to the given channel, splitting if necessary.
        
        Args:
            channel: The Discord channel to post to
            summary_text: The summary text in JSON format
        """
        try:
            # Parse the JSON summary
            summary_data = json.loads(summary_text)
            
            # Create a formatted message for each topic
            for topic in summary_data:
                embed = discord.Embed(
                    title=topic["topic"],
                    color=discord.Color.blue()
                )
                
                # Add details as bullet points
                details = "\n".join(f"• {detail}" for detail in topic["details"])
                if details:
                    embed.add_field(name="Details", value=details, inline=False)
                
                # Add sources as links if any
                if topic["sources"]:
                    sources = "\n".join(f"[Link]({source})" for source in topic["sources"])
                    embed.add_field(name="Sources", value=sources, inline=False)
                
                await channel.send(embed=embed)
                
        except json.JSONDecodeError:
            # Fallback to simple text chunks if JSON parsing fails
            chunks = self._split_into_discord_chunks(summary_text, max_chars=2000)
            for chunk in chunks:
                await channel.send(chunk)

    def _split_into_discord_chunks(self, text: str, max_chars: int = 2000) -> List[str]:
        """Split text into smaller strings that fit within Discord's message limit."""
        lines = text.split("\n")
        chunks = []
        current_chunk = ""
        
        for line in lines:
            if len(current_chunk) + len(line) + 1 > max_chars:
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

        # In test mode, always use last 24 hours
        if self.test_mode:
            effective_after = datetime.now(timezone.utc) - timedelta(days=1)
            logger.info(f"Test mode: Using last 24 hours (after {effective_after})")
        else:
            effective_after = self.last_summary_time
            logger.info(f"Production mode: Fetching new messages since {effective_after}")

        # Fetch new messages
        new_messages = []
        async for msg in source_channel.history(limit=None, oldest_first=True, after=effective_after):
            if msg.content and not msg.author.bot:
                new_messages.append(msg)
        
        if not new_messages:
            logger.info("No new messages found. Exiting.")
            # Update the last_summary_time to now so we don't keep picking up older messages
            # Only update in production mode
            if not self.test_mode:
                self._save_last_summary_time(datetime.now(timezone.utc))
            await self.close()
            return
        
        # Summarise the messages
        summary_text = await self._summarise_messages(new_messages)

        # Post the summary to target channel
        if summary_text.strip():
            logger.info(f"Posting summary to #{target_channel.name} ...")
            await self._post_summary(target_channel, summary_text)
            
            # Log final stats
            logger.info("\nFinal Statistics:")
            logger.info(f"Total messages processed: {len(new_messages)}")
            logger.info(f"Total prompt tokens: {self.total_prompt_tokens}")
            logger.info(f"Total completion tokens: {self.total_completion_tokens}")
            logger.info(f"Total cost: ${self.total_cost:.4f}")
        else:
            logger.info("Summary is empty. Nothing to post.")

        # Update the last_summary_time to now
        # Only update in production mode
        if not self.test_mode:
            self._save_last_summary_time(datetime.now(timezone.utc))

        # Done! Close the bot.
        await self.close()


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