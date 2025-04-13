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
import argparse
import re

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
STATE_FILE = Path(__file__).parent.parent / "data" / "recap_state.json"

# Constants
OPENAI_TIMEOUT = 60  # seconds
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
MAX_MESSAGES_PER_CHUNK = 300  # Maximum messages to process in one API call
MAX_CONCURRENT_CHUNKS = 3  # Maximum number of chunks to process in parallel


class RecapBot(commands.Bot):
    def __init__(self, test_mode: bool = False, start_time: Optional[datetime] = None, checkpoint_dir: Optional[str] = None, load_checkpoint: Optional[str] = None, **kwargs):
        # Give it a prefix (we won't actually use commands, but needed for Bot init)
        intents = discord.Intents.default()
        intents.message_content = True
        
        # Increase the heartbeat timeout
        kwargs['heartbeat_timeout'] = 150.0  # Increase from default 60s
        super().__init__(command_prefix="!", intents=intents, **kwargs)

        # Store the custom start time if provided
        self.custom_start_time = start_time
        
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
        self.model_name = "gpt-4o-mini"  # As requested
        self.state_file = Path(STATE_FILE)
        
        # Create aiohttp session
        self.session = None

        # Load or init the state
        self.last_summary_time = self._load_last_summary_time()
        
        # Track token usage and cost
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0

        # Add checkpoint directory
        self.base_checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path(__file__).parent.parent / "data" / "checkpoints"
        self.base_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate run ID and create run-specific checkpoint directory
        self.current_run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        # If loading from checkpoint, extract run ID from the checkpoint path
        if load_checkpoint:
            checkpoint_path = Path(load_checkpoint)
            if checkpoint_path.exists():
                # Try to extract run ID from checkpoint path
                try:
                    self.current_run_id = checkpoint_path.parent.name
                except Exception:
                    pass  # Keep the new run ID if we can't extract it
        
        # Create run-specific checkpoint directory
        self.checkpoint_dir = self.base_checkpoint_dir / self.current_run_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Store the load_checkpoint path
        self.load_checkpoint = load_checkpoint
        
        # Track processing state
        self.processed_chunks: Dict[str, Any] = {}

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
        """Make an async request to OpenAI API with retry logic."""
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        # Define the function schema for JSON response - using short field names
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
                                    },
                                    "ts": {
                                        "type": "array",
                                        "description": "Array containing [earliest_timestamp, latest_timestamp] for messages in this topic",
                                        "items": {"type": "string"},
                                        "minItems": 2,
                                        "maxItems": 2
                                    }
                                },
                                "required": ["t", "d", "s", "ts"]
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
            "function_call": {
                "name": "create_summary",
                "arguments": "{\"topics\": []}"  # Provide default arguments
            }
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

    def _validate_and_transform_topic(self, topic: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and transform a topic to ensure it matches our expected format."""
        # Check if we got the new format (full names) or old format (single letters)
        if "topic" in topic:
            return {
                "t": topic["topic"],
                "d": topic["details"],
                "s": topic["sources"],
                "ts": topic["timestamps"]
            }
        elif "t" in topic:
            return topic
        else:
            raise ValueError(f"Invalid topic format: {topic}")

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
        """Format a single message for summarization."""
        # Discord.py uses created_at for the timestamp
        timestamp_dt = message.created_at
        timestamp = timestamp_dt.strftime("%Y-%m-%d %H:%M")
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
            
        # Add message ID and timestamp as hidden references
        ts_str = timestamp_dt.isoformat()
        return f"[{timestamp}] {author}: {content}{attachments}{embeds} <!-- msg:{message.id} ts:{ts_str} -->"
    
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
        """Summarise a single chunk of messages."""
        chunk_id = f"chunk_{chunk_number}"  # Simplified since we're in a run-specific directory
        checkpoint_file = self.checkpoint_dir / f"{chunk_id}.json"

        # Check if we have a saved checkpoint for this chunk
        if checkpoint_file.exists():
            logger.info(f"Loading checkpoint for chunk {chunk_number} from {checkpoint_file}")
            with open(checkpoint_file, 'r') as f:
                return f.read()

        logger.info(f"Summarising chunk {chunk_number}/{total_chunks} ({len(chunk)} messages)")
        
        # Format messages
        formatted_messages = [self._format_message(msg) for msg in chunk]
        conversation_text = "\n".join(formatted_messages)

        system_prompt = f"""You are a specialized JSON summarizer that creates concise, structured summaries of Discord conversations.
        You MUST create MULTIPLE topics for each chunk of messages - at least 2-3 distinct topics per chunk.
        
        For each topic, you MUST include:
        1. 't': Topic title (string, max 100 chars)
        2. 'd': Array of 2-5 detail points about this topic. Each detail MUST:
           - Be specific and concrete (e.g., "User X suggested implementing Y feature" instead of "Discussion about features")
           - Include relevant names, numbers, and direct quotes where appropriate
           - Avoid vague phrases like "discussion about" or "talked about"
           - Focus on the actual content and decisions/conclusions
           - Capture unique information not covered in other points
        3. 's': Array of relevant message URLs
        4. 'ts': Array of [earliest_timestamp, latest_timestamp] from the topic's messages
        
        CRITICAL: Extract timestamps from the <!-- ts:TIME --> comments in messages.
        For each topic:
        - Find the earliest and latest timestamps from messages in that topic
        - Include them in the 'ts' array: [earliest_iso_time, latest_iso_time]
        - Ensure timestamps are in ISO format
        
        CRITICAL: For message URLs:
        - Extract message IDs from the <!-- msg:ID --> comments
        - Use this exact format: https://discord.com/channels/{self.server_id}/{self.source_channel_id}/MESSAGE_ID
        - Include at least one source URL per topic
        - Example URL: https://discord.com/channels/{self.server_id}/{self.source_channel_id}/1234567890
        
        Example output format with SPECIFIC details:
        {{
          "topics": [
            {{
              "t": "New Website Launch Plans",
              "d": [
                "Airy revealed plans to launch yzy-lore.com by April 20th with comprehensive timeline features",
                "Website will include an interactive map showing 15+ key locations from the ARG",
                "Community voted 12-3 in favor of dark mode as the default theme",
                "Sarah and John volunteered to help with beta testing next week"
              ],
              "s": [
                "https://discord.com/channels/{self.server_id}/{self.source_channel_id}/1234567890"
              ],
              "ts": [
                "2025-04-09T18:02:36.177000+00:00",
                "2025-04-09T18:08:34.206000+00:00"
              ]
            }}
          ]
        }}
        
        You MUST follow this exact format and use the short field names (t, d, s, ts).
        Message IDs must be extracted from <!-- msg:ID --> comments and used in the URLs.
        """

        try:
            api_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation_text}
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
            if 'topics' not in args:
                raise ValueError("No topics found in response")
                
            # Validate and transform each topic
            topics = []
            for topic in args['topics']:
                try:
                    transformed_topic = self._validate_and_transform_topic(topic)
                    topics.append(transformed_topic)
                except Exception as e:
                    logger.error(f"Failed to validate topic: {e}")
                    continue
            
            if not topics:
                raise ValueError("No valid topics found in response")
            
            logger.info(f"Chunk {chunk_number}: Generated {len(topics)} topics")
            summary_text = json.dumps({"topics": topics}, indent=2)
            
            # Save checkpoint
            try:
                with open(checkpoint_file, 'w') as f:
                    f.write(summary_text)
                logger.info(f"Saved checkpoint for chunk {chunk_number}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint for chunk {chunk_number}: {e}")

            return summary_text

        except Exception as e:
            logger.error(f"Failed to summarize chunk {chunk_number}: {e}")
            return json.dumps({
                "topics": [{
                    "t": f"Error Summarizing Chunk {chunk_number}",
                    "d": ["Failed to generate summary due to an error."],
                    "s": [],
                    "ts": [datetime.now(timezone.utc).isoformat()] * 2
                }]
            })

    async def _merge_summaries(self, chunk_summaries: List[str]) -> str:
        """Merge multiple chunk summaries into a cohesive final summary."""
        merge_checkpoint_file = self.checkpoint_dir / "merge.json"
        
        # If loading from a specific checkpoint
        if self.load_checkpoint:
            checkpoint_path = Path(self.load_checkpoint)
            if checkpoint_path.exists():
                logger.info(f"Loading from checkpoint: {self.load_checkpoint}")
                with open(checkpoint_path, 'r') as f:
                    return f.read()
            else:
                logger.error(f"Checkpoint file not found: {self.load_checkpoint}")

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

        # For large numbers of topics, do recursive merging
        MAX_TOPICS_PER_MERGE = 20  # Maximum number of topics to merge in one API call
        
        async def merge_topic_batch(topics: List[dict]) -> List[dict]:
            topics_json = json.dumps(topics, indent=2)
            
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
                
                logger.info(f"Merge batch token usage - Prompt: {response['usage']['prompt_tokens']}, Completion: {response['usage']['completion_tokens']}")
                logger.info(f"Merge batch cost: ${cost:.4f}")

                # Check if we got a function call response or direct content
                if 'function_call' in response['choices'][0]['message']:
                    content = response['choices'][0]['message']['function_call']['arguments']
                else:
                    content = response['choices'][0]['message'].get('content', '[]')
                    
                content = content.strip()
                
                # Parse the merged topics
                merged = json.loads(content)
                if isinstance(merged, dict):
                    if 'topics' in merged:
                        merged = merged['topics']
                    else:
                        merged = [merged]
                
                return merged

            except Exception as e:
                logger.error(f"Failed to merge topic batch: {e}")
                return topics  # Return original topics if merge fails

        # Recursive merging function
        async def recursive_merge(topics: List[dict]) -> List[dict]:
            if len(topics) <= MAX_TOPICS_PER_MERGE:
                return await merge_topic_batch(topics)
            
            # Split into batches and merge recursively
            batches = [topics[i:i + MAX_TOPICS_PER_MERGE] 
                      for i in range(0, len(topics), MAX_TOPICS_PER_MERGE)]
            
            merged_batches = []
            for batch in batches:
                merged = await merge_topic_batch(batch)
                merged_batches.extend(merged)
            
            # If we still have too many topics, merge again
            if len(merged_batches) > MAX_TOPICS_PER_MERGE:
                return await recursive_merge(merged_batches)
            
            return merged_batches

        try:
            # Perform recursive merging
            final_topics = await recursive_merge(all_topics)
            logger.info(f"Final summary contains {len(final_topics)} topics")
            final_json = json.dumps(final_topics, indent=2)

            # Save checkpoint
            try:
                with open(merge_checkpoint_file, 'w') as f:
                    f.write(final_json)
                logger.info(f"Saved merge checkpoint to {merge_checkpoint_file}")
                return final_json
            except Exception as e:
                logger.error(f"Failed to save merge checkpoint: {e}")
                return json.dumps(final_topics, indent=2)
            
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
                "t": "No Messages",
                "d": ["No new messages to summarize."],
                "s": []
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

    def _format_date_range(self, start_time: str, end_time: str) -> str:
        """Format a date range for display in Discord.
        
        Args:
            start_time: ISO format start time
            end_time: ISO format end time
            
        Returns:
            Formatted date range string
        """
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(end_time)
        
        # If same day
        if start_dt.date() == end_dt.date():
            # Format: "April 13, 2025 (12:12 AM - 04:56 AM UTC)"
            date_str = start_dt.strftime("%B %d, %Y")
            start_time_str = start_dt.strftime("%I:%M %p").lstrip('0')  # Remove leading 0
            end_time_str = end_dt.strftime("%I:%M %p").lstrip('0')  # Remove leading 0
            return f"{date_str} ({start_time_str} - {end_time_str} UTC)"
        
        # If within 24 hours but crosses midnight
        elif (end_dt - start_dt).total_seconds() <= 86400:  # 24 hours in seconds
            # Format: "April 13-14, 2025 (11:45 PM - 12:30 AM UTC)"
            if start_dt.year == end_dt.year:
                if start_dt.month == end_dt.month:
                    date_str = f"{start_dt.strftime('%B')} {start_dt.day}-{end_dt.day}, {start_dt.year}"
                else:
                    date_str = f"{start_dt.strftime('%B %d')} - {end_dt.strftime('%B %d')}, {start_dt.year}"
            else:
                date_str = f"{start_dt.strftime('%B %d, %Y')} - {end_dt.strftime('%B %d, %Y')}"
            
            start_time_str = start_dt.strftime("%I:%M %p").lstrip('0')
            end_time_str = end_dt.strftime("%I:%M %p").lstrip('0')
            return f"{date_str} ({start_time_str} - {end_time_str} UTC)"
        
        # If same month
        elif start_dt.year == end_dt.year and start_dt.month == end_dt.month:
            # Format: "April 13-15, 2025"
            return f"{start_dt.strftime('%B')} {start_dt.day}-{end_dt.day}, {start_dt.year}"
        
        # If same year
        elif start_dt.year == end_dt.year:
            # Format: "April 13 - May 1, 2025"
            return f"{start_dt.strftime('%B %d')} - {end_dt.strftime('%B %d')}, {start_dt.year}"
        
        # Different years
        else:
            # Format: "December 31, 2024 - January 1, 2025"
            return f"{start_dt.strftime('%B %d, %Y')} - {end_dt.strftime('%B %d, %Y')}"

    async def _post_summary(self, channel: discord.TextChannel, summary_text: str):
        """Post the summary text to the given channel, splitting if necessary."""
        try:
            # Log the raw summary for debugging
            logger.debug(f"Raw summary text: {summary_text}")
            
            # Parse the JSON summary and handle nested structure
            try:
                raw_data = json.loads(summary_text)
                logger.debug(f"Parsed JSON data: {json.dumps(raw_data, indent=2)}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse summary JSON: {e}")
                logger.info("Falling back to text chunks...")
                chunks = self._split_into_discord_chunks(summary_text, max_chars=2000)
                for chunk in chunks:
                    await channel.send(chunk)
                return
            
            # Extract topics from nested structure if necessary
            try:
                if isinstance(raw_data, list) and len(raw_data) == 1 and 'topics' in raw_data[0]:
                    summary_data = raw_data[0]['topics']
                else:
                    summary_data = raw_data
                    
                if not isinstance(summary_data, list):
                    raise ValueError(f"Summary data is not a list of topics. Got: {type(summary_data)}")
                
                logger.debug(f"Processing {len(summary_data)} topics")
            except Exception as e:
                logger.error(f"Failed to process summary data structure: {e}")
                return
            
            # Format the time range description
            start_time = self.custom_start_time or (
                datetime.now(timezone.utc) - timedelta(days=1) if self.test_mode
                else self.last_summary_time
            )
            
            # Format the start time in a readable way
            if start_time.date() == datetime.now(timezone.utc).date():
                time_str = start_time.strftime("%I:%M %p UTC")
                time_range = f"today since {time_str}"
            else:
                time_str = start_time.strftime("%B %d, %Y at %I:%M %p UTC")
                time_range = f"since {time_str}"
            
            # Create a header embed
            header_embed = discord.Embed(
                title="📋 Recap",
                description=f"A summary of {len(summary_data)} discussion topics from {time_range}:",
                color=discord.Color.blue(),
                timestamp=datetime.now(timezone.utc)
            )
            await channel.send(embed=header_embed)
            
            # Create formatted message for each topic
            for i, topic in enumerate(summary_data, 1):
                try:
                    # Validate topic structure
                    required_fields = ['t', 'd', 's']
                    missing_fields = [field for field in required_fields if field not in topic]
                    if missing_fields:
                        logger.error(f"Topic {i} missing required fields: {missing_fields}")
                        logger.debug(f"Topic {i} content: {json.dumps(topic, indent=2)}")
                        continue
                    
                    # Choose emoji and color based on topic type
                    topic_type, color = self._get_topic_theme(topic["t"], topic.get("d", []))
                    
                    # Create embed for this topic
                    embed = discord.Embed(
                        title=f"{topic_type} {topic['t']}",
                        color=color
                    )
                    
                    # Add date range if available
                    if "ts" in topic and isinstance(topic["ts"], list) and len(topic["ts"]) == 2:
                        try:
                            date_range = self._format_date_range(topic["ts"][0], topic["ts"][1])
                            embed.add_field(
                                name=date_range,
                                value="",  # Empty value since we're putting everything in the name
                                inline=False
                            )
                        except Exception as e:
                            logger.error(f"Failed to format date range for topic {i}: {e}")
                    
                    # Add details as bullet points with consistent formatting
                    if topic["d"]:
                        details_text = "\n".join(f"• {detail}" for detail in topic["d"])
                        embed.add_field(name="Details", value=details_text, inline=False)
                    
                    # Add sources as a clean list of links
                    if topic["s"]:
                        source_links = []
                        for j, source in enumerate(topic["s"], 1):
                            source_links.append(f"[Message {j}]({source})")
                        
                        sources_text = " • ".join(source_links)
                        if len(sources_text) > 1024:  # Discord's field value limit
                            sources_text = " • ".join(source_links[:3]) + " • ..."
                        
                        embed.add_field(
                            name="🔗 Sources",
                            value=sources_text,
                            inline=False
                        )
                    
                    # Add footer showing correct topic number
                    embed.set_footer(text=f"Topic {i} of {len(summary_data)}")
                    
                    await channel.send(embed=embed)
                    
                except Exception as e:
                    logger.error(f"Failed to process topic {i}: {e}")
                    logger.debug(f"Problematic topic content: {json.dumps(topic, indent=2)}")
                    continue
                
        except Exception as e:
            logger.error(f"Failed to post summary: {e}")
            # Try to post raw text as fallback
            try:
                chunks = self._split_into_discord_chunks(summary_text, max_chars=2000)
                logger.info("Attempting to post as raw text...")
                for chunk in chunks:
                    await channel.send(chunk)
            except Exception as fallback_e:
                logger.error(f"Failed to post even as raw text: {fallback_e}")

    def _get_topic_theme(self, topic_title: str, topic_details: List[str]) -> Tuple[str, discord.Color]:
        """
        Get the appropriate emoji and color for a topic based on its content.
        Uses a scoring system to analyze both the title and details.
        
        Args:
            topic_title: The topic title to analyze
            topic_details: List of detail points about the topic
            
        Returns:
            Tuple of (emoji, discord.Color)
        """
        # Convert inputs to lowercase for matching
        title = topic_title.lower()
        details = [d.lower() for d in topic_details]
        
        # Define theme patterns with weights
        themes = [
            # Investigations and Breakthroughs
            {
                "keywords": ["investigation", "clue", "lore", "breakthrough", "discovery", 
                           "reveal", "finding", "evidence", "solved", "mystery", "hidden", "secret",
                           "uncovered", "discovered", "found", "identified"],
                "related_words": ["research", "analysis", "examine", "study", "explore", "search",
                                "investigate", "track", "trace", "follow", "lead", "connection"],
                "emoji": "🔍",
                "color": discord.Color.from_rgb(147, 112, 219)  # Medium Purple
            },
            # Discussions and Speculation
            {
                "keywords": ["discuss", "debate", "conversation", "speculation", "theory",
                           "suggest", "propose", "opinion", "perspective", "viewpoint"],
                "related_words": ["talk", "chat", "share", "exchange", "argue", "consider",
                                "think", "believe", "feel", "seem", "might", "could", "maybe"],
                "emoji": "💭",
                "color": discord.Color.blue()
            },
            # Announcements and Updates
            {
                "keywords": ["announce", "update", "news", "release", "launch", "introduce",
                           "reveal", "publish", "start", "begin", "official"],
                "related_words": ["new", "coming", "soon", "available", "released", "live",
                                "ready", "launching", "starting", "beginning", "announced"],
                "emoji": "📢",
                "color": discord.Color.green()
            },
            # Warnings and Issues
            {
                "keywords": ["warn", "issue", "problem", "error", "bug", "scam", "concern",
                           "risk", "danger", "threat", "vulnerability", "exploit"],
                "related_words": ["careful", "attention", "caution", "alert", "notice", "important",
                                "critical", "serious", "urgent", "warning", "dangerous"],
                "emoji": "⚠️",
                "color": discord.Color.red()
            },
            # Questions and Help
            {
                "keywords": ["question", "help", "support", "how to", "guide", "tutorial",
                           "explain", "clarify", "assist", "aid"],
                "related_words": ["need", "want", "looking for", "seeking", "trying to",
                                "can someone", "please help", "anyone know", "how do I"],
                "emoji": "❓",
                "color": discord.Color.gold()
            },
            # Community and Social
            {
                "keywords": ["community", "social", "member", "group", "team", "people",
                           "user", "follower", "supporter", "contributor"],
                "related_words": ["together", "everyone", "anybody", "someone", "join",
                                "welcome", "thanks", "appreciate", "grateful", "collaboration"],
                "emoji": "👥",
                "color": discord.Color.purple()
            },
            # Technical and Development
            {
                "keywords": ["technical", "dev", "code", "implementation", "feature", "bug",
                           "fix", "patch", "update", "version", "release", "api"],
                "related_words": ["develop", "build", "create", "implement", "deploy",
                                "test", "debug", "optimize", "improve", "enhance"],
                "emoji": "⚙️",
                "color": discord.Color.dark_grey()
            },
            # Investment and Trading
            {
                "keywords": ["invest", "trade", "price", "market", "coin", "token", "buy",
                           "sell", "exchange", "value", "worth", "cost"],
                "related_words": ["money", "financial", "economic", "profit", "loss",
                                "gain", "decrease", "increase", "change", "volatile"],
                "emoji": "📈",
                "color": discord.Color.green()
            },
            # Events and Activities
            {
                "keywords": ["event", "activity", "meeting", "space", "twitter", "stream",
                           "live", "session", "gathering", "meetup"],
                "related_words": ["schedule", "plan", "organize", "attend", "join", "participate",
                                "watch", "listen", "follow", "upcoming", "soon"],
                "emoji": "📅",
                "color": discord.Color.blue()
            },
            # Humor and Fun
            {
                "keywords": ["joke", "humor", "fun", "banter", "laugh", "meme", "funny",
                           "entertainment", "amusing", "hilarious"],
                "related_words": ["lol", "haha", "lmao", "rofl", "joy", "happy", "excited",
                                "amazing", "awesome", "cool", "nice", "great"],
                "emoji": "😄",
                "color": discord.Color.gold()
            }
        ]
        
        # Scoring weights
        TITLE_KEYWORD_WEIGHT = 3.0
        TITLE_RELATED_WEIGHT = 1.5
        DETAILS_KEYWORD_WEIGHT = 2.0
        DETAILS_RELATED_WEIGHT = 1.0
        
        # Calculate scores for each theme
        theme_scores = []
        for theme in themes:
            score = 0.0
            
            # Score title matches
            for keyword in theme["keywords"]:
                if keyword in title:
                    score += TITLE_KEYWORD_WEIGHT
            for related in theme["related_words"]:
                if related in title:
                    score += TITLE_RELATED_WEIGHT
            
            # Score details matches
            for detail in details:
                for keyword in theme["keywords"]:
                    if keyword in detail:
                        score += DETAILS_KEYWORD_WEIGHT
                for related in theme["related_words"]:
                    if related in detail:
                        score += DETAILS_RELATED_WEIGHT
            
            theme_scores.append((score, theme))
        
        # Sort by score in descending order
        theme_scores.sort(reverse=True, key=lambda x: x[0])
        
        # If we have a clear winner (score > 0), use it
        if theme_scores[0][0] > 0:
            return theme_scores[0][1]["emoji"], theme_scores[0][1]["color"]
        
        # Default theme if no matches found
        return "📜", discord.Color.blue()

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

    async def process_messages(self, messages: List[discord.Message], target_channel: discord.TextChannel) -> None:
        """New method to handle message processing with better error handling."""
        if not messages:
            logger.info("No messages to process")
            return

        try:
            # Split and process chunks
            chunks = self._split_messages_into_chunks(messages)
            chunk_summaries = []

            for i, chunk in enumerate(chunks, 1):
                try:
                    summary = await self._summarise_chunk(chunk, i, len(chunks))
                    chunk_summaries.append(summary)
                    logger.info(f"Successfully processed chunk {i}/{len(chunks)}")
                except Exception as e:
                    logger.error(f"Failed to process chunk {i}: {e}")
                    # Continue with other chunks instead of failing completely

            if not chunk_summaries:
                logger.error("No chunks were successfully processed")
                return

            # Merge summaries
            try:
                final_summary = await self._merge_summaries(chunk_summaries)
                await self._post_summary(target_channel, final_summary)
            except Exception as e:
                logger.error(f"Failed to merge or post summaries: {e}")
                # Try to post individual chunk summaries as fallback
                logger.info("Attempting to post individual chunk summaries...")
                for i, summary in enumerate(chunk_summaries, 1):
                    try:
                        await self._post_summary(target_channel, summary)
                    except Exception as post_e:
                        logger.error(f"Failed to post chunk {i} summary: {post_e}")

        finally:
            # Log final statistics
            logger.info("\nFinal Statistics:")
            logger.info(f"Total messages processed: {len(messages)}")
            logger.info(f"Total prompt tokens: {self.total_prompt_tokens}")
            logger.info(f"Total completion tokens: {self.total_completion_tokens}")
            logger.info(f"Total cost: ${self.total_cost:.4f}")

    def cleanup_old_checkpoints(self, max_age_days: int = 7):
        """Clean up checkpoint directories older than specified days."""
        try:
            now = datetime.now(timezone.utc)
            for run_dir in self.base_checkpoint_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                    
                try:
                    # Try to parse the directory name as a timestamp
                    dir_time = datetime.strptime(run_dir.name, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
                    age = now - dir_time
                    
                    if age.days > max_age_days:
                        logger.info(f"Cleaning up old checkpoint directory: {run_dir}")
                        for file in run_dir.iterdir():
                            file.unlink()
                        run_dir.rmdir()
                except (ValueError, OSError) as e:
                    logger.warning(f"Failed to process checkpoint directory {run_dir}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup old checkpoints: {e}")

    async def on_ready(self):
        """Called when the bot is ready."""
        logger.info(f"Logged in as {self.user} (id: {self.user.id})")
        
        # Clean up old checkpoints before starting
        self.cleanup_old_checkpoints()
        
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

        # Determine the effective start time
        if self.custom_start_time:
            effective_after = self.custom_start_time
            logger.info(f"Using custom start time: {effective_after}")
        elif self.test_mode:
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
        
        # Instead of direct processing, use new process_messages method
        await self.process_messages(new_messages, target_channel)
        
        # Update the last_summary_time to now
        # Only update in production mode
        if not self.test_mode:
            self._save_last_summary_time(datetime.now(timezone.utc))

        # Done! Close the bot.
        await self.close()


def parse_time_arg(time_str: str) -> datetime:
    """Parse a time argument string into a datetime object.
    
    Supports formats like:
    - "7d" for 7 days ago
    - "24h" for 24 hours ago
    - "YYYY-MM-DD" for a specific date
    - "YYYY-MM-DD HH:MM" for a specific date and time
    
    Args:
        time_str: The time string to parse
        
    Returns:
        datetime: The parsed datetime in UTC
        
    Raises:
        ValueError: If the time string format is invalid
    """
    # Check for relative time formats (e.g., "7d", "24h")
    relative_match = re.match(r"^(\d+)([dh])$", time_str)
    if relative_match:
        amount = int(relative_match.group(1))
        unit = relative_match.group(2)
        
        if unit == "d":
            delta = timedelta(days=amount)
        else:  # unit == "h"
            delta = timedelta(hours=amount)
            
        return datetime.now(timezone.utc) - delta
    
    # Try parsing as exact date/time
    try:
        # Try full datetime format first
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            # Try date-only format
            dt = datetime.strptime(time_str, "%Y-%m-%d")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            raise ValueError(
                "Invalid time format. Use one of:\n"
                "- Relative time: '7d' for 7 days, '24h' for 24 hours\n"
                "- Exact date: 'YYYY-MM-DD'\n"
                "- Exact datetime: 'YYYY-MM-DD HH:MM'"
            )

async def run_recap(test_mode: bool = False, start_time: Optional[str] = None,
                   checkpoint_dir: Optional[str] = None, load_checkpoint: Optional[str] = None,
                   keep_checkpoints_days: Optional[int] = 7):
    """Main async entry point."""
    bot_token = os.getenv("DISCORD_BOT_TOKEN", "")
    if not bot_token:
        raise ValueError("DISCORD_BOT_TOKEN not found in .env")

    # Parse start_time if provided
    custom_start = None
    if start_time:
        try:
            custom_start = parse_time_arg(start_time)
        except ValueError as e:
            logger.error(f"Invalid start time format: {e}")
            return

    bot = RecapBot(
        test_mode=test_mode,
        start_time=custom_start,
        checkpoint_dir=checkpoint_dir,
        load_checkpoint=load_checkpoint
    )
    
    # Clean up old checkpoints if requested
    if keep_checkpoints_days is not None:
        bot.cleanup_old_checkpoints(keep_checkpoints_days)
    
    await bot.start(bot_token)


def main():
    parser = argparse.ArgumentParser(description="Generate Discord channel summaries")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument(
        "--start", 
        type=str,
        help=(
            "When to start the summary from. Formats:\n"
            "- Relative time: '7d' for 7 days, '24h' for 24 hours\n"
            "- Exact date: 'YYYY-MM-DD'\n"
            "- Exact datetime: 'YYYY-MM-DD HH:MM'"
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Directory to store processing checkpoints"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        help="Load a specific checkpoint file instead of processing messages"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process messages and save checkpoints but don't post to Discord"
    )
    parser.add_argument(
        "--cost-estimate",
        action="store_true",
        help="Estimate the cost of processing without actually making API calls"
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=7,
        help="Number of days to keep checkpoint files (default: 7)"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't clean up old checkpoint files"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Check for test mode flag (either from args or env)
    test_mode = args.test or os.getenv("TEST_MODE", "").lower() == "true"
    
    # Run the bot
    asyncio.run(run_recap(
        test_mode=test_mode,
        start_time=args.start,
        checkpoint_dir=args.checkpoint_dir,
        load_checkpoint=args.load_checkpoint,
        keep_checkpoints_days=args.keep_checkpoints if not args.no_cleanup else None
    ))


if __name__ == "__main__":
    main()