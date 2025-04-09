"""Discord Message Summarizer module.

This module provides functionality to summarize Discord channel messages,
particularly handling large message histories by processing them in chunks
and then combining them into a final cohesive summary.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
import re
import concurrent.futures
from tqdm import tqdm
from enum import Enum
from datetime import datetime, timedelta
import os
import openai

class SummaryFormat(str, Enum):
    """Enum for different summary formats."""
    MARKDOWN = "markdown"  # Traditional markdown format with sections
    TIMELINE = "timeline"  # Chronological bullet points format

class DiscordMessageSummarizer:
    """A class to handle the summarization of Discord channel messages."""
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: str = "gpt-4",
                 chunk_size: int = 15000,
                 max_workers: int = 1,
                 output_dir: Optional[str] = None,
                 format_type: Union[str, SummaryFormat] = SummaryFormat.TIMELINE):
        """
        Initialize the DiscordMessageSummarizer.
        
        Args:
            api_key: OpenAI API key. If None, will try to use OPENAI_API_KEY environment variable.
            model: The OpenAI model to use for summarization.
            chunk_size: Number of characters per chunk for message processing.
            max_workers: Maximum number of workers for parallel processing.
            output_dir: Optional directory path where summaries will be saved.
            format_type: The format to use for the summary. Either 'markdown' or 'timeline'.
        """
        # Set up OpenAI API
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        openai.api_key = self.api_key
        self.model = model
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = None
            
        if isinstance(format_type, str):
            try:
                self.format_type = SummaryFormat(format_type.lower())
            except ValueError:
                raise ValueError(f"Invalid format_type: {format_type}. Must be one of: {[f.value for f in SummaryFormat]}")
        else:
            self.format_type = format_type
            
        # Track total token usage
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0

    def _format_message(self, message: Dict[str, Any]) -> str:
        """Format a single message for summarization.
        
        Args:
            message: Message data from Discord
            
        Returns:
            Formatted message string
        """
        timestamp = datetime.fromisoformat(message['timestamp']).strftime("%H:%M")
        author = message['author']
        content = message['content']
        message_id = message['id']
        
        # Handle attachments
        attachments = ""
        if message['attachments']:
            attachments = " [Attachments: " + ", ".join(a['filename'] for a in message['attachments']) + "]"
            
        # Handle embeds
        embeds = ""
        if message['embeds']:
            embed_types = []
            for embed in message['embeds']:
                if embed.get('type') == 'rich':
                    if embed.get('title'):
                        embed_types.append(f"Embed: {embed['title']}")
                    else:
                        embed_types.append("Rich Embed")
                else:
                    embed_types.append(embed.get('type', 'Embed'))
            embeds = " [Embeds: " + ", ".join(embed_types) + "]"
            
        # Add message ID as a hidden reference that won't clutter the display
        return f"[{timestamp}] {author}: {content}{attachments}{embeds} <!-- msg:{message_id} -->"

    def _split_messages_into_chunks(self, messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Split messages into manageable chunks.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            List of message chunks
        """
        chunks = []
        current_chunk = []
        current_size = 0
        
        for message in messages:
            formatted_message = self._format_message(message)
            message_size = len(formatted_message)
            
            if current_size + message_size > self.chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0
                
            current_chunk.append(message)
            current_size += message_size
            
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

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
        if model == "gpt-4.5":
            prompt_cost = 75.00 / 1_000_000
            completion_cost = 150.00 / 1_000_000
        elif model == "gpt-4o":
            prompt_cost = 2.50 / 1_000_000
            completion_cost = 10.00 / 1_000_000
        elif model == "gpt-4o-mini":
            prompt_cost = 0.150 / 1_000_000
            completion_cost = 0.600 / 1_000_000
        # Legacy models
        elif model.startswith("gpt-4"):
            if "32k" in model:
                prompt_cost = 0.06 / 1000  # Price per token
                completion_cost = 0.12 / 1000
            else:
                prompt_cost = 0.03 / 1000
                completion_cost = 0.06 / 1000
        elif model.startswith("gpt-3.5"):
            if "16k" in model:
                prompt_cost = 0.003 / 1000
                completion_cost = 0.004 / 1000
            else:
                prompt_cost = 0.0015 / 1000
                completion_cost = 0.002 / 1000
        else:
            return 0.0
            
        # Calculate total cost
        total_cost = (prompt_tokens * prompt_cost) + (completion_tokens * completion_cost)
        return round(total_cost, 4)  # Round to 4 decimal places

    def _summarize_chunk(self, chunk: List[Dict[str, Any]], chunk_number: int, total_chunks: int, server_id: str, channel_id: str) -> Tuple[str, Dict[str, int]]:
        """Summarize a single chunk of messages.
        
        Args:
            chunk: List of messages in the chunk
            chunk_number: The chunk index
            total_chunks: Total number of chunks
            server_id: Discord server ID for message links
            channel_id: Discord channel ID for message links
            
        Returns:
            Tuple of (summary text, token usage stats)
        """
        formatted_messages = [self._format_message(msg) for msg in chunk]
        messages_text = "\n".join(formatted_messages)
        
        # Base prompt for all formats
        base_prompt = f"""
                    You are summarizing part {chunk_number} of {total_chunks} from a Discord channel's message history.

                    Create a JSON array with multiple distinct topics covering the key points in this message segment.
                    Each topic in the array should have:
                    - "topic": A brief title for the discussion
                    - "details": An array of bullet points about this topic
                    - "sources": An array of message URLs for relevant messages

                    Focus on:
                    - Main topics discussed
                    - Important announcements or updates
                    - Questions asked and their answers
                    - Key decisions or action items
                    - Any other significant information

                    IMPORTANT:
                    - Include all important ideas
                    - Organize content into multiple distinct topics
                    - Avoid speculation or adding outside knowledge
                    - Keep it factual and based only on what is stated in the messages
                    - Do not omit any significant points or discussions
                    - Include the author and usernames of individuals involved in the discussion.
                    - For significant messages, extract their message ID from the <!-- msg:ID --> comment and include a link in this format:
                    https://discord.com/channels/{server_id}/{channel_id}/MESSAGE_ID
                    """

        prompt = base_prompt + f"""
                                Here are the messages to summarize:
                                ----------------
                                {messages_text}
                                ----------------

                                IMPORTANT: The response MUST be a valid JSON array with the format:
                                [
                                  {{
                                    "topic": "Discussion Topic 1",
                                    "details": ["Detail 1", "Detail 2", "Detail 3"],
                                    "sources": ["https://discord.com/channels/{server_id}/{channel_id}/MESSAGE_ID"]
                                  }},
                                  {{
                                    "topic": "Discussion Topic 2",
                                    "details": ["Detail 1", "Detail 2"],
                                    "sources": ["https://discord.com/channels/{server_id}/{channel_id}/MESSAGE_ID"]
                                  }}
                                ]
                                
                                Extract the message ID from the <!-- msg:ID --> comment in the message text.
                                """
        
        try:
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are a specialized JSON summarizer that creates concise, structured summaries in JSON format. Your output must be a valid JSON array with multiple topic objects. Each object must have 'topic', 'details' (array of points), and 'sources' (array of URLs) fields. Include Discord message links for important messages using the format: https://discord.com/channels/{server_id}/{channel_id}/MESSAGE_ID."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # Track token usage
            usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens
            }
            self.total_prompt_tokens += usage['prompt_tokens']
            self.total_completion_tokens += usage['completion_tokens']
            
            # Calculate and track cost
            chunk_cost = self._calculate_cost(
                self.model,
                usage['prompt_tokens'],
                usage['completion_tokens']
            )
            self.total_cost += chunk_cost
            
            print(f"\nChunk {chunk_number} token usage:")
            print(f"  Prompt tokens: {usage['prompt_tokens']}")
            print(f"  Completion tokens: {usage['completion_tokens']}")
            print(f"  Cost: ${chunk_cost:.4f}")
            
            content = response.choices[0].message.content.strip()
            
            # Validate the output is a proper JSON array
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    print(f"Warning: Chunk {chunk_number} returned a single object instead of an array. Converting to array.")
                    data = [data]
                    content = json.dumps(data, indent=2)
                
                print(f"Chunk {chunk_number} generated {len(data)} topics")
            except json.JSONDecodeError:
                print(f"Warning: Chunk {chunk_number} returned invalid JSON. Using raw content.")
            
            return content, usage
        except Exception as e:
            raise Exception(f"Failed to summarize chunk {chunk_number}: {str(e)}")

    def _merge_summaries(self, chunk_summaries: List[str], server_id: str = "SERVER_ID", channel_id: str = "CHANNEL_ID") -> str:
        """Merge individual chunk summaries into a cohesive final summary.
        
        Args:
            chunk_summaries: List of summaries from individual chunks
            server_id: Discord server ID for message links
            channel_id: Discord channel ID for message links
            
        Returns:
            Consolidated final summary in JSON format
        """
        combined_text = "\n\n".join(chunk_summaries)
        
        print(f"Merging {len(chunk_summaries)} chunk summaries")
        
        # Base prompt for all formats - keep most of the original
        base_prompt = f"""
                        You are creating a final summary of a Discord channel's message history.

                        IMPORTANT: 
                        - Return the summary as a valid JSON ARRAY with MULTIPLE distinct topics
                        - Do NOT merge all information into a single topic
                        - Your response must be an array of objects in JSON format with AT LEAST 5-10 distinct topics
                        - ALL topics must include a "details" array, even if it only has one item
                        - The response must look like: [{{"topic": "Topic 1", "details": ["point 1", "point 2", "point 3"],...}}, {{"topic": "Topic 2", "details": ["point 1"],...}}]
                        - Keep topics separate and distinct - don't combine different discussions into one topic
                        - Do not omit any significant information
                        - Break down complex topics into multiple detailed bullet points
                        - Maintain all important points from each summary
                        - Include message IDs for reference when relevant
                        - Include the author and usernames and entities involved in each specific discussion.
                        - For all significant messages, include a Discord message link in the format:
                        [Source](https://discord.com/channels/{server_id}/{channel_id}/MESSAGE_ID)
                        where MESSAGE_ID is the ID of the message you are referencing
                        """

        # Expected output format example
        example_format = """
        Here's the exact JSON format I need:

        [
          {
            "topic": "Discussion on Ye's Tweet and Codes",
            "details": [
              "flochst speculated on hidden meanings in Ye's tweet related to '4NBT', leading to a series of messages attempting to decode these symbols",
              "User414 responded with additional theories about potential double meanings in the tweet",
              "The community formed multiple interpretations about possible links to earlier cryptic messages"
            ],
            "sources": [
              "https://discord.com/channels/{server_id}/{channel_id}/123456789",
              "https://discord.com/channels/{server_id}/{channel_id}/987654321"
            ]
          },
          {
            "topic": "Community Investment Discussion",
            "details": [
              "Users prawnomics and actuallyethane expressed enthusiasm for investing in '4NBT'",
              "Discussions about risk assessment and potential return on investment followed",
              "Several members shared strategies for position sizing and entry points"
            ],
            "sources": [
              "https://discord.com/channels/{server_id}/{channel_id}/123456789"
            ]
          },
          {
            "topic": "Technical Updates",
            "details": [
              "Developer announced progress on feature X, responding to user questions about timeline"
            ],
            "sources": [
              "https://discord.com/channels/{server_id}/{channel_id}/123456789"
            ]
          }
        ]
        
        Note that this is a JSON array, not a single object.
        ALL topics must use a 'details' array of points, even if there's only one point.
        """

        # Format-specific prompts
        if self.format_type == SummaryFormat.MARKDOWN:
            prompt = base_prompt + f"""
                                    Create a well-structured summary formatted as a JSON array where each element has:
                                    - "topic": A string for the section title (e.g., "Community Updates", "Technical Discussion")
                                    - "details": An array containing MULTIPLE bulletin points about the topic (not just one)
                                    - "sources": An array of relevant Discord message links

                                    Your summary should:
                                    1. Include all key information that captures the main themes and key points
                                    2. Organize information into MULTIPLE distinct topics (at least 5-10)
                                    3. Use appropriate formatting within the details text:
                                       - Each detail should be a complete, informative bulletin point
                                       - Provide multiple (at least 2-3) bullet points per topic to ensure comprehensive coverage
                                       - Use *emphasis* for important terms or metrics
                                       - Include all significant information from the input
                                    4. Maintain factual accuracy:
                                       - Include only information from the messages
                                       - Preserve important numbers and metrics
                                       - Notes any conflicting information
                                       - Avoid speculation
                                       - Retain all significant points from the input summaries
"""
        else:  # SummaryFormat.TIMELINE
            prompt = base_prompt + f"""
                                    Create a chronological timeline summary formatted as a JSON array where each element has:
                                    - "topic": A brief descriptive title for this discussion point 
                                    - "details": An array containing points about this topic (even if only one)
                                    - "sources": An array of relevant Discord message links

                                    Your JSON array should maintain a timeline structure where:
                                    1. The array includes MULTIPLE distinct topics (at least 5-10)
                                    2. All points are in chronological order
                                    3. Each point is concise but complete
                                    4. Related points are grouped by topic
                                    5. All significant details are preserved including:
                                       - Exact numbers and metrics
                                       - Important quotes
                                       - Key discussions and their participants
                                    6. The factual accuracy is maintained:
                                       - Only information from the messages is included
                                       - Conflicting information is noted
                                       - No speculation is added
                                       - All key points from the input summaries are retained
                                    """

        prompt += f"""
                    Here are the summaries to consolidate:
                    ----------------
                    {combined_text}
                    ----------------

                    CRITICAL: Return your response as a valid JSON ARRAY of MULTIPLE distinct topic objects.
                    The final JSON must be an array starting with [ and ending with ] containing AT LEAST 5-10 different topic objects.
                    Each object in the array should represent a separate topic or point of discussion.
                    DO NOT merge all information into a single topic.

                    {example_format}
                    """
        
        try:
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are a helpful assistant that creates well-structured, factual summaries as valid JSON. Your output must be a valid JSON ARRAY (not a single object) without any explanatory text. The array MUST contain MULTIPLE distinct topic objects (at least 5-10), each with 'topic', 'details' (an array of bullet points, even if only one), and a 'sources' array of links. Always use 'details' array even for simple topics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # Track token usage
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens
            
            # Calculate and track cost
            merge_cost = self._calculate_cost(
                self.model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
            self.total_cost += merge_cost
            
            print(f"\nMerge token usage:")
            print(f"  Prompt tokens: {response.usage.prompt_tokens}")
            print(f"  Completion tokens: {response.usage.completion_tokens}")
            print(f"  Cost: ${merge_cost:.4f}")
            
            content = response.choices[0].message.content.strip()
            
            # Check if the response is a JSON array or object
            try:
                data = json.loads(content)
                # If it's a single object, wrap it in an array
                if isinstance(data, dict):
                    print("Warning: OpenAI returned a single object instead of an array. Converting to array.")
                    data = [data]
                    content = json.dumps(data, indent=2)
                
                # Check the number of topics
                if isinstance(data, list):
                    print(f"Received {len(data)} topics in the summary")
                    if len(data) < 2 and len(chunk_summaries) > 1:
                        print("Warning: OpenAI merged everything into a single topic. Attempting to create multiple topics.")
                        
                        # Try to extract multiple topics from the chunk summaries
                        all_topics = []
                        for chunk in chunk_summaries:
                            try:
                                # See if any chunk contains a list of topics
                                chunk_data = json.loads(chunk)
                                if isinstance(chunk_data, list):
                                    all_topics.extend(chunk_data)
                                elif isinstance(chunk_data, dict):
                                    all_topics.append(chunk_data)
                            except json.JSONDecodeError:
                                # Not valid JSON, just skip
                                pass
                        
                        if len(all_topics) > 1:
                            print(f"Extracted {len(all_topics)} topics from individual chunks")
                            # Use these topics instead
                            content = json.dumps(all_topics, indent=2)
            except json.JSONDecodeError:
                print("Warning: Invalid JSON returned. Returning as-is.")
            
            return content
        except Exception as e:
            raise Exception(f"Failed to merge summaries: {str(e)}")

    def summarize_messages(
        self, 
        messages: List[Dict[str, Any]], 
        output_path: Optional[Union[str, Path]] = None,
        server_id: Optional[str] = None,
        channel_id: Optional[str] = None
    ) -> str:
        """Summarize a list of Discord messages.
        
        Args:
            messages: List of message dictionaries
            output_path: Optional path to save the summary
            server_id: Optional Discord server ID for message links
            channel_id: Optional Discord channel ID for message links
            
        Returns:
            The final summary text as JSON
        """
        # Reset token tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        
        # Try to get server_id and channel_id from environment if not provided
        if not server_id:
            server_id = os.environ.get("DISCORD_SERVER_ID", "SERVER_ID")
        if not channel_id and messages and len(messages) > 0:
            # Try to extract from the first message if possible
            # This assumes messages are from a single channel
            channel_id = messages[0].get('channel_id', os.environ.get("DISCORD_CHANNEL_ID", "CHANNEL_ID"))
        
        print("Splitting messages into chunks...")
        chunks = self._split_messages_into_chunks(messages)
        print(f"Messages split into {len(chunks)} chunks")
        
        # Process all chunks
        chunk_summaries = []
        
        if self.max_workers > 1:
            # Parallel processing
            print(f"Processing chunks in parallel with {self.max_workers} workers...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_chunk = {
                    executor.submit(
                        self._summarize_chunk, chunk, i+1, len(chunks), server_id, channel_id
                    ): i for i, chunk in enumerate(chunks)
                }
                
                for future in tqdm(concurrent.futures.as_completed(future_to_chunk), total=len(chunks)):
                    chunk_idx = future_to_chunk[future]
                    try:
                        summary, _ = future.result()
                        chunk_summaries.append((chunk_idx, summary))
                    except Exception as e:
                        print(f"Chunk {chunk_idx+1} failed: {str(e)}")
            
            # Sort summaries by chunk index
            chunk_summaries.sort()
            chunk_summaries = [summary for _, summary in chunk_summaries]
            
        else:
            # Sequential processing
            print("Processing chunks sequentially...")
            for i, chunk in enumerate(tqdm(chunks)):
                summary, _ = self._summarize_chunk(chunk, i+1, len(chunks), server_id, channel_id)
                chunk_summaries.append(summary)
        
        print("\nMerging chunk summaries into final summary...")
        final_summary = self._merge_summaries(chunk_summaries, server_id, channel_id)
        
        # Print total usage
        print("\nTotal token usage:")
        print(f"  Prompt tokens: {self.total_prompt_tokens}")
        print(f"  Completion tokens: {self.total_completion_tokens}")
        print(f"  Total tokens: {self.total_prompt_tokens + self.total_completion_tokens}")
        print(f"  Total cost: ${self.total_cost:.4f}")
        
        # Save if requested
        if output_path:
            # Update file extension to json if not already specified
            output_path = Path(output_path)
            if output_path.suffix.lower() != '.json':
                # If original had .md, replace it, otherwise just add .json
                if output_path.suffix.lower() == '.md':
                    output_path = output_path.with_suffix('.json')
                else:
                    output_path = Path(str(output_path) + '.json')
                    
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(final_summary)
            print(f"\nSummary saved to: {output_path}")
        
        return final_summary 