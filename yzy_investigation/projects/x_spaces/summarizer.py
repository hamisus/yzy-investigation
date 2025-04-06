"""
X/Twitter Space Transcript Summarizer module.

This module provides functionality to summarize transcriptions of X/Twitter Spaces,
particularly handling large transcripts by processing them in chunks and then
combining them into a final cohesive summary.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
import re
import concurrent.futures
from tqdm import tqdm

# For OpenAI API usage
import os
import openai


class SpaceSummarizer:
    """A class to handle the summarization of X/Twitter Spaces transcriptions."""
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: str = "gpt-4o-mini",
                 chunk_size: int = 15000,
                 max_workers: int = 1,
                 output_dir: Optional[str] = None):
        """
        Initialize the SpaceSummarizer.
        
        Args:
            api_key: OpenAI API key. If None, will try to use OPENAI_API_KEY environment variable.
            model: The OpenAI model to use for summarization.
            chunk_size: Number of characters per chunk for transcript processing.
            max_workers: Maximum number of workers for parallel processing.
            output_dir: Optional directory path where summaries will be saved.
                        If None, saves in summaries subdirectory.
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
        
        # Set default output directory to 'summaries' subdirectory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Default to 'summaries' subdirectory in same directory as input file
            self.output_dir = None
        
        # Track total token usage
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
    
    def _get_output_path(self, transcript_path: Path) -> Path:
        """
        Get the path where the summary should be saved.
        
        Args:
            transcript_path: Path to the transcript file
            
        Returns:
            Path where the summary should be saved
        """
        # Use the same name as the transcript file but with .summary.md extension
        output_name = transcript_path.stem.replace(".transcription", "") + ".summary.md"
        
        if self.output_dir:
            # Use specified output directory
            output_dir = Path(self.output_dir)
        else:
            # Default to 'summaries' directory parallel to transcripts
            # Assuming transcript_path is in data/processed/spaces/transcripts/
            output_dir = transcript_path.parent.parent / "summaries"
            
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / output_name
    
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate cost in USD for API usage based on token counts.
        
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

    def _split_transcript_into_chunks(self, full_text: str) -> List[str]:
        """
        Split the transcript text into manageable chunks.
        
        Args:
            full_text: The complete transcript text
            
        Returns:
            List of text chunks
        """
        chunks = []
        current_position = 0
        
        while current_position < len(full_text):
            # If we're near the end of the text
            if current_position + self.chunk_size >= len(full_text):
                chunks.append(full_text[current_position:])
                break
            
            # Find a good breaking point within the chunk_size limit
            chunk_end = current_position + self.chunk_size
            
            # Look for break points in order of preference:
            # 1. Long pause (..)
            # 2. Short pause (.)
            # 3. End of sentence (. ? !)
            # 4. Fallback to chunk_size if no natural breaks found
            
            # Search for pauses and sentence endings
            long_pause = full_text.rfind('(..)', current_position, chunk_end)
            short_pause = full_text.rfind('(.)', current_position, chunk_end)
            period = full_text.rfind('. ', current_position, chunk_end)
            question = full_text.rfind('? ', current_position, chunk_end)
            exclamation = full_text.rfind('! ', current_position, chunk_end)
            
            # Find the best break point
            break_points = [
                (long_pause + 4 if long_pause != -1 else -1),  # Add 4 to account for (..) length
                (short_pause + 3 if short_pause != -1 else -1),  # Add 3 to account for (.) length
                (period + 2 if period != -1 else -1),  # Add 2 to account for '. ' length
                (question + 2 if question != -1 else -1),
                (exclamation + 2 if exclamation != -1 else -1)
            ]
            
            # Filter out -1 values and find the latest break point
            valid_break_points = [p for p in break_points if p > current_position]
            
            if valid_break_points:
                # Take the latest valid break point
                break_point = max(valid_break_points)
                chunks.append(full_text[current_position:break_point])
                current_position = break_point
            else:
                # If no natural break points found, break at chunk_size
                # but try to break at a word boundary
                temp_end = chunk_end
                while temp_end > current_position and not full_text[temp_end - 1].isspace():
                    temp_end -= 1
                if temp_end > current_position:
                    chunk_end = temp_end
                chunks.append(full_text[current_position:chunk_end])
                current_position = chunk_end
        
        return chunks
    
    def _summarize_chunk(self, chunk_text: str, chunk_number: int, total_chunks: int) -> Tuple[str, Dict[str, int]]:
        """
        Summarize a single chunk of transcript using the OpenAI API.
        
        Args:
            chunk_text: The text chunk to summarize
            chunk_number: The index of this chunk
            total_chunks: The total number of chunks
            
        Returns:
            Tuple of (summary text, token usage stats)
        """
        prompt = f"""
You are summarizing part {chunk_number} of {total_chunks} from a transcript about a crypto coin called "YzY" or "4NBT" by Ye, formerly known as Kanye West.

Create a concise, factual summary of the key points in this transcript segment. Focus on:
- Main topics discussed
- Details about YzY/4NBT
- Important statements or claims made by speakers
- Any other significant information

IMPORTANT:
- Include all important ideas, not just YzY/4NBT-related content
- Avoid speculation or adding outside knowledge
- Keep it factual and based only on what is stated in the text
- Present information in a clear, minutes-style format

Here's the transcript segment:
----------------
{chunk_text}
----------------
"""
        
        try:
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise, factual summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Track token usage
            usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens
            }
            self.total_prompt_tokens += usage['prompt_tokens']
            self.total_completion_tokens += usage['completion_tokens']
            self.total_cost += self._calculate_cost(
                self.model, 
                usage['prompt_tokens'], 
                usage['completion_tokens']
            )
            
            print(f"\nChunk {chunk_number} token usage:")
            print(f"  Prompt tokens: {usage['prompt_tokens']}")
            print(f"  Completion tokens: {usage['completion_tokens']}")
            print(f"  Cost: ${self._calculate_cost(self.model, usage['prompt_tokens'], usage['completion_tokens']):.4f}")
            
            return response.choices[0].message.content.strip(), usage
        except Exception as e:
            raise Exception(f"Failed to summarize chunk {chunk_number}: {str(e)}")
    
    def _merge_summaries(self, chunk_summaries: List[str]) -> str:
        """
        Merge individual chunk summaries into a cohesive final summary.
        
        Args:
            chunk_summaries: List of summaries from individual chunks
            
        Returns:
            Consolidated final summary
        """
        # If we have too many summaries, merge them in batches
        MAX_SUMMARIES_PER_BATCH = 3  # Adjust this based on typical summary length
        
        if len(chunk_summaries) > MAX_SUMMARIES_PER_BATCH:
            print(f"\nMerging {len(chunk_summaries)} summaries in batches of {MAX_SUMMARIES_PER_BATCH}...")
            
            # First level: merge in small batches
            intermediate_summaries = []
            for i in range(0, len(chunk_summaries), MAX_SUMMARIES_PER_BATCH):
                batch = chunk_summaries[i:i + MAX_SUMMARIES_PER_BATCH]
                print(f"\nMerging batch {(i//MAX_SUMMARIES_PER_BATCH)+1} of {(len(chunk_summaries)-1)//MAX_SUMMARIES_PER_BATCH + 1}")
                intermediate_summary = self._merge_batch(batch, is_final=False)
                intermediate_summaries.append(intermediate_summary)
            
            # If we still have too many summaries, recurse
            if len(intermediate_summaries) > MAX_SUMMARIES_PER_BATCH:
                return self._merge_summaries(intermediate_summaries)
            
            # Final merge of intermediate summaries
            print("\nPerforming final merge of intermediate summaries...")
            return self._merge_batch(intermediate_summaries, is_final=True)
        
        # If we have a manageable number of summaries, merge them directly
        return self._merge_batch(chunk_summaries, is_final=True)
    
    def _merge_batch(self, summaries: List[str], is_final: bool = True) -> str:
        """Merge a small batch of summaries."""
        combined_text = "\n\n".join(summaries)
        
        prompt = f"""
You are creating a{'final' if is_final else 'n intermediate'} summary of a transcript about a crypto coin called "YzY" or "4NBT" by Ye (Kanye West).

{'Below are summaries from different parts of the transcript.' if is_final else 'Below are partial summaries that need to be combined.'} Create a well-structured Markdown summary that:

1. Starts with a brief overview paragraph that captures the main themes and key points
2. Organizes the remaining information into logical sections based on the actual content:
   - Use level 1 headers (# ) only for major themes that have substantial discussion
   - Use level 2 headers (## ) sparingly for significant subtopics
   - If there are only 1-2 main topics, prefer a simpler structure without many headers
   - Let the content dictate the organization - don't force a specific structure

3. Uses appropriate Markdown formatting:
   - Bullet points for lists of related items
   - *Emphasis* for important terms or metrics
   - > Quote blocks for significant direct statements
   - Paragraphs for flowing narrative

4. Maintains factual accuracy:
   - Includes only information from the transcript
   - Preserves important numbers and metrics
   - Notes any conflicting information
   - Avoids speculation

IMPORTANT: The structure should emerge from the content. Don't create sections unless they naturally arise from having multiple distinct topics with substantial discussion.

Here are the summaries to consolidate:
----------------
{combined_text}
----------------
"""
        
        try:
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates well-structured, factual summaries in Markdown format. You adapt the structure to match the content rather than forcing a predefined format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            # Track token usage
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens
            self.total_cost += self._calculate_cost(
                self.model, 
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
            
            print(f"\n{'Final' if is_final else 'Intermediate'} merge token usage:")
            print(f"  Prompt tokens: {response.usage.prompt_tokens}")
            print(f"  Completion tokens: {response.usage.completion_tokens}")
            print(f"  Cost: ${self._calculate_cost(self.model, response.usage.prompt_tokens, response.usage.completion_tokens):.4f}")
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"Failed to merge summaries batch: {str(e)}")

    def _get_temp_dir(self, transcript_path: Path) -> Path:
        """Get path to temporary directory for storing intermediate summaries."""
        # Store temp files in data/processed/spaces/summaries/temp/
        temp_dir = transcript_path.parent.parent / "summaries" / "temp" / transcript_path.stem
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir

    def _save_chunk_summary(self, chunk_summary: str, chunk_text: str, chunk_number: int, temp_dir: Path) -> None:
        """
        Save an individual chunk summary and its original text to temporary storage.
        
        Args:
            chunk_summary: The generated summary for this chunk
            chunk_text: The original text of this chunk
            chunk_number: The chunk index number
            temp_dir: Directory to save the files
        """
        # Save the summary
        summary_file = temp_dir / f"chunk_{chunk_number:03d}.summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(chunk_summary)
            
        # Save the original chunk text
        chunk_file = temp_dir / f"chunk_{chunk_number:03d}.text.txt"
        with open(chunk_file, 'w', encoding='utf-8') as f:
            f.write(chunk_text)

    def summarize_from_text(self, transcript_text: str, output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Summarize a transcript from raw text.
        
        Args:
            transcript_text: The complete transcript text to summarize
            output_path: Optional path to save the summary. If None and self.output_dir
                        is None, the summary will only be returned without saving.
            
        Returns:
            The final merged summary text
        """
        # Create temp directory for intermediate summaries
        if output_path:
            temp_dir = self._get_temp_dir(Path(output_path))
        else:
            raise ValueError("output_path is required to store intermediate summaries")

        print("Splitting transcript into chunks...")
        chunks = self._split_transcript_into_chunks(transcript_text)
        print(f"Transcript split into {len(chunks)} chunks")
        
        # Reset token tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        
        # Process all chunks
        chunk_summaries = []
        
        if self.max_workers > 1:
            # Parallel processing
            print(f"Processing chunks in parallel with {self.max_workers} workers...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_chunk = {
                    executor.submit(
                        self._summarize_chunk, chunk, i+1, len(chunks)
                    ): i for i, chunk in enumerate(chunks)
                }
                
                for future in tqdm(concurrent.futures.as_completed(future_to_chunk), total=len(chunks)):
                    chunk_idx = future_to_chunk[future]
                    try:
                        summary, _ = future.result()
                        chunk_summaries.append((chunk_idx, summary))
                        # Save intermediate summary
                        self._save_chunk_summary(summary, chunks[chunk_idx], chunk_idx + 1, temp_dir)
                    except Exception as e:
                        print(f"Chunk {chunk_idx+1} failed: {str(e)}")
            
            # Sort summaries by chunk index
            chunk_summaries.sort()
            chunk_summaries = [summary for _, summary in chunk_summaries]
            
        else:
            # Sequential processing
            print("Processing chunks sequentially...")
            for i, chunk in enumerate(tqdm(chunks)):
                summary, _ = self._summarize_chunk(chunk, i+1, len(chunks))
                chunk_summaries.append(summary)
                # Save intermediate summary
                self._save_chunk_summary(summary, chunk, i + 1, temp_dir)
        
        print("\nMerging chunk summaries into final summary...")
        print(f"Total chunks to merge: {len(chunk_summaries)}")
        print("Chunk summaries saved in:", temp_dir)
        
        try:
            final_summary = self._merge_summaries(chunk_summaries)
        except Exception as e:
            print(f"\nError during merge: {str(e)}")
            print("Individual chunk summaries are available in:", temp_dir)
            raise
        
        # Print total usage
        print("\nTotal token usage:")
        print(f"  Prompt tokens: {self.total_prompt_tokens}")
        print(f"  Completion tokens: {self.total_completion_tokens}")
        print(f"  Total tokens: {self.total_prompt_tokens + self.total_completion_tokens}")
        print(f"  Total cost: ${self.total_cost:.4f}")
        
        # Save if requested
        if output_path:
            output_path = Path(output_path)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(final_summary)
            print(f"\nSummary saved to: {output_path}")
        
        return final_summary
    
    def summarize_from_file(self, transcript_path: Union[str, Path]) -> str:
        """
        Summarize a transcript from a file.
        
        Args:
            transcript_path: Path to the transcript file (either plain text or JSON from SpaceTranscriber)
            
        Returns:
            The final summary text
        """
        transcript_path = Path(transcript_path)
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
        
        print(f"Reading transcript from: {transcript_path}")
        
        # Determine output path
        output_path = self._get_output_path(transcript_path)
        
        # Read the transcript
        with open(transcript_path, 'r', encoding='utf-8') as f:
            # First try to handle as plain text (most common format)
            if transcript_path.suffix.lower() == '.txt':
                full_text = f.read()
                return self.summarize_from_text(full_text, output_path)
            
            # If not plaintext, try JSON format (from SpaceTranscriber)
            try:
                transcript_data = json.load(f)
                
                # Extract full text from the transcript
                if isinstance(transcript_data, dict) and 'text' in transcript_data:
                    # Standard Whisper output format
                    full_text = transcript_data['text']
                else:
                    # Unknown format
                    raise ValueError("Unsupported transcript format. Expected Whisper output JSON.")
                
                return self.summarize_from_text(full_text, output_path)
                
            except json.JSONDecodeError:
                # Fallback for non-JSON files
                f.seek(0)
                full_text = f.read()
                
                return self.summarize_from_text(full_text, output_path) 