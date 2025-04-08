"""
Command-line interface for X/Twitter Spaces tools.

This module provides a command-line interface for downloading and transcribing
X/Twitter Spaces.
"""

import sys
import traceback
import argparse
from pathlib import Path
from typing import Optional, NoReturn
from .downloader import SpaceDownloader
from .transcriber import SpaceTranscriber
from .summarizer import SpaceSummarizer, SummaryFormat


def handle_error(e: Exception) -> NoReturn:
    """Handle errors in a user-friendly way."""
    print("\nError occurred:", file=sys.stderr)
    print(f"  {str(e)}", file=sys.stderr)
    if "--debug" in sys.argv:
        print("\nFull traceback:", file=sys.stderr)
        traceback.print_exc()
    sys.exit(1)


def download_space(args: argparse.Namespace) -> Optional[Path]:
    """Download a Space from the provided URL."""
    print("Starting download_space function...")  # Debug print
    try:
        print(f"Creating downloader with output_dir: {args.output_dir}")  # Debug print
        downloader = SpaceDownloader(output_dir=args.output_dir)
        print(f"Attempting to download from URL: {args.url}")  # Debug print
        return downloader.download_space(args.url)
    except Exception as e:
        print(f"Caught exception in download_space: {str(e)}")  # Debug print
        handle_error(e)


def transcribe_space(args: argparse.Namespace) -> None:
    """Transcribe a downloaded Space audio file."""
    try:
        transcriber = SpaceTranscriber(
            model_name=args.model,
            output_dir=args.output_dir
        )
        transcriber.transcribe(args.audio_file, language=args.language)
    except Exception as e:
        handle_error(e)


def summarize_transcript(args: argparse.Namespace) -> None:
    """Summarize a transcript file."""
    try:
        summarizer = SpaceSummarizer(
            api_key=args.api_key,
            model=args.model,
            chunk_size=args.chunk_size,
            max_workers=args.max_workers,
            output_dir=args.output_dir,
            format_type=args.format
        )
        summarizer.summarize_from_file(args.transcript_file)
    except Exception as e:
        handle_error(e)


def download_and_transcribe(args: argparse.Namespace) -> None:
    """Download a Space and then transcribe it."""
    try:
        # First download the space
        audio_path = download_space(args)
        if audio_path is None:
            return
            
        # Update args for transcription
        args.audio_file = audio_path
        
        # Then transcribe it
        transcribe_space(args)
    except Exception as e:
        handle_error(e)


def download_transcribe_and_summarize(args: argparse.Namespace) -> None:
    """Download a Space, transcribe it, and then summarize the transcript."""
    try:
        # First download the space
        audio_path = download_space(args)
        if audio_path is None:
            return
            
        # Update args for transcription
        args.audio_file = audio_path
        
        # Then transcribe it
        transcriber = SpaceTranscriber(
            model_name=args.whisper_model,
            output_dir=args.output_dir
        )
        transcript_result = transcriber.transcribe(args.audio_file, language=args.language)
        
        # Get the path to the transcript file
        transcript_path = Path(audio_path).parent / f"{Path(audio_path).stem}.transcription.json"
        
        # Update args for summarization
        args.transcript_file = transcript_path
        
        # Finally, summarize the transcript
        summarize_transcript(args)
    except Exception as e:
        handle_error(e)


def main() -> None:
    """Main entry point for the CLI."""
    print("Starting main function...")  # Debug print
    parser = argparse.ArgumentParser(
        description="X/Twitter Spaces downloader and transcriber"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug information on errors"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Download command
    download_parser = subparsers.add_parser(
        "download",
        help="Download a Space from URL"
    )
    download_parser.add_argument(
        "url",
        help="URL of the X/Twitter Space"
    )
    download_parser.add_argument(
        "--output-dir",
        help="Directory to save the downloaded audio",
        default=None
    )
    
    # Transcribe command
    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Transcribe a downloaded Space"
    )
    transcribe_parser.add_argument(
        "audio_file",
        help="Path to the audio file to transcribe"
    )
    transcribe_parser.add_argument(
        "--model",
        help="Whisper model to use",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base"
    )
    transcribe_parser.add_argument(
        "--language",
        help="Language code (e.g., 'en' for English)",
        default=None
    )
    transcribe_parser.add_argument(
        "--output-dir",
        help="Directory to save the transcription",
        default=None
    )
    
    # Common OpenAI API arguments
    openai_args = argparse.ArgumentParser(add_help=False)
    openai_args.add_argument(
        "--api-key",
        help="OpenAI API key (if not provided, will use OPENAI_API_KEY environment variable)",
        default=None
    )
    openai_args.add_argument(
        "--model",
        help="OpenAI model to use for summarization",
        default="gpt-4o"
    )
    openai_args.add_argument(
        "--format",
        help="Format for the summary output",
        choices=[f.value for f in SummaryFormat],
        default=SummaryFormat.TIMELINE.value
    )
    
    # Summarize command
    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize a transcript file",
        parents=[openai_args]
    )
    summarize_parser.add_argument(
        "transcript_file",
        help="Path to the transcript file to summarize"
    )
    summarize_parser.add_argument(
        "--chunk-size",
        help="Number of characters per chunk for transcript processing (minimum 8000)",
        type=int,
        default=15000
    )
    summarize_parser.add_argument(
        "--max-workers",
        help="Maximum number of workers for parallel processing",
        type=int,
        default=1
    )
    summarize_parser.add_argument(
        "--output-dir",
        help="Directory to save the summary",
        default=None
    )
    
    # Download and transcribe command
    download_transcribe_parser = subparsers.add_parser(
        "download-transcribe",
        help="Download and transcribe a Space"
    )
    download_transcribe_parser.add_argument(
        "url",
        help="URL of the X/Twitter Space"
    )
    download_transcribe_parser.add_argument(
        "--model",
        help="Whisper model to use",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base"
    )
    download_transcribe_parser.add_argument(
        "--language",
        help="Language code (e.g., 'en' for English)",
        default=None
    )
    download_transcribe_parser.add_argument(
        "--output-dir",
        help="Directory to save both audio and transcription",
        default=None
    )
    
    # Download, transcribe, and summarize command
    download_transcribe_summarize_parser = subparsers.add_parser(
        "download-transcribe-summarize",
        help="Download, transcribe, and summarize a Space",
        parents=[openai_args]
    )
    download_transcribe_summarize_parser.add_argument(
        "url",
        help="URL of the X/Twitter Space"
    )
    download_transcribe_summarize_parser.add_argument(
        "--whisper-model",
        help="Whisper model to use for transcription",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        dest="whisper_model"
    )
    download_transcribe_summarize_parser.add_argument(
        "--language",
        help="Language code (e.g., 'en' for English)",
        default=None
    )
    download_transcribe_summarize_parser.add_argument(
        "--chunk-size",
        help="Number of characters per chunk for transcript processing (minimum 8000)",
        type=int,
        default=15000
    )
    download_transcribe_summarize_parser.add_argument(
        "--max-workers",
        help="Maximum number of workers for parallel processing",
        type=int,
        default=1
    )
    download_transcribe_summarize_parser.add_argument(
        "--output-dir",
        help="Directory to save audio, transcription, and summary",
        default=None
    )
    
    args = parser.parse_args()
    print(f"Parsed args: {args}")  # Debug print
    
    try:
        if args.command == "download":
            print("Executing download command...")  # Debug print
            download_space(args)
        elif args.command == "transcribe":
            print("Executing transcribe command...")  # Debug print
            transcribe_space(args)
        elif args.command == "summarize":
            print("Executing summarize command...")  # Debug print
            summarize_transcript(args)
        elif args.command == "download-transcribe":
            print("Executing download-transcribe command...")  # Debug print
            download_and_transcribe(args)
        elif args.command == "download-transcribe-summarize":
            print("Executing download-transcribe-summarize command...")  # Debug print
            download_transcribe_and_summarize(args)
        else:
            print("No command specified, showing help...")  # Debug print
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Caught exception in main: {str(e)}")  # Debug print
        handle_error(e)

if __name__ == "__main__":
    print("Running CLI directly...")  # Debug print
    main() 