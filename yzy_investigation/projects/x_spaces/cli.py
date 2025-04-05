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
    
    args = parser.parse_args()
    print(f"Parsed args: {args}")  # Debug print
    
    try:
        if args.command == "download":
            print("Executing download command...")  # Debug print
            download_space(args)
        elif args.command == "transcribe":
            print("Executing transcribe command...")  # Debug print
            transcribe_space(args)
        elif args.command == "download-transcribe":
            print("Executing download-transcribe command...")  # Debug print
            download_and_transcribe(args)
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