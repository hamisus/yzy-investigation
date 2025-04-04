"""
Command-line interface for X/Twitter Spaces tools.

This module provides a command-line interface for downloading and transcribing
X/Twitter Spaces.
"""

import argparse
from pathlib import Path
from typing import Optional
from .downloader import SpaceDownloader
from .transcriber import SpaceTranscriber


def download_space(args: argparse.Namespace) -> Optional[Path]:
    """Download a Space from the provided URL."""
    downloader = SpaceDownloader(output_dir=args.output_dir)
    try:
        return downloader.download_space(args.url)
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def transcribe_space(args: argparse.Namespace) -> None:
    """Transcribe a downloaded Space audio file."""
    transcriber = SpaceTranscriber(
        model_name=args.model,
        output_dir=args.output_dir
    )
    try:
        transcriber.transcribe(args.audio_file, language=args.language)
    except Exception as e:
        print(f"Error: {str(e)}")


def download_and_transcribe(args: argparse.Namespace) -> None:
    """Download a Space and then transcribe it."""
    # First download the space
    audio_path = download_space(args)
    if audio_path is None:
        return
        
    # Update args for transcription
    args.audio_file = audio_path
    
    # Then transcribe it
    transcribe_space(args)


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="X/Twitter Spaces downloader and transcriber"
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
    
    if args.command == "download":
        download_space(args)
    elif args.command == "transcribe":
        transcribe_space(args)
    elif args.command == "download-transcribe":
        download_and_transcribe(args)
    else:
        parser.print_help() 