"""
X/Twitter Space Downloader module.

This module provides functionality to download X/Twitter Spaces and save them as audio files.
"""

import os
import sys
import subprocess
import re
import tempfile
import shutil
import logging
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
import yt_dlp
import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ProgressLogger:
    """Handles progress logging for different stages of the download process."""
    
    def __init__(self):
        """Initialize the progress logger."""
        self.current_stage: Optional[tqdm] = None
        
    def start_stage(self, desc: str, total: Optional[int] = None) -> None:
        """
        Start a new progress stage.
        
        Args:
            desc: Description of the stage
            total: Total steps for progress bar
        """
        if self.current_stage is not None:
            self.current_stage.close()
        
        self.current_stage = tqdm(
            total=total,
            desc=desc,
            unit='chunks' if total else '',
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit}' if total else '{desc}'
        )
        
    def update(self, n: int = 1) -> None:
        """Update the progress by increment."""
        if self.current_stage is not None:
            self.current_stage.update(n)
    
    def finish_stage(self) -> None:
        """Complete the current stage."""
        if self.current_stage is not None:
            self.current_stage.close()
            self.current_stage = None


class SpaceDownloader:
    """A class to handle the downloading and processing of X/Twitter Spaces."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the SpaceDownloader with an optional output directory.
        
        Args:
            output_dir: Optional directory path where downloaded spaces will be saved.
                       If None, saves to data/raw/spaces directory.
        """
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Find the project root by looking for the data/raw directory
            current_dir = Path(__file__).resolve().parent
            while current_dir.name != "yzy-investigation" and current_dir.parent != current_dir:
                current_dir = current_dir.parent
            if current_dir.name != "yzy-investigation":
                raise Exception("Could not find project root directory")
            
            self.output_dir = current_dir / 'data' / 'raw' / 'spaces'
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory set to: {self.output_dir}")
        self.progress = ProgressLogger()
        
    def _format_timestamp(self, timestamp: Optional[int]) -> Tuple[str, str]:
        """
        Format a Unix timestamp into both sortable and readable formats.
        
        Args:
            timestamp: Unix timestamp
            
        Returns:
            Tuple of (sortable_format, readable_format)
            sortable_format: YYYYMMDD_HHMM
            readable_format: YYYY-MM-DD HH:MM
        """
        if not timestamp:
            now = datetime.now()
            return (
                now.strftime("%Y%m%d_%H%M"),
                now.strftime("%Y-%m-%d %H:%M")
            )
            
        dt = datetime.fromtimestamp(timestamp)
        return (
            dt.strftime("%Y%m%d_%H%M"),
            dt.strftime("%Y-%m-%d %H:%M")
        )

    def _save_metadata(self, info: Dict[str, Any], base_path: Path) -> None:
        """
        Save space metadata to a JSON file.
        
        Args:
            info: Dictionary containing space metadata
            base_path: Path to the audio file (without extension)
        """
        metadata = {
            'title': info.get('title'),
            'description': info.get('description'),
            'uploader': info.get('uploader'),
            'uploader_id': info.get('uploader_id'),
            'live_status': info.get('live_status'),
            'recorded_at': self._format_timestamp(info.get('timestamp'))[1],
            'released_at': self._format_timestamp(info.get('release_timestamp'))[1],
            'duration': info.get('duration_string'),
            'id': info.get('id'),
            'url': info.get('webpage_url'),
            'participants': self._extract_participants(info.get('description', '')),
        }
        
        metadata_path = base_path.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata to: {metadata_path}")

    def _extract_participants(self, description: str) -> list[str]:
        """
        Extract participant names from the space description.
        
        Args:
            description: Space description text
            
        Returns:
            List of participant names
        """
        if not description or 'participated by' not in description.lower():
            return []
            
        # Extract everything after "participated by"
        parts = description.lower().split('participated by')
        if len(parts) < 2:
            return []
            
        # Split participants and clean up names
        participants = parts[1].split(',')
        return [p.strip() for p in participants]

    def _get_stream_info(self, space_url: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        Get the stream URL and filename for the Space.
        
        Args:
            space_url: The URL of the X/Twitter Space.
            
        Returns:
            Tuple containing the stream URL, desired filename, and info dict.
            
        Raises:
            Exception: If yt-dlp fails to extract the stream information.
        """
        logger.info(f"Extracting information from Space URL: {space_url}")
        self.progress.start_stage("Extracting Space information")
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(space_url, download=False)
                
                # Print all available info fields
                logger.info("Available Space information:")
                for key, value in info.items():
                    logger.info(f"- {key}: {value}")
                
                stream_url = info['url']
                
                # Get sortable and readable timestamps
                sortable_date, readable_date = self._format_timestamp(info.get('timestamp'))
                
                # Create base filename without extension
                base_filename = f"{sortable_date} - {info['title']}"
                # Sanitize filename
                base_filename = "".join(c for c in base_filename if c.isalnum() or c in ".- ")
                
                # Full filename with extension
                filename = f"{base_filename}.m4a"
                
                self.progress.finish_stage()
                
                # Print more detailed information about the space
                logger.info("\nSpace Details:")
                logger.info(f"- Title: {info['title']}")
                logger.info(f"- Creator: {info['uploader']} (@{info.get('uploader_id', 'unknown')})")
                logger.info(f"- Duration: {info.get('duration_string', 'Unknown')}")
                logger.info(f"- Recorded at: {readable_date}")
                if info.get('release_timestamp'):
                    release_date = self._format_timestamp(info['release_timestamp'])[1]
                    logger.info(f"- Released at: {release_date}")
                logger.info(f"- ID: {info['id']}")
                
                # Save base filename for metadata
                info['base_filename'] = base_filename
                
                return stream_url, filename, info
                
        except Exception as e:
            self.progress.finish_stage()
            logger.error(f"Failed to extract stream information: {str(e)}")
            raise Exception(f"Failed to extract stream information: {str(e)}")

    def _download_m3u8(self, url: str, temp_dir: Path) -> Tuple[str, str, int]:
        """
        Download and process the m3u8 playlist.
        
        Args:
            url: The m3u8 URL
            temp_dir: Directory to store temporary files
            
        Returns:
            Tuple of (playlist path, modified playlist path, chunk count)
        """
        logger.info("Downloading m3u8 playlist")
        # Download the m3u8 file
        response = requests.get(url)
        if not response.ok:
            logger.error("Failed to download m3u8 playlist")
            raise Exception("Failed to download m3u8 playlist")
            
        # Get the base URL for the stream
        stream_path = '/'.join(url.split('/')[:-1]) + '/'
        
        playlist_path = temp_dir / "stream.m3u8"
        modified_path = temp_dir / "modified.m3u8"
        
        # Save original playlist
        playlist_path.write_text(response.text)
        
        # Modify playlist with full URLs
        content = response.text
        modified_content = []
        chunk_count = 0
        
        for line in content.splitlines():
            if line.startswith('#'):
                modified_content.append(line)
            else:
                modified_content.append(stream_path + line)
                chunk_count += 1
                
        modified_path.write_text('\n'.join(modified_content))
        logger.info(f"Found {chunk_count} audio chunks to download")
        
        return str(playlist_path), str(modified_path), chunk_count

    def download_space(self, space_url: str) -> Path:
        """
        Download and process an X/Twitter Space.
        
        Args:
            space_url: The URL of the X/Twitter Space to download.
            
        Returns:
            Path object pointing to the downloaded file.
            
        Raises:
            Exception: If any step of the download process fails.
        """
        temp_dir = None
        try:
            # Create temporary directory for all processing
            temp_dir = Path(tempfile.mkdtemp())
            logger.info(f"Created temporary directory: {temp_dir}")
            
            # Get stream information
            stream_url, filename, info = self._get_stream_info(space_url)
            output_path = self.output_dir / filename
            
            if output_path.exists():
                logger.warning(f"File already exists: {output_path}")
                return output_path
            
            # Download and process m3u8 playlist
            self.progress.start_stage("Downloading playlist")
            playlist_path, modified_path, chunk_count = self._download_m3u8(stream_url, temp_dir)
            self.progress.finish_stage()
            
            # Download audio chunks
            self.progress.start_stage("Downloading audio chunks", total=chunk_count)
            logger.info("Starting audio chunk download")
            
            # Use aria2c for efficient chunk downloading
            aria_cmd = [
                'aria2c',
                '-x', '10',  # Max connections
                '--console-log-level=notice',  # Changed from warn to notice to see progress
                '--dir', str(temp_dir),  # Download chunks to temp directory
                '--auto-file-renaming=false',  # Don't rename files
                '--summary-interval=0',  # Disable summary
                '-i', modified_path
            ]
            
            process = subprocess.Popen(
                aria_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                cwd=str(temp_dir)  # Set working directory to temp_dir
            )
            
            # Monitor aria2c progress
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if "download completed." in line.lower():
                    self.progress.update(1)
            
            process.wait()
            self.progress.finish_stage()
            
            if process.returncode != 0:
                logger.error("Failed to download audio chunks")
                raise Exception("Failed to download audio chunks")
            
            # Combine chunks into final file
            self.progress.start_stage("Combining audio chunks")
            logger.info("Combining audio chunks into final file")
            
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', playlist_path,
                '-vn',
                '-acodec', 'copy',
                '-movflags', '+faststart',
                str(output_path)
            ]
            
            process = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, cwd=str(temp_dir))
            self.progress.finish_stage()
            
            if process.returncode != 0:
                logger.error("Failed to combine audio chunks")
                raise Exception("Failed to combine audio chunks")
            
            # Save metadata
            base_path = output_path.with_suffix('')  # Remove extension
            self._save_metadata(info, base_path)
            
            # Final status
            file_size = output_path.stat().st_size / (1024 * 1024)  # Size in MB
            logger.info("\nDownload completed successfully:")
            logger.info(f"- Audio file: {output_path}")
            logger.info(f"- Metadata: {base_path.with_suffix('.json')}")
            logger.info(f"- Size: {file_size:.1f} MB")
            logger.info(f"- Duration: {info.get('duration_string', 'Unknown')}")
            
            return output_path
            
        except Exception as e:
            self.progress.finish_stage()
            logger.error(f"Failed to download space: {str(e)}")
            raise Exception(f"Failed to download space: {str(e)}")
            
        finally:
            # Clean up temporary directory and all its contents
            if temp_dir and temp_dir.exists():
                logger.info("Cleaning up temporary files")
                shutil.rmtree(temp_dir, ignore_errors=True) 