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
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
import yt_dlp
import requests
from tqdm import tqdm


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
                       If None, creates a 'downloaded_spaces' directory in the current path.
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / 'downloaded_spaces'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress = ProgressLogger()
        
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
        self.progress.start_stage("Extracting Space information")
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(space_url, download=False)
                stream_url = info['url']
                # Format: YYYYMMDD - username.title.id.m4a
                filename = f"{datetime.now().strftime('%Y%m%d')} - {info['uploader']}.{info['title']}.{info['id']}.m4a"
                
                self.progress.finish_stage()
                duration_str = info.get('duration_string', 'Unknown duration')
                print(f"Found Space: {info['title']} by {info['uploader']} ({duration_str})")
                return stream_url, filename, info
                
        except Exception as e:
            self.progress.finish_stage()
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
        # Download the m3u8 file
        response = requests.get(url)
        if not response.ok:
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
            
            # Get stream information
            stream_url, filename, info = self._get_stream_info(space_url)
            output_path = self.output_dir / filename
            
            # Download and process m3u8 playlist
            self.progress.start_stage("Downloading playlist")
            playlist_path, modified_path, chunk_count = self._download_m3u8(stream_url, temp_dir)
            self.progress.finish_stage()
            
            # Download audio chunks
            self.progress.start_stage("Downloading audio chunks", total=chunk_count)
            
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
                raise Exception("Failed to download audio chunks")
            
            # Combine chunks into final file
            self.progress.start_stage("Combining audio chunks")
            
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
                raise Exception("Failed to combine audio chunks")
            
            # Final status
            file_size = output_path.stat().st_size / (1024 * 1024)  # Size in MB
            print(f"\nDownload completed successfully:")
            print(f"- File: {output_path}")
            print(f"- Size: {file_size:.1f} MB")
            print(f"- Duration: {info.get('duration_string', 'Unknown')}")
            
            return output_path
            
        except Exception as e:
            self.progress.finish_stage()
            raise Exception(f"Failed to download space: {str(e)}")
            
        finally:
            # Clean up temporary directory and all its contents
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True) 