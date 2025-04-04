"""
X Spaces downloader and transcription module.

This module provides functionality to download X (formerly Twitter) Spaces
and transcribe their audio content.
"""

from .downloader import SpaceDownloader
from .transcriber import SpaceTranscriber

__all__ = ['SpaceDownloader', 'SpaceTranscriber'] 