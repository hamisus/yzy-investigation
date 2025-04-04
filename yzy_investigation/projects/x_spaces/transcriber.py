"""
X/Twitter Space Transcriber module.

This module provides functionality to transcribe X/Twitter Spaces audio files
using OpenAI's Whisper model.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
import whisper
from tqdm import tqdm


class SpaceTranscriber:
    """A class to handle the transcription of X/Twitter Spaces audio files."""
    
    def __init__(self, model_name: str = "base", output_dir: Optional[str] = None):
        """
        Initialize the SpaceTranscriber.
        
        Args:
            model_name: The Whisper model to use for transcription.
                       Options: "tiny", "base", "small", "medium", "large"
            output_dir: Optional directory path where transcriptions will be saved.
                       If None, saves in the same directory as the audio file.
        """
        self.model = whisper.load_model(model_name)
        self.output_dir = Path(output_dir) if output_dir else None
        
    def transcribe(self, audio_path: Union[str, Path], 
                  language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to the audio file to transcribe
            language: Optional language code (e.g., "en" for English).
                     If None, language will be auto-detected.
        
        Returns:
            Dictionary containing the transcription results including:
            - text: The full transcription text
            - segments: List of transcribed segments with timestamps
            - language: Detected or specified language
        
        Raises:
            FileNotFoundError: If the audio file doesn't exist
            Exception: If transcription fails
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        print(f"Transcribing: {audio_path.name}")
        print("This may take a while depending on the file length...")
        
        try:
            # Transcribe the audio
            result = self.model.transcribe(
                str(audio_path),
                language=language,
                verbose=False
            )
            
            # Save the transcription
            output_path = self._get_output_path(audio_path)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                
            print(f"\nTranscription completed successfully:")
            print(f"- Output file: {output_path}")
            print(f"- Detected language: {result['language']}")
            print(f"- Number of segments: {len(result['segments'])}")
            
            return result
            
        except Exception as e:
            raise Exception(f"Failed to transcribe audio: {str(e)}")
            
    def _get_output_path(self, audio_path: Path) -> Path:
        """
        Get the path where the transcription should be saved.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Path where the transcription JSON should be saved
        """
        # Use the same name as the audio file but with .json extension
        output_name = audio_path.stem + ".transcription.json"
        
        if self.output_dir:
            # Use specified output directory
            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir / output_name
        else:
            # Save alongside the audio file
            return audio_path.parent / output_name 