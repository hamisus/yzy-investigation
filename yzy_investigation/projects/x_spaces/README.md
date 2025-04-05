# X Spaces Tools

A set of tools for working with X/Twitter Spaces, including downloading, transcribing, and summarizing audio content.

## Features

- **Download**: Download X/Twitter Spaces audio
- **Transcribe**: Transcribe audio files using OpenAI's Whisper model
- **Summarize**: Create concise summaries of transcripts using OpenAI models

## Installation

```bash
# From the root of the repository
pip install -e .
```

## Usage

### CLI Commands

The module provides several command-line tools:

#### Download a Space

```bash
python -m yzy_investigation.projects.x_spaces download <URL> [--output-dir OUTPUT_DIR]
```

#### Transcribe an audio file

```bash
python -m yzy_investigation.projects.x_spaces transcribe <AUDIO_FILE> [--model {tiny,base,small,medium,large}] [--language LANGUAGE] [--output-dir OUTPUT_DIR]
```

#### Summarize a transcript

```bash
python -m yzy_investigation.projects.x_spaces summarize <TRANSCRIPT_FILE> [--api-key API_KEY] [--model MODEL] [--chunk-size CHUNK_SIZE] [--max-workers MAX_WORKERS] [--output-dir OUTPUT_DIR]
```

#### Download and transcribe

```bash
python -m yzy_investigation.projects.x_spaces download-transcribe <URL> [--model {tiny,base,small,medium,large}] [--language LANGUAGE] [--output-dir OUTPUT_DIR]
```

#### Download, transcribe, and summarize

```bash
python -m yzy_investigation.projects.x_spaces download-transcribe-summarize <URL> [--whisper-model {tiny,base,small,medium,large}] [--language LANGUAGE] [--api-key API_KEY] [--model MODEL] [--chunk-size CHUNK_SIZE] [--max-workers MAX_WORKERS] [--output-dir OUTPUT_DIR]
```

### Transcript Summarization

The summarizer is designed to process large transcripts in chunks, then combine them into a final cohesive summary:

1. **For each chunk**:
   - It summarizes the main points discussed
   - Captures details relevant to the topic (for example, YzY/4NBT coin)
   - Avoids speculation or adding information not in the transcript

2. **Final summary**:
   - Merges all chunk summaries into a cohesive report
   - Removes redundant information
   - Presents key takeaways in a clear format

#### Options for summarization:

- `--api-key`: OpenAI API key (if not provided, uses OPENAI_API_KEY environment variable)
- `--model`: OpenAI model to use (default: "gpt-4")
- `--chunk-size`: Number of characters per chunk (default: 3000)
- `--max-workers`: Number of parallel workers (default: 1)
- `--output-dir`: Directory to save the summary

## Python API

You can also use the tools programmatically:

```python
from yzy_investigation.projects.x_spaces import SpaceDownloader, SpaceTranscriber, SpaceSummarizer

# Download a Space
downloader = SpaceDownloader(output_dir="downloads")
audio_path = downloader.download_space("https://twitter.com/i/spaces/...")

# Transcribe the Space
transcriber = SpaceTranscriber(model_name="base", output_dir="transcripts")
transcript_data = transcriber.transcribe(audio_path)

# Summarize the transcript
summarizer = SpaceSummarizer(
    api_key="your-api-key",  # Optional if OPENAI_API_KEY is set
    model="gpt-4",
    chunk_size=3000,
    max_workers=1,
    output_dir="summaries"
)
summary = summarizer.summarize_from_file(f"{audio_path.stem}.transcription.json")
```

## Requirements

- Python 3.7+
- OpenAI API key (for summarization)
- FFmpeg (for downloading Spaces)
- Whisper (for transcription)
- tqdm (for progress bars)
- Other dependencies as listed in requirements.txt 