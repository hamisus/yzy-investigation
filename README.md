# YzY Investigation

A Python-based toolkit for investigating the 4NBT (yzY) meme coin phenomenon, including puzzle solving, web scraping, and lyrics analysis capabilities.

## Project Structure

```
yzy-investigation/
├── yzy_investigation/       # Main Python package
│   ├── core/               # Shared infrastructure
│   ├── projects/           # Investigation sub-projects
│   └── tests/              # Unit tests
├── data/                   # Data storage
│   ├── raw/               # Raw data
│   └── processed/         # Processed data
├── results/               # Output and logs
└── docs/                  # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hamisus/yzy-investigation.git
cd yzy-investigation
```

2. Install system dependencies:
```bash
# On macOS
brew install ffmpeg aria2 octave imagemagick steghide pytorch

# On Debian/Ubuntu
sudo apt-get install ffmpeg aria2 octave octave-image octave-signal octave-nan liboctave-dev imagemagick steghide outguess python3-torch
```

3. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install Python dependencies:
```bash
pip install -e .
```

Note: For Apple Silicon (M1/M2/M3) Macs, ensure you have macOS 12.3 or later for proper GPU support with PyTorch/MPS.

## Usage

The project includes a command-line interface for running various tools. Here are some examples:

### X Spaces Downloader & Transcriber

The X Spaces project allows you to download X/Twitter Spaces audio. Transcription to be added later.

```bash
# Download a Space
python -m yzy_investigation.projects.x_spaces.cli download "https://twitter.com/i/spaces/..."

# Download to a specific directory
python -m yzy_investigation.projects.x_spaces.cli download "https://twitter.com/i/spaces/..." --output-dir ./my_spaces

# Transcribe a downloaded Space
python -m yzy_investigation.projects.x_spaces.cli transcribe "path/to/space.m4a"

# Transcribe with a specific Whisper model and language
python -m yzy_investigation.projects.x_spaces.cli transcribe "path/to/space.m4a" \
    --model medium --language en

# Download and transcribe in one go
python -m yzy_investigation.projects.x_spaces.cli download-transcribe "https://twitter.com/i/spaces/..."
```

You can also use the X Spaces tools programmatically:

```python
from yzy_investigation.projects.x_spaces import SpaceDownloader, SpaceTranscriber

# Download a Space
downloader = SpaceDownloader(output_dir="./my_spaces")
audio_path = downloader.download_space("https://twitter.com/i/spaces/...")

# Transcribe the Space
transcriber = SpaceTranscriber(model_name="base")
result = transcriber.transcribe(audio_path, language="en")

# Access transcription results
print(result["text"])  # Full transcription
for segment in result["segments"]:
    print(f"{segment['start']:.1f}s - {segment['end']:.1f}s: {segment['text']}")
```

### Web Scraper (YEWS.news)

```bash
# Basic usage
python -m yzy_investigation.main scrape-yews

# With verbose logging
python -m yzy_investigation.main scrape-yews -v

# Specifying a custom output directory
python -m yzy_investigation.main scrape-yews --output-dir ./custom_output
```

### Image Cracking

```bash
# Basic usage
python -m yzy_investigation.main image-crack --input-dir data/raw/yews/2025-03-27

# With specific date
python -m yzy_investigation.main image-crack --input-dir "data/raw/yews/2025-03-27"

# With custom output directory
python -m yzy_investigation.main image-crack --input-dir data/raw/yews/2025-03-27 --output-dir ./custom_output
```

### Steganography Analysis

The stego analysis project provides lower-level tools for analyzing images using [Aletheia](https://github.com/daniellerch/aletheia) and other steganography detection techniques.

```bash
# Analyze a single image
python -m yzy_investigation.projects.stego_analysis.stego_analyzer analyze path/to/image.jpg

# Analyze with custom keywords (uses image_cracking project's keywords)
python -m yzy_investigation.projects.stego_analysis.stego_analyzer analyze path/to/image.jpg --use-keywords

# Analyze multiple images in a directory
python -m yzy_investigation.projects.stego_analysis.stego_analyzer batch path/to/directory

# Save results to a file
python -m yzy_investigation.projects.stego_analysis.stego_analyzer analyze path/to/image.jpg -o results.json

# Specify file extensions for batch analysis
python -m yzy_investigation.projects.stego_analysis.stego_analyzer batch path/to/directory -e jpg png
```

## Development

- Code formatting: `black .`
- Import sorting: `isort .`
- Type checking: `mypy .`
- Run tests: `pytest`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.