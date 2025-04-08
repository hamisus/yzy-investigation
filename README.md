# YzY Investigation

A Python-based toolkit for investigating the 4NBT (yzY) meme coin phenomenon, including puzzle solving, web scraping, X spaces downloading, and Discord management capabilities.

## Project Structure

```
yzy-investigation/
├── yzy_investigation/       # Main Python package
│   ├── core/               # Shared infrastructure
│   ├── projects/           # Investigation sub-projects
│   └── tests/              # Unit tests
├── data/                   # Data storage
│   ├── raw/               # Raw data
│   ├── processed/         # Processed data
│   └── discord/           # Discord data
│       ├── backups/       # Message backups
│       ├── summaries/     # Message summaries
│       └── recaps/        # Daily recaps
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
pip install -r requirements.txt
```

5. Set up Discord bot token:
```bash
# Create .env file in project root
echo "DISCORD_BOT_TOKEN=your_bot_token_here" >> .env
echo "DISCORD_SERVER_ID=your_server_id_here" >> .env
```

Note: For Apple Silicon (M1/M2/M3) Macs, ensure you have macOS 12.3 or later for proper GPU support with PyTorch/MPS.

## Usage

The project includes a command-line interface for running various tools. Here are some examples:

### Discord Manager

The Discord manager provides tools for backing up and analyzing Discord server messages.

```bash
# Step 1: Backup server messages (past 24 hours by default)
python -m yzy_investigation.main discord-backup

# Backup all messages (could take a while!)
python -m yzy_investigation.main discord-backup --all

# Step 2: Summarize messages from the backup
python -m yzy_investigation.main discord-summarize --input-dir ./data/discord/backups

# Step 3: Generate daily recap
python -m yzy_investigation.main discord-daily-recap

# Advanced usage with time ranges
python -m yzy_investigation.main discord-backup --start-time "2024-03-20 00:00:00" --end-time "2024-03-21 00:00:00"
python -m yzy_investigation.main discord-daily-recap --start-time "2024-03-20 00:00:00" --end-time "2024-03-21 00:00:00"
```

### X Spaces Downloader

The X Spaces project allows you to download X/Twitter Spaces audio.

```bash
# Download a Space
python -m yzy_investigation.main x-spaces "https://twitter.com/i/spaces/..."

# Download to a specific directory
python -m yzy_investigation.main x-spaces "https://twitter.com/i/spaces/..." --output-dir ./my_spaces
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
python -m yzy_investigation.main stego-analyze path/to/image.jpg

# Analyze with custom keywords (uses image_cracking project's keywords)
python -m yzy_investigation.main stego-analyze path/to/image.jpg --use-keywords

# Analyze multiple images in a directory
python -m yzy_investigation.main stego-analyze path/to/directory

# Save results to a file
python -m yzy_investigation.main stego-analyze path/to/image.jpg -o results.json

# Specify file extensions for batch analysis
python -m yzy_investigation.main stego-analyze path/to/directory -e jpg png
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