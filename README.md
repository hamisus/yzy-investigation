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

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e .
```

## Usage

The project includes a command-line interface for running various tools. Here are some examples:

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
python -m yzy_investigation.main image-crack --input-dir "data/raw/yews/2025-03-27"
```

### View Available Commands

```bash
python -m yzy_investigation.main --help
```

## Code Examples

Each sub-project can also be used programmatically:

### Puzzle Cracking
```python
from yzy_investigation.projects.puzzle_cracking import PuzzleCracker

cracker = PuzzleCracker(input_path="data/raw/puzzles")
results = cracker.run()
```

### Web Scraping
```python
from yzy_investigation.projects.web_scraper import YewsScraper

scraper = YewsScraper()
results = scraper.run()
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