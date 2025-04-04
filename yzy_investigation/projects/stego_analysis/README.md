# Steganography Analysis

This project uses Aletheia to analyze steganographic techniques and detect hidden data in images.

## Setup

1. Install Python dependencies:
```
pip install -r requirements.txt
```

2. Install system dependencies:
For Debian/Ubuntu:
```
sudo apt-get install octave octave-image octave-signal octave-nan
sudo apt-get install liboctave-dev imagemagick steghide outguess
```

For macOS:
```
brew install octave
brew install imagemagick steghide
```

## Usage

Basic usage examples:

```python
from src.analyzer import detect_steganography

# Analyze an image for steganography
results = detect_steganography("path/to/image.jpg")
``` 