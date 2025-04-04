from setuptools import setup, find_packages

setup(
    name="yzy_investigation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "stegano>=0.9.12",
        "cryptography>=41.0.0",
        "pytesseract>=0.3.10",
        "spacy>=3.7.0",
        "nltk>=3.8.1",
        "transformers>=4.36.0",
        "playwright>=1.41.0",
        "aiohttp>=3.9.0",
        # X Spaces dependencies
        "yt-dlp>=2024.3.10",
        "openai-whisper>=20231117",
        "ffmpeg-python>=0.2.0",
        "tqdm>=4.66.0",
    ],
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'x-spaces=yzy_investigation.projects.x_spaces.cli:main',
        ],
    },
) 