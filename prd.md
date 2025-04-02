1. Project Overview
Project Name: YzY Investigation
Primary Goal: Create a single Python-based repository that encapsulates multiple investigative tools and mini-projects for exploring the rumoured 4NBT (aka yzY) meme coin, said to be affiliated with Kanye West.

These “tools” include (but are not limited to):

Puzzle Cracking (cryptography, steganography, hidden messages in text/images)

Web Scraping (e.g., scraping data from yews.news)

Lyrics Analysis (scraping and NLP-based analysis of Kanye’s lyrics and interviews)

Additional Investigations (any other sub-projects that may emerge)

The repository is intentionally modular and extensible, so new mini-projects or “plugins” can be added without refactoring existing code.

2. Stakeholders & Use Cases
Investigation Team Members: Individuals or small groups who will collaborate on puzzle solving, data scraping, or text analysis.

Future Contributors: Developers or AI agents who want to extend or maintain the codebase (add new ciphers, new data sources, advanced analytics, etc.).

Project Manager / Lead: Oversees high-level direction, ensures deliverables meet quality and timeline expectations.

Key Use Cases:

Puzzle Cracking: A user can point the system to a folder of puzzle files (text, images) and run a suite of strategies (steganography extraction, cryptographic decoders, brute-force attempts, etc.).

Web Scraping: A user or automated pipeline can fetch images/text from yews.news (or other relevant sites), storing them for puzzle analysis or long-term reference.

Lyrics Analysis: A user runs a pipeline to collect Kanye West’s lyrics (from an API or local repository), then performs text analysis (NLP, keyword extraction, sentiment, etc.) to glean potential clues.

Extensibility: A contributor can create a new sub-project—e.g., “Audio Analysis” or “Image Classification”—by adding a folder under projects/ and hooking into the shared infrastructure.

3. Scope & Non-Scope
In Scope:

Shared Python codebase with a well-defined folder structure.

Modular sub-projects for puzzle solving, scraping, lyrics analysis, etc.

Common managers/utilities (logging, data IO, concurrency) in a core/ module.

Standard methods to store, retrieve, and log results in consistent formats.

Documentation & minimal testing to ensure sub-projects run reliably.

Out of Scope (For now):

Full-scale deployment systems (e.g., Docker images, K8s) – unless specifically added later.

Production-grade data pipelines for massive-scale scraping or ML.

Security audits or advanced DevOps pipelines.

Any guaranteed solution to puzzles – the framework facilitates puzzle cracking, but success depends on the puzzle content.

4. Detailed Requirements
Below are the core requirements that define how another AI agent should set up and run this repository.

4.1 Repository & Folder Structure
Top-Level

A single GitHub repository named, for instance, yzY_investigation/.

README.md containing basic installation steps, usage instructions, and an overview of sub-projects.

requirements.txt or poetry/conda environment file listing dependencies.

Python Package Layout:


yzy-investigation/           # Current root directory and git repo
├── yzy_investigation/       # Main Python package
│   ├── __init__.py
│   ├── main.py              # Optional "umbrella" CLI or orchestrator
│   ├── config.py            # Global config & constants
│   ├── core/                # Shared infrastructure
│   │   ├── base_pipeline.py # Abstract base classes for sub-project pipelines
│   │   ├── log_manager.py   # Centralised logging utility
│   │   ├── data_manager.py  # Common data IO or DB interactions
│   │   └── ...
│   ├── projects/            # Each sub-project as its own folder
│   │   ├── puzzle_cracking/
│   │   ├── web_scraper/
│   │   ├── lyrics_analysis/
│   │   └── ...
│   ├── tests/               # Basic unit tests
│   └── ...
├── data/                    # Local data folder for raw/processed artifacts
│   ├── raw/
│   ├── processed/
│   └── ...
├── results/                 # Default folder for logs, final puzzle solutions, etc.
├── docs/                    # Any design documents, usage guides
├── requirements.txt         # Dependencies
├── README.md
└── LICENSE


Sub-Project Structure
Each sub-project (under projects/) must contain:

A minimal __init__.py (so it’s recognised as a Python module).

One or more Python scripts or sub-folders implementing the core logic.

(Optional) A runner.py or main script that orchestrates that sub-project’s tasks.

Common Modules

core/base_pipeline.py: Provides an abstract pipeline class (BasePipeline) which sub-projects can extend.

core/log_manager.py: Standardised approach to logging actions, successes, and errors (in JSON or a DB).

core/data_manager.py: Shared utilities for reading/writing data from data/, or connecting to a local DB if needed.

main.py (Optional)

A single CLI entry point that can route to different sub-project “commands.”

For example, python main.py puzzle --input some_folder or python main.py scraper --site yews.news.

4.2 Installation & Environment
Python Version: 3.8+ recommended.

Dependency Management:

Minimal approach: pip install -r requirements.txt

Or use Poetry/Conda for environment creation if the project grows in complexity.

Expected Libraries (examples, not exhaustive):

requests, beautifulsoup4 (scraping)

pillow, stegano, cryptography, pytesseract, numpy (puzzle image analysis & cryptography)

spacy, nltk, or transformers (lyrics NLP analysis)

Possibly selenium for more advanced web automation.

4.3 Core Functional Requirements
Puzzle-Cracking Sub-Project

A puzzle_cracking/ folder containing:

“Strategies” for steganography, cryptographic ciphers, hidden data extraction.

A runner or pipeline that can iterate over text/image files, apply each strategy, and log results.

Must ingest a folder path (e.g. data/raw/puzzles) and produce logs in results/ or a DB.

Web-Scraper Sub-Project

A web_scraper/ folder containing scraping logic (e.g., yews_scraper.py).

Should be able to fetch HTML content, parse relevant text or images, and store them in data/raw/ or a subdirectory.

Logging or error handling to record successful fetches or failures.

Lyrics Analysis Sub-Project

A lyrics_analysis/ folder with scripts to fetch, store, and parse Kanye’s lyrics.

Possibly includes code for text analytics (sentiment, frequency counts, named entity recognition).

Outputs can be stored in data/processed/lyrics or a relevant subfolder.

Logging & Results

System must store both intermediate and final outcomes in a structured format.

JSON logs are acceptable for minimal overhead; for more complexity, a local database (SQLite) can be used.

The project’s log_manager should unify how logs are captured.

CLI / Orchestration

An optional unified CLI in main.py that sub-projects can tie into.

Alternatively, each sub-project can maintain its own runner script.

Extensibility & Plugin-Like Architecture

Each new sub-project or tool should have minimal friction: simply create a new folder under projects/, add the code, and optionally connect it to main.py or share logic in core/ as needed.

5. Non-Functional Requirements
Maintainability: Code must be reasonably documented (docstrings, comments) so new collaborators can understand and extend it.

Scalability: The architecture should allow concurrency or distributed processing if the data size grows (though not necessarily implemented from day one).

Testability: Include at least a few basic tests in the tests/ folder to demonstrate how sub-projects can be tested.

Version Control: The entire codebase should be in a Git-based repository (e.g., on GitHub) for collaborative development, branching, and pull requests.

6. Constraints & Assumptions
Operating System: Linux, macOS, and Windows are all plausible; Python code should be cross-platform.

External Dependencies: Scraping certain sites (like yews.news) may have rate limits or require an API if direct HTML scraping is not possible.

Data Licensing: Lyrics, interview text, or images may have copyright restrictions. Must ensure usage is for “fair use” or research.

Time Constraint: No strict deadline is specified, but the project should be modular enough that new tools can be added quickly.

7. Milestones & Timeline
M1: Repository Initialization

Create the yzY_investigation/ repo.

Set up the basic folder structure, requirements.txt, README.md.

M2: Core Modules

Implement core/base_pipeline.py, core/log_manager.py, core/data_manager.py.

Provide minimal tests to confirm they function.

M3: Puzzle Cracking Sub-Project

Add puzzle_cracking/ folder with at least one or two example strategies (e.g., base64 decoding, Caesar shift).

Confirm it runs via the CLI or a local script.

M4: Web Scraper Sub-Project

Create web_scraper/ subfolder with a yews_scraper.py stub.

Implement minimal logic to fetch data from yews.news and store in data/raw/scrapes.

M5: Lyrics Analysis Sub-Project

Create lyrics_analysis/ with a simple fetch or parse routine for Kanye lyrics.

Implement a basic analysis function (word count, sentiment, or anything useful).

M6: Documentation & Enhancement

Update README.md with instructions on how to run each sub-project.

Expand tests if needed.

Provide usage examples.

8. Acceptance Criteria
Repository Setup:

The repository can be cloned and set up (with pip install -r requirements.txt) without errors.

A user can run a minimal test suite (pytest, or any chosen framework).

Puzzle-Cracking:

Running the puzzle sub-project on a folder of text or images produces logs with success/failure for each strategy.

Scraper:

The web scraper sub-project, when pointed at yews.news, can retrieve some HTML or images, storing them in data/.

Lyrics Analysis:

A script in lyrics_analysis/ can process local lyrics data and generate a short textual output (e.g., a word frequency distribution).

Extensibility:

Another developer (or AI agent) can add a new sub-project folder under projects/ and run it independently or integrate with main.py without major refactoring.

Documentation:

README.md or docs in docs/ describe installation, usage, and extension guidelines.

9. Future Enhancements
Advanced Cryptanalysis: Add ciphers like Vigenère, substitution, or XOR with brute force using wordlists.

Image/Audio Forensics: Expand puzzle-cracking to handle advanced steganalysis on audio files or more complex image manipulations.

NLP Pipelines: Incorporate machine-learning-based text classification or topic modeling for deeper interview/lyric analysis.

Database Integration: Switch from JSON logs to a robust SQL or NoSQL database if data volume and complexity grow.

GUI/Frontend: Build a simple web interface to manage or visualize puzzle-cracking, scraping status, or lyric analytics.

Distributed/Cloud: Dockerize the entire solution, or integrate with Celery / Ray for distributed tasks if the workload becomes large.

10. Final Notes
This PRD ensures another AI agent (or developer) can:

Create the repository with the described folder structure.

Install necessary dependencies.

Populate each sub-project with the indicated stubs and example scripts.

Run each sub-project (or the “umbrella” main.py) to demonstrate basic functionality.