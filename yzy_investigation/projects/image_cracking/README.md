# Image Cracking

A comprehensive system for analyzing images to detect and extract steganographically hidden data.

## Overview

The Image Cracking module provides a set of tools and strategies for:

1. Analyzing images using multiple steganography detection strategies
2. Processing and combining results from various strategies
3. Identifying significant patterns or hidden texts
4. Reconstructing hidden files from the extracted data

The system is designed to automatically detect if any results contain the target string "4NBTf8PfLH4oLFnwf3knv46FY9i5oXjDxffCetXRpump" or the key terms "4NBT" and "silver".

## Usage

### Running the Image Cracking Pipeline

The simplest way to run the image cracking pipeline is:

```bash
python -m yzy_investigation.main image-crack --input-dir ./data/raw/yews
```

This will:
1. Analyze all images in the input directory with all available strategies
2. Show progress bars and estimated time remaining for each step
3. Process the results looking for patterns and hidden data
4. Generate reports and extract any discovered files
5. Produce a summary of the findings

### Command-line Options

* `--input-dir` or `-i`: Directory containing images to analyze (required)
* `--output-dir` or `-o`: Directory to store results (default: results/image_cracking)
* `--keywords` or `-k`: Additional keywords to search for in the analysis
* `--key-numbers` or `-n`: Additional key numbers to use in the analysis
* `--config` or `-c`: Path to configuration file with keywords and key numbers
* `--verbose` or `-v`: Enable verbose logging

### Using a Configuration File

You can store keywords and key numbers in a JSON configuration file:

```json
{
  "keywords": [
    "4NBT",
    "silver",
    "YZY",
    "Tyger",
    "Blake",
    "WilliamBlake",
    "otherKeyword"
  ],
  "key_numbers": [4, 333, 353, 42]
}
```

Then use it with:

```bash
python -m yzy_investigation.main image-crack --input-dir ./data/raw/yews --config ./path/to/keywords.json
```

The default configuration file is located at:
`yzy_investigation/projects/image_cracking/config/keywords.json`

## Strategies

The image cracking system includes the following strategies:

1. **LSB Strategy**: Extracts data hidden in the least significant bits of pixel values
2. **Color Histogram Strategy**: Analyzes color distributions for anomalies
3. **File Signature Strategy**: Looks for embedded file signatures (magic numbers)
4. **Metadata Analysis Strategy**: Examines image metadata for hidden information
5. **Keyword XOR Strategy**: Applies XOR operations with key terms
6. **Shift Cipher Strategy**: Tests various character shift operations
7. **Bit Sequence Strategy**: Analyzes bit patterns in different arrangements
8. **Blake Hash Strategy**: Uses Blake hash functions with William Blake related keys to detect hidden data

### William Blake Connection

The Blake Hash Strategy specifically explores the connection between William Blake (the poet/artist) and Blake hash functions. It applies Blake2b and Blake2s hash functions using:
- Works and poems by William Blake as keys (e.g., "Tyger", "Songs of Innocence")
- Famous phrases from Blake's works (e.g., "fearful symmetry", "burning bright")
- Analysis of specific image regions corresponding to key numbers
- Blake hash verification of potential hidden messages

This strategy is valuable if the steganography technique involves Blake hashes as a verification mechanism or if the encoding is related to William Blake's works.

## Results and Output

The pipeline generates several outputs:

1. **Terminal Progress**: Real-time progress bars showing completion percentage and ETA
2. **Summary Output**: Brief summary at the end showing key findings
3. **Report Files**:
   - `final_report.json`: Complete report of all findings
   - `summary.txt`: Human-readable summary
   - `TARGET_FOUND.txt`: Details if target string was found
4. **Extracted Data**:
   - Reconstructed files in the `reconstructed_files` directory
   - Combined and transformed data in various formats

## Customizing

To add new strategies or customize the pipeline:

1. Create new strategy classes inheriting from `StegStrategy`
2. Add them to the strategies list in `image_crack_cli.py`
3. Update the keywords and key numbers in the configuration file

For more detailed information, see the code documentation in the respective modules. 