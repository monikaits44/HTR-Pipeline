# Synthetic Handwritten Text Data Generation Pipeline

## Table of Contents
- [Overview](#overview)
- [Environment & Storage](#environment--storage)
- [Pipeline Architecture](#pipeline-architecture)
- [Step-by-Step Process](#step-by-step-process)
  - [Step 0: Font Crawling](#step-0-font-crawling)
  - [Step 1: Font Download](#step-1-font-download)
  - [Step 1b: Font Extraction](#step-1b-font-extraction)
  - [Step 2: Text Generation](#step-2-text-generation)
  - [Step 3: IAM Length Distribution](#step-3-iam-length-distribution)
  - [Step 4: Text Preprocessing](#step-4-text-preprocessing)
  - [Step 4-1: Character Filtering](#step-4-1-character-filtering)
  - [Step 5-0: Remove Corrupted Fonts](#step-5-0-remove-corrupted-fonts)
  - [Step 5-1: Font Character Support Check](#step-5-1-font-character-support-check)
  - [Step 5-2: Render Verification](#step-5-2-render-verification)
  - [Step 5: Duplicate Check](#step-5-duplicate-check)
  - [Step 6: Image Rendering + LMDB](#step-6-image-rendering--lmdb)
- [Data Files Reference](#data-files-reference)
- [LMDB Loader](#lmdb-loader)
- [Code Issues Fixed](#code-issues-fixed)
- [Quick Execution Guide](#quick-execution-guide)

---

## Overview

### Purpose
Generate **1 million synthetic handwritten text line images** for training HTR (Handwriting Text Recognition) models. Images are rendered using diverse handwriting fonts on varied text, then stored in an LMDB database for efficient PyTorch DataLoader access.

### Why Synthetic Data?
- **IAM dataset** has only ~10,000 line images — insufficient for deep learning
- Real handwritten data collection is expensive and slow
- Basic augmentation (rotation, scaling) doesn't add new writing styles
- This pipeline uses **11,748 handwriting fonts** to generate diverse writing styles
- Text is matched to IAM's character-length distribution for domain consistency

### Key Numbers
| Metric | Value |
|--------|-------|
| Total fonts crawled | 16,244 |
| Fonts downloaded (ZIPs) | 8,177 |
| Extracted font files (.ttf/.otf) | 11,748 |
| Valid fonts (after corruption check) | 11,748 |
| Fonts supporting full IAM charset | 9,367 |
| Text lines generated | 1,000,000 |
| Lines after filtering | 999,987 |
| Target images | ~999,500 |
| Estimated LMDB size | ~25-30 GB |

---

## Environment & Storage

### HPC Platform
- **Cluster**: FAU TinyGPU (login node: `tinyx`)
- **Scheduler**: SLURM
- **Python**: 3.12, venv at `/home/hpc/iwi5/iwi5369h/HTR-Pipeline/.venv/`

### Storage Layout
| Location | Purpose | Quota |
|----------|---------|-------|
| `/home/hpc/iwi5/iwi5369h/HTR-Pipeline/` | Code repository | 100GB (over quota) |
| `/home/woody/iwi5/iwi5369h/projects/synth_htr/` | **All generated data** | 954GB |
| `/home/hpc/iwi5/iwi5369h/HTR-Pipeline/data/IAM/` | IAM dataset (read-only reference) | — |

All large data files (fonts, text, LMDB) are stored on **woody** due to hpchome being over quota.

### Dependencies
```
Pillow          # Image rendering
fonttools       # Font validation and character map inspection
lmdb            # Lightning Memory-Mapped Database
tqdm            # Progress bars
matplotlib      # Distribution plotting (Step 3 only)
datasets        # HuggingFace datasets (Step 2 CC100 option only)
requests        # Font downloading (Step 1 only)
```

---

## Pipeline Architecture

```
Step 0: Font Crawling (notebook, run locally with browser)
    | font_links_license.csv (16,244 URLs)
    v
Step 1: Font Download
    | 8,177 ZIP files in fonts/
    v
Step 1b: Font Extraction
    | 11,748 .ttf/.otf files in fonts_extracted/
    v
                                                    Step 5-0: Remove Corrupted Fonts
                                                        | 11,748 valid fonts remain
                                                        v
                                                    Step 5-1: Character Support Check
                                                        | missing_characters_fonts.txt (2,381 entries)
                                                        v
                                                    Step 5-2: Render Verification (optional)
                                                        | Visual check of font rendering

Step 2: Text Generation (alternative: synthetic English)
    | cc100_random_subset_1M.txt (1,000,000 lines)
    v
Step 3: IAM Length Distribution Analysis
    | text_length_counts.txt
    v
Step 4: Text Preprocessing (match IAM distribution)
    | cc100_random_subset_1M_modified.txt
    v
Step 4-1: Character Filtering (remove non-IAM chars)
    | cc100_random_subset_1M_modified_filtered.txt (999,987 lines)
    v
Step 5: Duplicate Check
    | (report only - 124 duplicates found, negligible)
    v
Step 6: Image Rendering + LMDB (SLURM job)
    | synthesized_images_1M.lmdb (~25-30 GB)
    v
```

### Data Flow (filename chain)
```
cc100_random_subset_1M.txt
  -> cc100_random_subset_1M_modified.txt
    -> cc100_random_subset_1M_modified_filtered.txt
      -> synthesized_images_1M.lmdb
```

---

## Step-by-Step Process

### Step 0: Font Crawling
**File**: `0. font_crawling_with_lincense.ipynb`
**Purpose**: Scrape handwriting font download links from 1001fonts.com
**Environment**: Requires Selenium + browser (run locally, not on HPC)
**Output**: `font_links_license.csv` — 16,244 font URLs with license type (commercial/personal)

CSV format:
```csv
link,use_type
/download/rustic-roadway-personal-use.zip,personal
/download/priestacy.zip,personal
```

> **Note**: This notebook was already executed. The CSV exists at
> `/home/woody/iwi5/iwi5369h/projects/synth_htr/font_links_license.csv`

---

### Step 1: Font Download
**File**: `1. font_download.py`
**Purpose**: Download font ZIP files from 1001fonts.com
**Input**: `font_links_license.csv`
**Output**: ZIP files in `/home/woody/.../synth_htr/fonts/`

**Key details**:
- Downloads from `https://www.1001fonts.com{link}`
- Retry logic: 3 attempts per font with 2s delay
- Skip logic: skips already-downloaded files
- Saves `failed_downloads.txt` for any failures

**Result**: 8,177 of 16,244 fonts downloaded (remainder may have been removed from the website)

**Run**:
```bash
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline/synthetic_data_generation
source ../.venv/bin/activate
python3 "1. font_download.py"
```

---

### Step 1b: Font Extraction
**File**: `1_extract_fonts.py`
**Purpose**: Extract all downloaded font ZIP files
**Input**: ZIP files in `fonts/`
**Output**: .ttf/.otf files in `fonts_extracted/`

**Key details**:
- Extracts using Python `zipfile` module
- Fonts are extracted flat into `fonts_extracted/` (some in subdirectories from ZIP structure)
- Counts .ttf, .otf, .TTF, .OTF files recursively

**Result**: 11,748 font files extracted (some ZIPs contain multiple font variants)

**Run**:
```bash
python3 1_extract_fonts.py
```

---

### Step 2: Text Generation
Two options available:

#### Option A: CC100/C4 Dataset (not used in current run)
**File**: `2. lines_extraction_from_CC100_multiprocessing.py`
**Purpose**: Extract random English text lines from HuggingFace's C4 dataset
**Requires**: ~40GB dataset download, SLURM job recommended

#### Option B: Synthetic Text (used in current run)
**File**: `2_alternative_text_generation.py`
**Purpose**: Generate synthetic English text from a vocabulary of ~170 common words
**Input**: None (self-contained)
**Output**: `cc100_random_subset_1M.txt` (1,000,000 lines, ~89MB)

**Key details**:
- Vocabulary: 170+ common English words + 10 proper nouns + 8 city names
- Each line: 1-3 randomly generated sentences (3-15 words each)
- 30% chance of ending punctuation (. ! ?)
- 5% chance of proper nouns in sentence body
- Default: `--num_lines=1000000`

**Sample output**:
```
Over walk book up write into grass their or mountain country false us Left me take on ugly sad say Would dog wake must
Other Rome fact star any.
Give high going use on can Tokyo well If have speak run big were listen good
```

**Run**:
```bash
python3 2_alternative_text_generation.py --num_lines 1000000
```

---

### Step 3: IAM Length Distribution
**File**: `3. get_length_distribution_of_IAM.py`
**Purpose**: Analyze character-length distribution of IAM dataset transcriptions
**Input**: IAM ground truth files at `/home/hpc/.../data/IAM/processed_lines/{train,val,test}/gt.txt`
**Output**:
- `text_length_counts.txt` — tab-separated (length, frequency)
- `text_length_distribution.png` — histogram plot

**Key details**:
- Reads all three splits (train, val, test) for complete distribution
- GT format: `image_name transcription` (space-separated, split on first space)
- Uses `matplotlib` with `Agg` backend (headless)

**Result** (excerpt from `text_length_counts.txt`):
```
Text Length     Frequency
1       1
3       3
4       2
5       14
...
```

**Run**:
```bash
python3 "3. get_length_distribution_of_IAM.py"
```

---

### Step 4: Text Preprocessing
**File**: `4. cc100_random_subset_preprocessing_multiprocessing.py`
**Purpose**: Reshape synthetic text to match IAM's character-length distribution
**Input**:
- `cc100_random_subset_1M.txt` (raw text)
- `text_length_counts.txt` (target distribution)
**Output**: `cc100_random_subset_1M_modified.txt` (1,000,000 lines)

**Algorithm**:
1. Build cumulative distribution from IAM length counts
2. For each text line, sample a target length from the distribution
3. If text is longer: randomly crop a substring of target length
4. If text is shorter: repeat/pad text to target length
5. If exact match: keep as-is

**Key details**:
- Uses `ProcessPoolExecutor` for multiprocessing
- Chunk size: 10,000 lines per chunk
- Distribution is rebuilt in each worker process via `initializer`

**Run**:
```bash
python3 "4. cc100_random_subset_preprocessing_multiprocessing.py"
```

---

### Step 4-1: Character Filtering
**File**: `4-1. remove_unkown_characters_multiprocessing.py`
**Purpose**: Remove characters not in the IAM character set
**Input**: `cc100_random_subset_1M_modified.txt`
**Output**: `cc100_random_subset_1M_modified_filtered.txt`

**IAM character set** (79 characters):
```
 !"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
```

**Key details**:
- Uses `multiprocessing.Process` (not Pool) with file-based byte-offset chunking
- Each process writes to a temporary file `{output_file}_{i}`
- Files merged sequentially after all processes complete
- Lines that become empty after filtering are dropped

**Result**: 1,000,000 -> 999,987 lines (13 lines were entirely non-IAM characters)

**Run**:
```bash
python3 "4-1. remove_unkown_characters_multiprocessing.py"
```

---

### Step 5-0: Remove Corrupted Fonts
**File**: `5-0. remove_font_not_loaded.py`
**Purpose**: Remove non-font files and corrupted fonts from `fonts_extracted/`
**Input**: `fonts_extracted/` directory
**Output**: Modified `fonts_extracted/` (corrupted files deleted in-place)

**Key details**:
- Removes files without .ttf/.otf extension (e.g., .woff, .eot, .svg, macOS ._ files)
- Validates each font by attempting `TTFont(file_path)` load
- Uses multiprocessing `Pool` across all CPU cores
- **Destructive**: actually deletes invalid files from disk

**Result**: 11,748 valid fonts remain (removed non-font files like .woff, .eot, macOS resource forks)

**Run**:
```bash
python3 "5-0. remove_font_not_loaded.py"
```

---

### Step 5-1: Font Character Support Check
**File**: `5-1. font_check_support_all_characters.py`
**Purpose**: Identify which fonts don't support all IAM characters
**Input**: `fonts_extracted/` directory
**Output**: `missing_characters_fonts.txt` — fonts with missing characters

**Key details**:
- Checks each font's cmap table for all 79 IAM characters
- Uses multiprocessing `Pool` for parallel checking
- Output format: `FontName.ttf\tMISSING: <chars>` or `FontName.ttf\tLOAD_FAILED`
- **Does NOT delete** fonts — only reports

**Result**: 2,381 fonts with missing characters, 9,367 fully compatible fonts

**Run**:
```bash
python3 "5-1. font_check_support_all_characters.py"
```

---

### Step 5-2: Render Verification
**File**: `5-2.check_text_is_rendered_correctly.py`
**Purpose**: Visual quality check — render the full IAM charset with each valid font
**Input**: `fonts_extracted/`, `missing_characters_fonts.txt`
**Output**: PNG images in `synthesized_images/` (one per font)

**Key details**:
- Filters out fonts listed in `missing_characters_fonts.txt`
- Renders the full character set string at font size 100
- Saves as `{font_filename}.png`
- Records fonts that fail rendering to `skipped_fonts.txt`

**This step is optional** — for manual visual inspection only.

**Run**:
```bash
python3 "5-2.check_text_is_rendered_correctly.py"
```

---

### Step 5: Duplicate Check
**File**: `5. final_check_any_duplicate_lines.py`
**Purpose**: Check for duplicate text lines in the filtered text
**Input**: `cc100_random_subset_1M_modified_filtered.txt`
**Output**: Console report only (no file modification)

**Key details**:
- Chunks the file across CPU cores
- Each chunk finds local duplicates
- Results merged (union of all duplicates)
- Note: only finds duplicates within chunks, not cross-chunk — this is by design for efficiency

**Result**: 124 duplicate lines found (0.012% — negligible)

**Run**:
```bash
python3 "5. final_check_any_duplicate_lines.py"
```

---

### Step 6: Image Rendering + LMDB
**File**: `6. text_rendering_filter_not_support_all_charcters_LMDB.py`
**SLURM script**: `render_1M_images.slurm`
**Purpose**: Render each text line as an image and store in LMDB
**Input**:
- `cc100_random_subset_1M_modified_filtered.txt` (999,987 lines)
- `fonts_extracted/` (11,748 fonts)
**Output**: `synthesized_images_1M.lmdb` (~25-30 GB)

**Algorithm** (per text line):
1. Copy the full font list
2. Randomly pick a font, check if it supports all characters in the text
3. If not, remove that font and try another (up to all fonts exhausted)
4. Once a compatible font is found, render at font size 100
5. Create PIL Image (white background, black text)
6. Convert to PNG bytes
7. Store as `{index:010}` key in LMDB with `pickle.dumps({'image': bytes, 'text': str})`

**Key details**:
- Uses `multiprocessing.Pool` with 8 workers (capped to control memory)
- **Incremental LMDB writes**: batches of 1,000 images committed at a time (avoids OOM)
- LMDB `map_size`: 40GB
- Key format: 10-digit zero-padded index (`f'{i:010}'`)
- Max image dimension: 10,000 pixels (lines exceeding this are skipped)
- Font support check uses `fontTools.ttLib.TTFont` cmap tables
- Images saved as PNG format in LMDB values

**LMDB entry format**:
```python
key = f'{index:010}'.encode('ascii')  # e.g., b'0000000042'
value = pickle.dumps({
    'image': png_bytes,   # PNG image as bytes
    'text': 'the text'    # ground truth transcription
})
```

**SLURM configuration** (`render_1M_images.slurm`):
```bash
#SBATCH --partition=work
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
```

**Run**:
```bash
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline/synthetic_data_generation
sbatch render_1M_images.slurm
```

**Monitor**:
```bash
squeue -u iwi5369h
tail -f /home/woody/iwi5/iwi5369h/projects/synth_htr/render_1M_<JOBID>.log
```

---

## Data Files Reference

All data is stored at `/home/woody/iwi5/iwi5369h/projects/synth_htr/`:

| File | Size | Description |
|------|------|-------------|
| `font_links_license.csv` | 599KB | 16,244 font download URLs + license types |
| `fonts/` | ~7.3GB | 8,177 downloaded font ZIP files |
| `fonts_extracted/` | varies | 11,748 extracted .ttf/.otf font files |
| `cc100_random_subset_1M.txt` | 89MB | 1,000,000 raw synthetic text lines |
| `cc100_random_subset_1M_modified.txt` | 42MB | Text reshaped to IAM length distribution |
| `cc100_random_subset_1M_modified_filtered.txt` | 41MB | 999,987 lines after IAM charset filtering |
| `text_length_counts.txt` | 508B | IAM character-length distribution |
| `text_length_distribution.png` | 42KB | Histogram plot of IAM lengths |
| `missing_characters_fonts.txt` | 109KB | 2,381 fonts lacking IAM charset coverage |
| `synthesized_images_1M.lmdb/` | ~25-30GB | Final LMDB with rendered images |
| `render_1M_*.log` | ~15KB | SLURM job stdout |
| `render_1M_*.err` | ~4.5MB | SLURM job stderr (font warnings) |

---

## LMDB Loader

**File**: `lmdbloader.py`
**Class**: `LMDBDataset(Dataset)`

PyTorch Dataset that reads from the generated LMDB. Used by the main HTR training pipeline.

**Constructor**:
```python
LMDBDataset(
    lmdb_path='/home/woody/.../synthesized_images_1M.lmdb',
    processor=processor,        # HTR image/text processor
    max_tgt_length=128,         # Max token sequence length
    height=96, width=96,        # Resize dimensions
    do_aug=True                 # Apply augmentation
)
```

**Key format**: `f'{idx:010}'.encode('ascii')` — matches Step 6 output
**Image processing**: Converts to grayscale ("L"), applies rescale/normalize
**Label processing**: Tokenizes text, replaces PAD tokens with -100 for loss masking

**Pickle-safe**: Implements `__getstate__`/`__setstate__` to handle LMDB env serialization for DataLoader workers.

---

## Code Issues Fixed

### 1. OOM Kill During LMDB Write (Critical)
**Problem**: Original Step 6 generated ALL ~1M images in memory (`results = list(pool.imap_unordered(...))`) then wrote to LMDB. Used 20.7GB RAM, OOM killed at 12% of LMDB write phase.

**Fix**: Changed to incremental LMDB writes — images are written in batches of 1,000 as they're generated, keeping memory usage bounded.

### 2. LMDB Key Format Mismatch (Critical)
**Problem**: Step 6 wrote keys as `f'{i:07}'` (7-digit) but `lmdbloader.py` reads with `f'{idx:010}'` (10-digit). No keys would ever match.

**Fix**: Aligned Step 6 to use `f'{i:010}'` matching the loader.

### 3. Windows Paths (All Scripts)
**Problem**: All scripts had hardcoded Windows paths `r"E:\Projects\HTR_PR_Lab\HTR-Pipeline\data\synthetic\..."`.

**Fix**: Updated all paths to HPC paths:
- Data: `/home/woody/iwi5/iwi5369h/projects/synth_htr/`
- IAM: `/home/hpc/iwi5/iwi5369h/HTR-Pipeline/data/IAM/processed_lines`

### 4. Filename Consistency
**Problem**: Scripts used inconsistent filenames (`synthetic_text_100K.txt` vs `cc100_random_subset_10M.txt`).

**Fix**: Standardized all filenames to `cc100_random_subset_1M*` pattern flowing through the pipeline.

### 5. Worker Count Control
**Problem**: Step 6 used `cpu_count()` (32 on compute nodes) which amplified per-worker memory.

**Fix**: Capped at `min(cpu_count(), 8)` workers.

---

## Quick Execution Guide

### Prerequisites
```bash
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline/synthetic_data_generation
source ../.venv/bin/activate
```

### Full Pipeline (Steps 1b through 6)

Steps 1-5 run on the login node (< 5 min each):
```bash
# Step 1b: Extract fonts (if not done)
python3 1_extract_fonts.py

# Step 2: Generate text (alternative - instant)
python3 2_alternative_text_generation.py --num_lines 1000000

# Step 3: IAM distribution
python3 "3. get_length_distribution_of_IAM.py"

# Step 4: Match distribution
python3 "4. cc100_random_subset_preprocessing_multiprocessing.py"

# Step 4-1: Filter characters
python3 "4-1. remove_unkown_characters_multiprocessing.py"

# Step 5-0: Validate fonts
python3 "5-0. remove_font_not_loaded.py"

# Step 5-1: Check character support
python3 "5-1. font_check_support_all_characters.py"

# Step 5: Duplicate check (optional)
python3 "5. final_check_any_duplicate_lines.py"

# Step 6: Render images (SLURM - ~2 hours)
sbatch render_1M_images.slurm
```

### Verify Output
```bash
# Check LMDB size
ls -lh /home/woody/iwi5/iwi5369h/projects/synth_htr/synthesized_images_1M.lmdb/

# Quick Python verification
python3 -c "
import lmdb, pickle
env = lmdb.open('/home/woody/iwi5/iwi5369h/projects/synth_htr/synthesized_images_1M.lmdb', readonly=True)
with env.begin() as txn:
    print(f'Total entries: {txn.stat()[\"entries\"]}')
    key = f'{0:010}'.encode('ascii')
    data = pickle.loads(txn.get(key))
    print(f'First text: {data[\"text\"]}')
    print(f'Image size: {len(data[\"image\"])} bytes')
env.close()
"
```
