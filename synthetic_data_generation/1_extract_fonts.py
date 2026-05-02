"""
Extract all downloaded font ZIP files
"""

import os
import zipfile
from pathlib import Path
from tqdm import tqdm
import shutil

FONT_DIR = r"E:\Projects\HTR_PR_Lab\HTR-Pipeline\data\synthetic\fonts"
EXTRACTED_DIR = r"E:\Projects\HTR_PR_Lab\HTR-Pipeline\data\synthetic\fonts_extracted"

def extract_zip(zip_path, extract_to):
    """Extract a ZIP file"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False

def main():
    print("="*60)
    print("EXTRACTING FONT ZIP FILES")
    print("="*60)
    
    # Create extraction directory
    os.makedirs(EXTRACTED_DIR, exist_ok=True)
    
    # Get all ZIP files
    zip_files = list(Path(FONT_DIR).glob("*.zip"))
    print(f"Found {len(zip_files)} ZIP files")
    
    # Extract all ZIPs
    print("\nExtracting fonts...")
    success = 0
    failed = 0
    
    for zip_file in tqdm(zip_files, desc="Extracting"):
        if extract_zip(zip_file, EXTRACTED_DIR):
            success += 1
        else:
            failed += 1
    
    # Count extracted font files
    font_extensions = ['.ttf', '.otf', '.TTF', '.OTF']
    font_files = []
    for ext in font_extensions:
        font_files.extend(list(Path(EXTRACTED_DIR).rglob(f"*{ext}")))
    
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully extracted: {success} ZIP files")
    print(f"Failed: {failed} ZIP files")
    print(f"Total font files found: {len(font_files)}")
    print(f"Extracted to: {EXTRACTED_DIR}")

if __name__ == "__main__":
    main()
