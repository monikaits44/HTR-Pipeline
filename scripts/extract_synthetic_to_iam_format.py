#!/usr/bin/env python3
"""
Extract synthetic images from LMDB to IAM-compatible folder format.

Converts:
    LMDB database (pickled {image: PNG_bytes, text: str} entries)
To:
    IAM-format directory structure:
        output_dir/
            train/
                gt.txt                  (image_rel_path transcription per line)
                0000/                   (shard subdirectory, 10K images each)
                    synth_0000000000.png
                    ...
                0001/
                    ...
            val/
                gt.txt
                0000/
                    synth_0000000000.png
                    ...

This makes synthetic data directly usable by HTRDataset (same format as IAM processed_lines).
The gt.txt format matches IAM: "image_name transcription" per line.

Subdirectory sharding (10K files per shard) prevents filesystem performance issues
on HPC parallel filesystems (Lustre/GPFS) with large file counts.

Memory Safety:
    - Uses LMDB cursor iteration (one entry at a time, no bulk loading)
    - Hash-based deterministic train/val split (no pre-computation needed)
    - Peak memory: <100 MB regardless of dataset size
    - Estimated output: ~15-25 GB for 1M grayscale images on disk

Usage:
    python scripts/extract_synthetic_to_iam_format.py \\
        --lmdb_path /home/woody/iwi5/iwi5369h/projects/synth_htr/synthesized_images_1M.lmdb \\
        --output_dir /home/woody/iwi5/iwi5369h/projects/synth_htr/synthetic_processed_lines \\
        --val_ratio 0.05 \\
        --seed 42
"""

import os
import sys
import lmdb
import pickle
import io
import argparse
import hashlib
from PIL import Image
from tqdm import tqdm


def get_split(key_bytes, val_ratio=0.05, seed=42):
    """
    Deterministically assign an LMDB entry to train or val split using hash.

    Uses MD5 hash of (key + seed) to produce a uniform distribution.
    This avoids needing to pre-load all keys or shuffle a large index array.

    Args:
        key_bytes: LMDB key as bytes
        val_ratio: Fraction of entries for validation (0.0 to 1.0)
        seed: Integer seed for reproducibility

    Returns:
        'val' or 'train'
    """
    h = hashlib.md5(key_bytes + seed.to_bytes(4, 'big')).digest()
    val_threshold = int(val_ratio * 10000)
    return 'val' if (int.from_bytes(h[:4], 'big') % 10000) < val_threshold else 'train'


def extract_lmdb_to_iam_format(lmdb_path, output_dir, val_ratio=0.05, seed=42):
    """
    Extract LMDB synthetic data to IAM-compatible format with train/val split.

    Images are converted to grayscale PNGs (matching IAM format).
    Files are organized into shard subdirectories (10K images per shard)
    to avoid filesystem performance issues with hundreds of thousands of files.

    Args:
        lmdb_path: Path to LMDB database
        output_dir: Output directory (will contain train/ and val/ subdirs)
        val_ratio: Fraction of data for validation (default: 5%)
        seed: Random seed for deterministic, reproducible split
    """
    # Validate input
    if not os.path.exists(lmdb_path):
        print(f"ERROR: LMDB path does not exist: {lmdb_path}")
        sys.exit(1)

    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Open LMDB (read-only, no locking for safe concurrent access)
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    with env.begin(write=False) as txn:
        total_entries = txn.stat()['entries']

    expected_train = int(total_entries * (1 - val_ratio))
    expected_val = int(total_entries * val_ratio)

    print("=" * 70)
    print("SYNTHETIC LMDB → IAM FORMAT EXTRACTION")
    print("=" * 70)
    print(f"  LMDB path:      {lmdb_path}")
    print(f"  Output dir:     {output_dir}")
    print(f"  Total entries:  {total_entries:,}")
    print(f"  Val ratio:      {val_ratio} (seed={seed})")
    print(f"  Expected split: ~{expected_train:,} train / ~{expected_val:,} val")
    print(f"  Shard size:     10,000 files per subdirectory")

    # Estimate disk usage (~15KB average per grayscale PNG)
    estimated_gb = total_entries * 15 / (1024 ** 3)
    print(f"  Est. disk:      ~{estimated_gb:.1f} GB")
    print("=" * 70)

    # Open gt.txt files for streaming writes (no buffering large lists)
    train_gt = open(os.path.join(train_dir, 'gt.txt'), 'w')
    val_gt = open(os.path.join(val_dir, 'gt.txt'), 'w')

    train_count = 0
    val_count = 0
    failed = 0
    created_shards = set()  # Track created shard dirs to minimize os.makedirs calls

    # Single-pass cursor iteration — memory-safe
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key_bytes, value_bytes in tqdm(cursor, total=total_entries, desc="Extracting"):
            try:
                entry = pickle.loads(value_bytes)
                img_bytes = entry['image']
                text = entry['text'].strip()

                if not text:
                    failed += 1
                    continue

                # Deterministic train/val assignment via hash
                split = get_split(key_bytes, val_ratio, seed)

                if split == 'val':
                    count = val_count
                    split_dir = val_dir
                    gt_file = val_gt
                    val_count += 1
                else:
                    count = train_count
                    split_dir = train_dir
                    gt_file = train_gt
                    train_count += 1

                # Shard into subdirectories (10K files each)
                shard = f'{count // 10000:04d}'
                shard_key = (split, shard)
                if shard_key not in created_shards:
                    os.makedirs(os.path.join(split_dir, shard), exist_ok=True)
                    created_shards.add(shard_key)

                img_name = f'synth_{count:010d}'
                img_rel_path = f'{shard}/{img_name}'
                img_abs_path = os.path.join(split_dir, shard, f'{img_name}.png')

                # Convert to grayscale (matches IAM format) and save
                img = Image.open(io.BytesIO(img_bytes)).convert('L')
                img.save(img_abs_path, format='PNG')

                # Write gt line: "relative_path transcription" (IAM gt.txt format)
                gt_file.write(f'{img_rel_path} {text}\n')

            except Exception as e:
                failed += 1
                if failed <= 10:
                    print(f"\n  Warning (entry {failed}): {e}")
                elif failed == 11:
                    print(f"\n  (suppressing further warnings...)")

    train_gt.close()
    val_gt.close()
    env.close()

    # Summary
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"  Train images: {train_count:,}")
    print(f"  Val images:   {val_count:,}")
    print(f"  Failed:       {failed}")
    print(f"  Output:       {output_dir}")

    # Verify gt.txt line counts match image counts
    train_gt_lines = sum(1 for _ in open(os.path.join(train_dir, 'gt.txt')))
    val_gt_lines = sum(1 for _ in open(os.path.join(val_dir, 'gt.txt')))
    print(f"\n  Verification:")
    print(f"    train/gt.txt lines: {train_gt_lines:,} (expected {train_count:,})")
    print(f"    val/gt.txt lines:   {val_gt_lines:,} (expected {val_count:,})")

    if train_gt_lines == train_count and val_gt_lines == val_count:
        print(f"    ✓ All counts match")
    else:
        print(f"    ✗ COUNT MISMATCH — check for errors")

    # Character set analysis
    train_chars = set()
    with open(os.path.join(train_dir, 'gt.txt'), 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) >= 2:
                train_chars.update(list(' '.join(parts[1:])))

    val_chars = set()
    with open(os.path.join(val_dir, 'gt.txt'), 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) >= 2:
                val_chars.update(list(' '.join(parts[1:])))

    print(f"\n  Character analysis:")
    print(f"    Train charset: {len(train_chars)} unique characters")
    print(f"    Val charset:   {len(val_chars)} unique characters")
    print(f"    Union:         {len(train_chars | val_chars)} unique characters")

    val_only = val_chars - train_chars
    if val_only:
        print(f"    ⚠ Val-only chars (not in train): {sorted(val_only)}")

    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract LMDB synthetic data to IAM-compatible folder format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default extraction (5%% val split, seed=42)
  python scripts/extract_synthetic_to_iam_format.py

  # Custom split
  python scripts/extract_synthetic_to_iam_format.py --val_ratio 0.10 --seed 123

  # Custom paths
  python scripts/extract_synthetic_to_iam_format.py \\
      --lmdb_path /path/to/my.lmdb \\
      --output_dir /path/to/output
        """
    )
    parser.add_argument(
        '--lmdb_path', type=str,
        default='/home/woody/iwi5/iwi5369h/projects/synth_htr/synthesized_images_1M.lmdb',
        help='Path to source LMDB database'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='/home/woody/iwi5/iwi5369h/projects/synth_htr/synthetic_processed_lines',
        help='Output directory for IAM-format data'
    )
    parser.add_argument(
        '--val_ratio', type=float, default=0.05,
        help='Fraction of data for validation split (default: 0.05 = 5%%)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for deterministic split (default: 42)'
    )

    args = parser.parse_args()

    # Disk space advisory
    print(f"\n⚠️  DISK SPACE ADVISORY:")
    print(f"   This will extract ~1M images as individual PNG files.")
    print(f"   Estimated disk usage: ~15-25 GB")
    print(f"   Target filesystem: {os.path.dirname(args.output_dir)}\n")

    extract_lmdb_to_iam_format(
        args.lmdb_path,
        args.output_dir,
        args.val_ratio,
        args.seed
    )
