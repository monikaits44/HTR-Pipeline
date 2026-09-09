#!/usr/bin/env python3
"""
Prepare the READ2016 line-level dataset into the repo's standard HTRDataset
format, identical to data/IAM/processed_lines:

    data/READ2016/processed/
        train/gt.txt   +  train/<img_id>.png
        val/gt.txt     +  val/<img_id>.png
        test/gt.txt    +  test/<img_id>.png

Source (downloaded from the HTR-VT dataset drive, READ_lines.zip):
    data/READ2016/lines/<split>_<idx>.jpeg  (+ .txt)  +  labels.pkl

Key conversion detail
---------------------
READ2016 images are RGB JPEGs.  The repo's utils.preprocessing.load_image does
`1 - image/255`, which only works for 2-D uint8 grayscale (IAM PNGs).  Feeding
RGB through skimage.rgb2gray yields floats in [0,1] and destroys contrast.
We therefore convert every image to single-channel 8-bit grayscale PNG here so
the existing, unmodified load_image / HTRDataset pipeline works correctly.

Splits: labels.pkl uses {train, valid, test}; we map valid -> val to match the
repo convention.

Usage:
    python scripts/prepare_read2016.py
"""
import os
import pickle
import sys

from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT, "data", "READ2016", "lines")
OUT_DIR = os.path.join(ROOT, "data", "READ2016", "processed")
LABELS_PKL = os.path.join(SRC_DIR, "labels.pkl")

# labels.pkl split name -> repo split folder name
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}


def main():
    if not os.path.isfile(LABELS_PKL):
        sys.exit(f"labels.pkl not found at {LABELS_PKL}. "
                 f"Did you unzip READ_lines.zip into data/READ2016/lines/ ?")

    with open(LABELS_PKL, "rb") as f:
        labels = pickle.load(f)
    ground_truth = labels["ground_truth"]
    charset = labels.get("charset", None)

    total = 0
    for src_split, out_split in SPLIT_MAP.items():
        split_dir = os.path.join(OUT_DIR, out_split)
        os.makedirs(split_dir, exist_ok=True)

        entries = ground_truth[src_split]
        gt_lines = []
        n = 0
        for img_name, meta in sorted(entries.items()):
            text = meta["text"]
            if text is None or len(text.strip()) == 0:
                continue  # skip empty transcriptions (CTC needs non-empty labels)

            img_id = os.path.splitext(img_name)[0]            # e.g. train_0
            src_path = os.path.join(SRC_DIR, img_name)        # .jpeg
            if not os.path.isfile(src_path):
                # some entries may use .png extension in the key
                alt = os.path.join(SRC_DIR, img_id + ".jpeg")
                src_path = alt if os.path.isfile(alt) else src_path
            dst_path = os.path.join(split_dir, img_id + ".png")

            # Convert RGB JPEG -> 8-bit grayscale PNG (single channel)
            with Image.open(src_path) as im:
                im.convert("L").save(dst_path, format="PNG")

            gt_lines.append(f"{img_id} {text}")
            n += 1
            if n % 1000 == 0:
                print(f"  [{out_split}] {n} images...", flush=True)

        with open(os.path.join(split_dir, "gt.txt"), "w") as f:
            f.write("\n".join(gt_lines) + "\n")

        print(f"[{out_split}] wrote {n} lines -> {split_dir}/gt.txt", flush=True)
        total += n

    if charset is not None:
        print(f"\nSource charset ({len(charset)} chars): {''.join(charset)}")
    print(f"\nDone. {total} images converted into {OUT_DIR}")
    print("Train the model with:  data.path: ./data/READ2016/processed")


if __name__ == "__main__":
    main()
