#!/usr/bin/env python3
"""
Generate All Key Figures
=========================
One-command batch runner that invokes every visualization script
in the attention_viz package with sensible defaults.

Usage:
    python scripts/postprocessing/attention_viz/generate_all.py \\
        --device cpu \\
        --dpi 200

Produces all figures needed for the project report.
"""

import subprocess, sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "postprocessing" / "attention_viz"
SAMPLE_IMAGES = [
    "notebook/sample_images/a01-038-12.png",
    "notebook/sample_images/a06-110-08.png",
    "notebook/sample_images/c04-110-00.png",
]
KEY_RUNS = "55 58 63 71"  # Reg-0, Reg-4, Reg-8, Reg-16


def run(cmd: str, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(
        cmd, shell=True, cwd=str(PROJECT_ROOT),
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"  WARNING: {label} exited with code {result.returncode}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--images", nargs="*", default=SAMPLE_IMAGES)
    args = parser.parse_args()

    py = sys.executable

    # 1. Comprehensive comparison (4-panel figure) — one per image
    for img in args.images:
        run(
            f"{py} {SCRIPTS_DIR}/comprehensive_comparison.py "
            f"--image {img} --device {args.device} --dpi {args.dpi}",
            f"Comprehensive Comparison — {Path(img).name}",
        )

    # 2. Attention rollout comparison — one per image
    for img in args.images:
        run(
            f"{py} {SCRIPTS_DIR}/rollout_comparison.py "
            f"--image {img} --runs {KEY_RUNS} --device {args.device} --dpi {args.dpi}",
            f"Rollout Comparison — {Path(img).name}",
        )

    # 3. Head specialization — one per model per image (first image only)
    run(
        f"{py} {SCRIPTS_DIR}/head_specialization.py "
        f"--image {args.images[0]} --runs {KEY_RUNS} --device {args.device} --dpi {args.dpi}",
        "Head Specialization",
    )

    # 4. Layer-wise evolution — one per model + cross-model
    run(
        f"{py} {SCRIPTS_DIR}/layerwise_evolution.py "
        f"--image {args.images[0]} --runs {KEY_RUNS} --device {args.device} --dpi {args.dpi}",
        "Layer-wise Evolution",
    )

    # 5. Attention overlay (sidebyside mode) — one per image
    for img in args.images:
        run(
            f"{py} {SCRIPTS_DIR}/attention_overlay.py "
            f"--image {img} --runs {KEY_RUNS} --mode sidebyside "
            f"--device {args.device} --dpi {args.dpi}",
            f"Attention Overlay — {Path(img).name}",
        )

    # 6. Attention concentration (full sweep, all images)
    run(
        f"{py} {SCRIPTS_DIR}/attention_concentration.py "
        f"--images-dir notebook/sample_images --device {args.device} --dpi {args.dpi}",
        "Attention Concentration Analysis (Full Sweep)",
    )

    print(f"\n{'='*60}")
    print("  ALL DONE — check visualizations/ for outputs")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
