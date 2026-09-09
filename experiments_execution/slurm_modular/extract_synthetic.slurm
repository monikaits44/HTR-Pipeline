#!/bin/bash
#SBATCH --job-name=extract_synth
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/extract_synth_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/extract_synth_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
# =============================================================================
# Extract synthetic LMDB (1M images) to IAM-compatible folder format.
# Must complete BEFORE any synthetic training experiments can run.
# =============================================================================
set -euo pipefail
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline
source .venv/bin/activate
export PYTHONUNBUFFERED=1

python scripts/extract_synthetic_to_iam_format.py \
    --lmdb_path /home/woody/iwi5/iwi5369h/projects/synth_htr/synthesized_images_1M.lmdb \
    --output_dir /home/woody/iwi5/iwi5369h/projects/synth_htr/synthetic_processed_lines \
    --val_ratio 0.05 \
    --seed 42

echo "EXTRACTION COMPLETE at $(date)"
