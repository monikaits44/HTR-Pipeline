#!/bin/bash
#SBATCH --job-name=trocr_base
#SBATCH --output=../logs/pretrained/output_trocr_base_%j.log
#SBATCH --error=../logs/pretrained/error_trocr_base_%j.log
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080

# Activate the virtual environment
source ~/.bashrc
source ../../.venv/bin/activate

# Change to project root directory
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline

# Set offline mode for HuggingFace (use local cached models only)
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Run TrOCR Base Handwritten
echo "========================================="
echo "Starting TrOCR Base Handwritten"
echo "Started at: $(date)"
echo "========================================="

/home/hpc/iwi5/iwi5369h/HTR-Pipeline/.venv/bin/python \
    scripts/trainer.py \
    configs/config.yaml \
    configs/trocr.yaml \
    arch.model_name='microsoft/trocr-base-handwritten'

echo "========================================="
echo "Completed at: $(date)"
echo "========================================="
