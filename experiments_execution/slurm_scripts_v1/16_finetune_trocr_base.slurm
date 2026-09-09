#!/bin/bash
#SBATCH --job-name=ft_trocr
#SBATCH --output=../logs/pretrained/output_ft_trocr_%j.log
#SBATCH --error=../logs/pretrained/error_ft_trocr_%j.log
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080

# Fine-tune TrOCR Base Handwritten with LLRD (all layers trainable)
# Strategy: Layer-wise LR decay (0.9×/layer), no gradual unfreezing

source ~/.bashrc
source ../../.venv/bin/activate

cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline

# TrOCR model should be cached locally
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export NO_ALBUMENTATIONS_UPDATE=1
export NLTK_DATA=/home/hpc/iwi5/iwi5369h/nltk_data

echo "========================================="
echo "Fine-tuning TrOCR Base (LLRD)"
echo "Started at: $(date)"
echo "========================================="

/home/hpc/iwi5/iwi5369h/HTR-Pipeline/.venv/bin/python \
    scripts/trainer.py \
    configs/config.yaml \
    configs/finetune_trocr.yaml

echo "========================================="
echo "Completed at: $(date)"
echo "========================================="
