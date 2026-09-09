#!/bin/bash
#SBATCH --job-name=ft_vit_b16
#SBATCH --output=../logs/pretrained/output_ft_vit_b16_%j.log
#SBATCH --error=../logs/pretrained/error_ft_vit_b16_%j.log
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080

# Fine-tune TorchVision ViT-B/16 with LLRD (all layers trainable)
# Strategy: Layer-wise LR decay (0.85×/layer), no gradual unfreezing

source ~/.bashrc
source ../../.venv/bin/activate

cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline

export PYTHONUNBUFFERED=1
export NO_ALBUMENTATIONS_UPDATE=1
export NLTK_DATA=/home/hpc/iwi5/iwi5369h/nltk_data

echo "========================================="
echo "Fine-tuning TorchVision ViT-B/16 (LLRD)"
echo "Started at: $(date)"
echo "========================================="

/home/hpc/iwi5/iwi5369h/HTR-Pipeline/.venv/bin/python \
    scripts/trainer.py \
    configs/config.yaml \
    configs/finetune_torchvision_vit.yaml

echo "========================================="
echo "Completed at: $(date)"
echo "========================================="
