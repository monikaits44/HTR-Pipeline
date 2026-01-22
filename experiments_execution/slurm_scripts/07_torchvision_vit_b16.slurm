#!/bin/bash
#SBATCH --job-name=tv_vit_b16
#SBATCH --output=../logs/pretrained/output_vit_b16_%j.log
#SBATCH --error=../logs/pretrained/error_vit_b16_%j.log
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080

# Activate the virtual environment
source ~/.bashrc
source ../../.venv/bin/activate

# Change to project root directory
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline

# Run TorchVision ViT-B/16 with 4 registers
echo "========================================="
echo "Starting TorchVision ViT-B/16"
echo "Started at: $(date)"
echo "========================================="

/home/hpc/iwi5/iwi5369h/HTR-Pipeline/.venv/bin/python \
    scripts/trainer.py \
    configs/config.yaml \
    configs/torchvision_vit.yaml \
    arch.model_name='vit_b_16' \
    arch.num_registers=4

echo "========================================="
echo "Completed at: $(date)"
echo "========================================="
