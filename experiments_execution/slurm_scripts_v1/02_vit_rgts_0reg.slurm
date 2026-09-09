#!/bin/bash
#SBATCH --job-name=vit_rgts_0reg
#SBATCH --output=../logs/vit_rgts_registers/output_0reg_%j.log
#SBATCH --error=../logs/vit_rgts_registers/error_0reg_%j.log
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080

# Activate the virtual environment
source ~/.bashrc
source ../../.venv/bin/activate

# Change to project root directory
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline

# Run ViT-RGTS with 0 registers (standard ViT)
echo "========================================="
echo "Starting ViT-RGTS with 0 Registers"
echo "Started at: $(date)"
echo "========================================="

/home/hpc/iwi5/iwi5369h/HTR-Pipeline/.venv/bin/python \
    scripts/trainer.py \
    configs/config.yaml \
    configs/baseline_vit_rgts.yaml \
    arch.num_registers=0

echo "========================================="
echo "Completed at: $(date)"
echo "========================================="
