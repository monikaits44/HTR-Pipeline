#!/bin/bash
#SBATCH --job-name=vit_rgts_v2_4reg
#SBATCH --output=../logs/vit_rgts_v2/output_4reg_%j.log
#SBATCH --error=../logs/vit_rgts_v2/error_4reg_%j.log
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080

# ViT-RGTS v2 with CNN Stem — Register sweep: 4 registers (default)

source ~/.bashrc
source ../../.venv/bin/activate
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline

echo "========================================="
echo "ViT-RGTS v2 (CNN Stem) — 4 Registers"
echo "Started at: $(date)"
echo "========================================="

/home/hpc/iwi5/iwi5369h/HTR-Pipeline/.venv/bin/python \
    scripts/trainer.py \
    configs/baseline_vit_rgts_v2.yaml \
    arch.num_registers=4

echo "========================================="
echo "Completed at: $(date)"
echo "========================================="
