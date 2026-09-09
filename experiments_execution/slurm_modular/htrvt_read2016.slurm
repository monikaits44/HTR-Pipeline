#!/bin/bash
#SBATCH --job-name=htrvt_read2016_r4
#SBATCH --output=../logs/htrvt/output_r4_%j.log
#SBATCH --error=../logs/htrvt/error_r4_%j.log
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080

# =============================================================================
# HTR-VT + Registers (R=4) + 2-D grid  —  READ2016
# Combines HTR-VT (CNN stem + ViT + CTC + span-mask), register tokens, and a
# thin 2-D token grid so we can extract Beyond-Memorization-style attention.
# =============================================================================

cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline
source ~/.bashrc
source /home/hpc/iwi5/iwi5369h/HTR-Pipeline/.venv/bin/activate

mkdir -p experiments_execution/logs/htrvt

echo "========================================="
echo "HTR-VT + R=4 registers + 2-D grid — READ2016"
echo "Started at: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "========================================="

/home/hpc/iwi5/iwi5369h/HTR-Pipeline/.venv/bin/python \
    scripts/trainer.py \
    configs/htrvt_read2016_r4.yaml \
    arch.num_registers=4

echo "========================================="
echo "Completed at: $(date)"
echo "========================================="
