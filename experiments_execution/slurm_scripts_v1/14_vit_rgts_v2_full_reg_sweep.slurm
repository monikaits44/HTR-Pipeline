#!/bin/bash
#SBATCH --job-name=vit_v2_reg%a
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/logs/vit_rgts_v2/output_reg%a_%j.log
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/logs/vit_rgts_v2/error_reg%a_%j.log
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
#SBATCH --array=0-16

# ViT-RGTS v2 with CNN Stem — Full register sweep: 0..16
# Each array task index == num_registers for that job.
#
# Registers covered: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
#
# Submit with:
#   sbatch slurm_scripts/14_vit_rgts_v2_full_reg_sweep.slurm
# Or via the accompanying submit script:
#   bash submit_vit_rgts_v2_full_reg_sweep.sh

NUM_REGISTERS=${SLURM_ARRAY_TASK_ID}

source ~/.bashrc
source /home/hpc/iwi5/iwi5369h/HTR-Pipeline/.venv/bin/activate
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline

echo "========================================="
echo "ViT-RGTS v2 (CNN Stem) — ${NUM_REGISTERS} Registers"
echo "Array task ID : ${SLURM_ARRAY_TASK_ID}"
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Started at    : $(date)"
echo "========================================="

/home/hpc/iwi5/iwi5369h/HTR-Pipeline/.venv/bin/python \
    scripts/trainer.py \
    configs/baseline_vit_rgts_v2.yaml \
    arch.num_registers=${NUM_REGISTERS}

echo "========================================="
echo "Completed at: $(date)"
echo "========================================="
