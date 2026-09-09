#!/bin/bash
#SBATCH --job-name=seed_456
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/seed_456_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/seed_456_%j.err
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# SEED VARIATION: ViT-RGTS v2 4reg with seed=456
# Tests: Reproducibility and variance across seeds
# Baseline: iam/07_vit_rgts_v2_4reg (seed=42)
# =============================================================================
export EXPERIMENT_NAME="vit_rgts_v2_4reg_seed456"
export CONFIG_FILES="configs/config.yaml configs/baseline_vit_rgts_v2.yaml"
export CLI_OVERRIDES="data.mode=iam arch.num_registers=4 seed=456"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
