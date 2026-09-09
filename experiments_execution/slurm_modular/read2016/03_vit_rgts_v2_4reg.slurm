#!/bin/bash
#SBATCH --job-name=r16_v2_4reg
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/r16_v2_4reg_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/r16_v2_4reg_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# ViT-RGTS v2 — 4 registers (default) — READ2016 dataset
# Validates register effect generalizes across datasets
# =============================================================================
export EXPERIMENT_NAME="vit_rgts_v2_4reg_read2016"
export CONFIG_FILES="configs/config.yaml configs/baseline_vit_rgts_v2.yaml"
export CLI_OVERRIDES="data.mode=iam data.path=./data/READ2016/processed arch.num_registers=4"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
