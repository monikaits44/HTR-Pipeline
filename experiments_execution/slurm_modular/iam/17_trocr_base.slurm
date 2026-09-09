#!/bin/bash
#SBATCH --job-name=trocr_iam
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/trocr_iam_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/trocr_iam_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# TrOCR-base (encoder frozen, CTC head only) — IAM — NO SWA
# =============================================================================
export EXPERIMENT_NAME="trocr_base_iam"
export CONFIG_FILES="configs/config.yaml configs/trocr.yaml"
export CLI_OVERRIDES="data.mode=iam"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
