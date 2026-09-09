#!/bin/bash
#SBATCH --job-name=ft_trocr_iam
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/ft_trocr_iam_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/ft_trocr_iam_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# TrOCR-base fine-tune (LLRD, all layers trainable) — IAM — NO SWA
# =============================================================================
export EXPERIMENT_NAME="finetune_trocr_base_iam"
export CONFIG_FILES="configs/config.yaml configs/finetune_trocr.yaml"
export CLI_OVERRIDES="data.mode=iam"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
