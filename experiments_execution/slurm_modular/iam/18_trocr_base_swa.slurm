#!/bin/bash
#SBATCH --job-name=trocr_swa
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/trocr_iam_swa_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/trocr_iam_swa_%j.err
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# TrOCR-base (encoder frozen, CTC head only) — IAM — WITH SWA
# =============================================================================
export EXPERIMENT_NAME="trocr_base_iam_swa"
export CONFIG_FILES="configs/config.yaml configs/trocr.yaml"
export CLI_OVERRIDES="data.mode=iam swa.enabled=true swa.start_epoch=110 swa.swa_lr=5e-5"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
