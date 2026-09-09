#!/bin/bash
#SBATCH --job-name=vit_b16_iam
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/vit_b16_iam_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/vit_b16_iam_%j.err
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# TorchVision ViT-B/16 (pretrained, legacy 2-group LR) — IAM — NO SWA
# =============================================================================
export EXPERIMENT_NAME="torchvision_vit_b16_iam"
export CONFIG_FILES="configs/config.yaml configs/torchvision_vit.yaml"
export CLI_OVERRIDES="data.mode=iam"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
