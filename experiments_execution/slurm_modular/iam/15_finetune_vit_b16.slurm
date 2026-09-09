#!/bin/bash
#SBATCH --job-name=ft_vit_iam
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/ft_vit_b16_iam_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/ft_vit_b16_iam_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# TorchVision ViT-B/16 fine-tune (2-group differential LR) — IAM — NO SWA
# =============================================================================
export EXPERIMENT_NAME="finetune_vit_b16_iam"
export CONFIG_FILES="configs/config.yaml configs/finetune_torchvision_vit.yaml"
export CLI_OVERRIDES="data.mode=iam"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
