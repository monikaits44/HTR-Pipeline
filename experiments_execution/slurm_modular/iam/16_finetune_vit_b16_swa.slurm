#!/bin/bash
#SBATCH --job-name=ft_vit_swa
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/ft_vit_b16_iam_swa_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/ft_vit_b16_iam_swa_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# TorchVision ViT-B/16 fine-tune (2-group differential LR) — IAM — WITH SWA
# =============================================================================
export EXPERIMENT_NAME="finetune_vit_b16_iam_swa"
export CONFIG_FILES="configs/config.yaml configs/finetune_torchvision_vit.yaml"
export CLI_OVERRIDES="data.mode=iam swa.enabled=true swa.start_epoch=110 swa.swa_lr=5e-6"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
