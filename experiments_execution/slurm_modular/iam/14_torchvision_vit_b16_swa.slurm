#!/bin/bash
#SBATCH --job-name=vit_b16_swa
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/vit_b16_iam_swa_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/vit_b16_iam_swa_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# TorchVision ViT-B/16 (pretrained, legacy 2-group LR) — IAM — WITH SWA
# =============================================================================
export EXPERIMENT_NAME="torchvision_vit_b16_iam_swa"
export CONFIG_FILES="configs/config.yaml configs/torchvision_vit.yaml"
export CLI_OVERRIDES="data.mode=iam swa.enabled=true swa.start_epoch=110 swa.swa_lr=5e-5"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
