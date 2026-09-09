#!/bin/bash
#SBATCH --job-name=vit_b_sys
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/vit_b16_synth_swa_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/vit_b16_synth_swa_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# TorchVision ViT-B/16 — Synthetic pretrain, test IAM — WITH SWA
# =============================================================================
export EXPERIMENT_NAME="torchvision_vit_b16_synthetic_swa"
export CONFIG_FILES="configs/config.yaml configs/torchvision_vit.yaml"
export CLI_OVERRIDES="data.mode=synthetic train.num_epochs=15 swa.enabled=true swa.start_epoch=11 swa.swa_lr=5e-5"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
