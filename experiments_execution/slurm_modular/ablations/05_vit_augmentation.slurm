#!/bin/bash
#SBATCH --job-name=abl_vit_a
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_vit_aug_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_vit_aug_%j.err
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# ABLATION: ViT-RGTS v2 with strong ViT augmentation instead of moderate CNN aug
# Tests: Does stronger augmentation help or hurt on 6.5K IAM?
# Baseline: 07_vit_rgts_v2_4reg.slurm (augmentation=cnn)
# =============================================================================
export EXPERIMENT_NAME="ablation_vit_augmentation_4reg"
export CONFIG_FILES="configs/config.yaml configs/baseline_vit_rgts_v2.yaml"
export CLI_OVERRIDES="data.mode=iam arch.num_registers=4 train.augmentation=vit"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
