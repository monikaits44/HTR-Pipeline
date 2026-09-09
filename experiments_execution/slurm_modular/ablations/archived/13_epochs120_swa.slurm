#!/bin/bash
#SBATCH --job-name=abl_ep120s
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_epochs120_swa_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_epochs120_swa_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# ABLATION: ViT-RGTS v2 with 120 epochs + SWA at epoch 90
# Tests: SWA benefit with longer training
# Baseline: 12_epochs120.slurm (no SWA)
# =============================================================================
export EXPERIMENT_NAME="ablation_epochs120_swa_4reg"
export CONFIG_FILES="configs/config.yaml configs/baseline_vit_rgts_v2.yaml"
export CLI_OVERRIDES="data.mode=iam arch.num_registers=4 train.num_epochs=120 swa.enabled=true swa.start_epoch=90 swa.swa_lr=5e-4"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
