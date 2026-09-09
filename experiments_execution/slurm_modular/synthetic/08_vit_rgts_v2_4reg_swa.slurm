#!/bin/bash
#SBATCH --job-name=v2_4r_sy_s
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/v2_4reg_synth_swa_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/v2_4reg_synth_swa_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# ViT-RGTS v2 — 4 registers — Synthetic pretrain, test IAM — WITH SWA
# =============================================================================
export EXPERIMENT_NAME="vit_rgts_v2_4reg_synthetic_swa"
export CONFIG_FILES="configs/config.yaml configs/baseline_vit_rgts_v2.yaml"
export CLI_OVERRIDES="data.mode=synthetic train.num_epochs=15 arch.num_registers=4 swa.enabled=true swa.start_epoch=11 swa.swa_lr=5e-4"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
