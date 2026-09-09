#!/bin/bash
#SBATCH --job-name=abl_swa_early
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_swa_early_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_swa_early_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# ABLATION: SWA early start (epoch 75 = 50% of training)
# Tests: Does starting SWA earlier improve generalisation?
# Baseline: 08_vit_rgts_v2_4reg_swa (swa.start_epoch=110)
# =============================================================================
export EXPERIMENT_NAME="ablation_swa_early_4reg"
export CONFIG_FILES="configs/config.yaml configs/baseline_vit_rgts_v2.yaml"
export CLI_OVERRIDES="data.mode=iam arch.num_registers=4 swa.enabled=true swa.start_epoch=75 swa.swa_lr=5e-4"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
