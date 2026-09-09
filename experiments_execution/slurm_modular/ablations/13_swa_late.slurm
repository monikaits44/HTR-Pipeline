#!/bin/bash
#SBATCH --job-name=abl_swa_late
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_swa_late_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_swa_late_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# ABLATION: SWA late start (epoch 135 = 90% of training)
# Tests: Does starting SWA later (only final 10%) still help?
# Baseline: 08_vit_rgts_v2_4reg_swa (swa.start_epoch=110)
# =============================================================================
export EXPERIMENT_NAME="ablation_swa_late_4reg"
export CONFIG_FILES="configs/config.yaml configs/baseline_vit_rgts_v2.yaml"
export CLI_OVERRIDES="data.mode=iam arch.num_registers=4 swa.enabled=true swa.start_epoch=135 swa.swa_lr=5e-4"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
