#!/bin/bash
#SBATCH --job-name=abl_bs16
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_batch16_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_batch16_%j.err
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# ABLATION: ViT-RGTS v2 with batch_size=16, no gradient accumulation
# Tests: Is gradient accumulation (effective BS=16) equivalent to real BS=16?
# Baseline: 07_vit_rgts_v2_4reg.slurm (BS=8 × accum=2)
# =============================================================================
export EXPERIMENT_NAME="ablation_batch16_4reg"
export CONFIG_FILES="configs/config.yaml configs/baseline_vit_rgts_v2.yaml"
export CLI_OVERRIDES="data.mode=iam arch.num_registers=4 train.batch_size=16 train.gradient_accumulation=1"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
