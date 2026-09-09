#!/bin/bash
#SBATCH --job-name=abl_nowd
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_no_warmup_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_no_warmup_%j.err
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# ABLATION: ViT-RGTS v2 without warmup (direct cosine from full LR)
# Tests: Is warmup critical for ViT from-scratch stability?
# NOTE: This uses auto augmentation so warmup is in the optimizer path
# Baseline: 07_vit_rgts_v2_4reg.slurm (warmup=5 epochs)
# WARNING: May diverge — this is an expected outcome validating warmup necessity
# =============================================================================
export EXPERIMENT_NAME="ablation_no_warmup_4reg"
export CONFIG_FILES="configs/config.yaml configs/baseline_vit_rgts_v2.yaml"
export CLI_OVERRIDES="data.mode=iam arch.num_registers=4 train.lr=5e-4"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
