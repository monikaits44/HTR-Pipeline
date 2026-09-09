#!/bin/bash
#SBATCH --job-name=abl_v1_4r
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_vit_rgts_v1_4reg_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_vit_rgts_v1_4reg_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# ABLATION: ViT-RGTS v1 (original, no CNN stem, dim=128, depth=4)
# Tests: v1 vs v2 architecture — validates CNN stem + scale-up contribution
# Baseline: 07_vit_rgts_v2_4reg.slurm (v2)
# =============================================================================
export EXPERIMENT_NAME="ablation_vit_rgts_v1_4reg"
export CONFIG_FILES="configs/config.yaml configs/baseline_vit_rgts.yaml"
export CLI_OVERRIDES="data.mode=iam arch.num_registers=4"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
