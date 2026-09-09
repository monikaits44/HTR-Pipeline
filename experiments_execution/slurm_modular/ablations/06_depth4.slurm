#!/bin/bash
#SBATCH --job-name=abl_d4
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_depth4_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_depth4_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# ABLATION: ViT-RGTS v2 with depth=4 (shallower transformer)
# Tests: Is 6-layer transformer necessary, or does 4 suffice for 6.5K data?
# Baseline: 07_vit_rgts_v2_4reg.slurm (depth=6)
# =============================================================================
export EXPERIMENT_NAME="ablation_depth4_4reg"
export CONFIG_FILES="configs/config.yaml configs/baseline_vit_rgts_v2.yaml"
export CLI_OVERRIDES="data.mode=iam arch.num_registers=4 arch.depth=4"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
