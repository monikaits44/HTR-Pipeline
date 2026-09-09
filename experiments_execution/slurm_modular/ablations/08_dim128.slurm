#!/bin/bash
#SBATCH --job-name=abl_dim128
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_dim128_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_dim128_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# ABLATION: ViT-RGTS v2 with dim=128, heads=4, mlp_dim=512 (smaller model)
# Tests: Does the v2 architecture benefit over v1's smaller capacity?
# Baseline: 07_vit_rgts_v2_4reg.slurm (dim=256, heads=8, mlp_dim=1024)
# =============================================================================
export EXPERIMENT_NAME="ablation_dim128_4reg"
export CONFIG_FILES="configs/config.yaml configs/baseline_vit_rgts_v2.yaml"
export CLI_OVERRIDES="data.mode=iam arch.num_registers=4 arch.dim=128 arch.heads=4 arch.mlp_dim=512"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
