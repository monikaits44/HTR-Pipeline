#!/bin/bash
#SBATCH --job-name=abl_dim512
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_dim512_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_dim512_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# ABLATION: ViT-RGTS v2 with dim=512, heads=8, mlp_dim=2048 (larger model)
# Tests: Does scaling up help or overfit on 6.5K data?
# Baseline: 07_vit_rgts_v2_4reg.slurm (dim=256, heads=8, mlp_dim=1024)
# =============================================================================
export EXPERIMENT_NAME="ablation_dim512_4reg"
export CONFIG_FILES="configs/config.yaml configs/baseline_vit_rgts_v2.yaml"
export CLI_OVERRIDES="data.mode=iam arch.num_registers=4 arch.dim=512 arch.mlp_dim=2048 arch.rnn_hidden_size=512"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
