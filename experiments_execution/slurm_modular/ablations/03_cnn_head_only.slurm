#!/bin/bash
#SBATCH --job-name=abl_cnn_h
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_cnn_head_only_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_cnn_head_only_%j.err
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# ABLATION: ViT-RGTS v2 with CNN-only head (linear CTC projection)
# Tests: Is the BiLSTM CTC head necessary, or is linear projection sufficient?
# Baseline: 07_vit_rgts_v2_4reg.slurm (head_type=both)
# =============================================================================
export EXPERIMENT_NAME="ablation_cnn_head_only_4reg"
export CONFIG_FILES="configs/config.yaml configs/baseline_vit_rgts_v2.yaml"
export CLI_OVERRIDES="data.mode=iam arch.num_registers=4 arch.head_type=cnn"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
