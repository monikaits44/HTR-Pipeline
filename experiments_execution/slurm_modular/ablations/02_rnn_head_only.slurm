#!/bin/bash
#SBATCH --job-name=abl_rnn
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_rnn_head_only_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_rnn_head_only_%j.err
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# ABLATION: ViT-RGTS v2 with RNN-only head (no dual CNN shortcut)
# Tests: Is the dual supervision (CTCtopB) better than RNN-only (CTCtopR)?
# Baseline: 07_vit_rgts_v2_4reg.slurm (head_type=both)
# =============================================================================
export EXPERIMENT_NAME="ablation_rnn_head_only_4reg"
export CONFIG_FILES="configs/config.yaml configs/baseline_vit_rgts_v2.yaml"
export CLI_OVERRIDES="data.mode=iam arch.num_registers=4 arch.head_type=rnn"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
