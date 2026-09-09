#!/bin/bash
#SBATCH --job-name=abl_gru
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_gru_head_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_gru_head_%j.err
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# ABLATION: ViT-RGTS v2 with BiGRU instead of BiLSTM
# Tests: Does LSTM's cell state benefit CTC over simpler GRU?
# Baseline: 07_vit_rgts_v2_4reg.slurm (rnn_type=lstm)
# =============================================================================
export EXPERIMENT_NAME="ablation_gru_head_4reg"
export CONFIG_FILES="configs/config.yaml configs/baseline_vit_rgts_v2.yaml"
export CLI_OVERRIDES="data.mode=iam arch.num_registers=4 arch.rnn_type=gru"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
