#!/bin/bash
#SBATCH --job-name=abl_rnn1
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_rnn1_layer_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_rnn1_layer_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# ABLATION: ViT-RGTS v2 with 1-layer BiLSTM (lighter head)
# Tests: Is 3-layer BiLSTM necessary or is 1 layer sufficient?
# Baseline: 07_vit_rgts_v2_4reg.slurm (rnn_layers=3)
# =============================================================================
export EXPERIMENT_NAME="ablation_rnn1_layer_4reg"
export CONFIG_FILES="configs/config.yaml configs/baseline_vit_rgts_v2.yaml"
export CLI_OVERRIDES="data.mode=iam arch.num_registers=4 arch.rnn_layers=1"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
