#!/bin/bash
#SBATCH --job-name=cnn_rnn_iam
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/cnn_rnn_iam_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/cnn_rnn_iam_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# CNN-RNN Baseline — IAM train/val/test — NO SWA
# =============================================================================
export EXPERIMENT_NAME="cnn_rnn_iam"
export CONFIG_FILES="configs/config.yaml configs/baseline.yaml"
export CLI_OVERRIDES="data.mode=iam"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
