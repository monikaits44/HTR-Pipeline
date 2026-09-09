#!/bin/bash
#SBATCH --job-name=cnn_synth
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/cnn_rnn_synth_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/cnn_rnn_synth_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# CNN-RNN Baseline — Synthetic pretrain, test on IAM — NO SWA
# Longer time: synthetic dataset is ~100x larger than IAM
# Fewer epochs: synthetic convergence is faster per-epoch
# =============================================================================
export EXPERIMENT_NAME="cnn_rnn_synthetic"
export CONFIG_FILES="configs/config.yaml configs/baseline.yaml"
export CLI_OVERRIDES="data.mode=synthetic train.num_epochs=15"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
