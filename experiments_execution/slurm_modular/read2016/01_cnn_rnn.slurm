#!/bin/bash
#SBATCH --job-name=r16_cnn_rnn
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/r16_cnn_rnn_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/r16_cnn_rnn_%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# CNN-RNN Baseline — READ2016 dataset — NO SWA
# Validates architecture generalization across datasets
# =============================================================================
export EXPERIMENT_NAME="cnn_rnn_read2016"
export CONFIG_FILES="configs/config.yaml configs/baseline.yaml"
export CLI_OVERRIDES="data.mode=iam data.path=./data/READ2016/processed"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
