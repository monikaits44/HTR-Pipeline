#!/bin/bash
#SBATCH --job-name=baseline_cnn_rnn
#SBATCH --output=../logs/baseline/output_%j.log
#SBATCH --error=../logs/baseline/error_%j.log
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080

# Activate the virtual environment
source ~/.bashrc
source ../../.venv/bin/activate

# Change to project root directory
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline

# Run the baseline CNN-RNN model
echo "========================================="
echo "Starting Baseline CNN-RNN Training"
echo "Started at: $(date)"
echo "========================================="

/home/hpc/iwi5/iwi5369h/HTR-Pipeline/.venv/bin/python \
    scripts/trainer.py \
    configs/config.yaml \
    configs/baseline.yaml

echo "========================================="
echo "Completed at: $(date)"
echo "========================================="
