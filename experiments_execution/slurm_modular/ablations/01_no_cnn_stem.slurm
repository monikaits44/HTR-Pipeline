#!/bin/bash
#SBATCH --job-name=abl_nostem
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_no_cnn_stem_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_no_cnn_stem_%j.err
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# ABLATION: ViT-RGTS v2 WITHOUT CNN stem (raw patch embedding)
# Tests: Is the CNN stem critical for CTC alignment?
# Baseline: 07_vit_rgts_v2_4reg.slurm (with CNN stem)
# =============================================================================
export EXPERIMENT_NAME="ablation_no_cnn_stem_4reg"
export CONFIG_FILES="configs/config.yaml configs/baseline_vit_rgts_v2.yaml"
export CLI_OVERRIDES="data.mode=iam arch.num_registers=4 arch.use_cnn_stem=false"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
