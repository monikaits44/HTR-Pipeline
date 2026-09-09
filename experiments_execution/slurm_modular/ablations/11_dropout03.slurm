#!/bin/bash
#SBATCH --job-name=abl_drop
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_dropout03_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_dropout03_%j.err
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# ABLATION: ViT-RGTS v2 with higher dropout (0.3)
# Tests: Does stronger regularization help on small IAM data?
# Baseline: 07_vit_rgts_v2_4reg.slurm (dropout=0.1)
# =============================================================================
export EXPERIMENT_NAME="ablation_dropout03_4reg"
export CONFIG_FILES="configs/config.yaml configs/baseline_vit_rgts_v2.yaml"
export CLI_OVERRIDES="data.mode=iam arch.num_registers=4 arch.dropout=0.3 arch.emb_dropout=0.3"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
