#!/bin/bash
#SBATCH --job-name=abl_ft_gu2
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_ft_trocr_gradual_unfreeze_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_ft_trocr_gradual_unfreeze_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# ABLATION: TrOCR fine-tune WITH gradual unfreezing
# Tests: Does gradual unfreezing help for in-domain pretrained TrOCR?
# Baseline: 19_finetune_trocr.slurm (no gradual unfreeze)
# =============================================================================
export EXPERIMENT_NAME="ablation_ft_trocr_gradual_unfreeze"
export CONFIG_FILES="configs/config.yaml configs/finetune_trocr.yaml"
export CLI_OVERRIDES="data.mode=iam finetune.gradual_unfreeze=true finetune.unfreeze_warmup=5 finetune.unfreeze_every=2"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
