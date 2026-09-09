#!/bin/bash
#SBATCH --job-name=abl_ft_gu
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_ft_vit_gradual_unfreeze_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/abl_ft_vit_gradual_unfreeze_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# ABLATION: TorchVision ViT-B/16 fine-tune WITH gradual unfreezing
# Tests: Does gradual unfreezing help vs training all layers from start?
# Baseline: 15_finetune_vit_b16.slurm (no gradual unfreeze)
# =============================================================================
export EXPERIMENT_NAME="ablation_ft_vit_gradual_unfreeze"
export CONFIG_FILES="configs/config.yaml configs/finetune_torchvision_vit.yaml"
export CLI_OVERRIDES="data.mode=iam finetune.enabled=true finetune.base_lr=2e-5 finetune.head_lr=5e-4 finetune.lr_decay_rate=0.85 finetune.gradual_unfreeze=true finetune.unfreeze_warmup=3 finetune.unfreeze_every=3"
source "/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular/run_experiment.sh"
