#!/bin/bash
# =============================================================================
# Submit ViT-RGTS v2 register sweep experiments (runs 09-13)
# =============================================================================
# CNN Stem + Transformer architecture with register count: 0, 2, 4, 8, 16
#
# Usage: cd experiments_execution && bash submit_vit_rgts_v2_sweep.sh
# =============================================================================

echo "=================================================="
echo "Submitting ViT-RGTS v2 (CNN Stem) Register Sweep"
echo "=================================================="

cd "$(dirname "$0")"

# Submit all register sweep experiments
echo "Submitting 0-register experiment..."
sbatch slurm_scripts/09_vit_rgts_v2_0reg.slurm

echo "Submitting 2-register experiment..."
sbatch slurm_scripts/10_vit_rgts_v2_2reg.slurm

echo "Submitting 4-register experiment..."
sbatch slurm_scripts/11_vit_rgts_v2_4reg.slurm

echo "Submitting 8-register experiment..."
sbatch slurm_scripts/12_vit_rgts_v2_8reg.slurm

echo "Submitting 16-register experiment..."
sbatch slurm_scripts/13_vit_rgts_v2_16reg.slurm

echo ""
echo "All 5 experiments submitted!"
echo "Monitor with: squeue -u $USER"
echo "Results will be in: saved_models/experiments/run_*/"
