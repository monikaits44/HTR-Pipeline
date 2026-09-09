#!/bin/bash
#SBATCH --job-name=reg_comparison
#SBATCH --output=logs/register_comparison_%j.log
#SBATCH --error=logs/register_comparison_%j.log
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rtx3080

# Activate the virtual environment
source ~/.bashrc
source /home/hpc/iwi5/iwi5369h/HTR-Pipeline/.venv/bin/activate

# Change to project root directory
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline

echo "========================================="
echo "Register Comparison — Attention Quality"
echo "Started at: $(date)"
echo "========================================="
echo "Models: run_76 (R=0), run_84 (R=4), run_68 (R=7), run_71 (R=16)"
echo "Samples: 100"
echo ""

python scripts/postprocessing/register_comparison.py \
    --models run_76:0 run_84:4 run_68:7 run_71:16 \
    --config configs/baseline_vit_rgts_v2.yaml \
    --data-path data/IAM \
    --dataset test \
    --num-samples 100 \
    --save-dir visualizations/register_comparison \
    --layer -1

echo ""
echo "========================================="
echo "Completed at: $(date)"
echo "========================================="
