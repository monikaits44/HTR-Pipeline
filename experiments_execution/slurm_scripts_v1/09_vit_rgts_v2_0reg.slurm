#!/bin/bash
#SBATCH --job-name=vit_rgts_v2_0reg
#SBATCH --output=../logs/vit_rgts_v2/output_0reg_%j.log
#SBATCH --error=../logs/vit_rgts_v2/error_0reg_%j.log
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080

# ViT-RGTS v2 with CNN Stem — Register sweep: 0 registers
# FIXES APPLIED (run_49 diagnosis):
#   1. Weight decay reduced 10-20x (was 1000x higher than CNN baseline)
#   2. Learning rates increased: stem=1e-3, transformer=5e-4, head=1e-3
#   3. 3-group optimizer (stem/transformer/head) — head gets very low WD
#   4. Warmup: 5 epochs at 10% start (was 10 epochs at 1% start)
#   5. Moderate augmentation (was strong 'vit')
#   6. MaxPool in CNN stem (was AvgPool)
#   7. Grad accum 2 (was 4), grad clip 10 (was 5)
# Expected: CER < 15%

source ~/.bashrc
source ../../.venv/bin/activate
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline

echo "========================================="
echo "ViT-RGTS v2 (CNN Stem) — 0 Registers"
echo "Started at: $(date)"
echo "========================================="

/home/hpc/iwi5/iwi5369h/HTR-Pipeline/.venv/bin/python \
    scripts/trainer.py \
    configs/baseline_vit_rgts_v2.yaml \
    arch.num_registers=0

echo "========================================="
echo "Completed at: $(date)"
echo "========================================="
