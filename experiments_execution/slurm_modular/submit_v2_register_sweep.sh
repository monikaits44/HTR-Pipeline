#!/bin/bash
# =============================================================================
# Submit only the ViT-RGTS v2 register sweep on IAM (10 jobs: 5 × with/without SWA)
# This is the PRIMARY experiment set for the register token investigation.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Submitting ViT-RGTS v2 Register Sweep (IAM)..."
echo "================================================"

for script in "${SCRIPT_DIR}/iam/"*vit_rgts_v2*.slurm; do
    name=$(basename "$script")
    echo "  Submitting: ${name}"
    sbatch "$script"
done

echo "================================================"
echo "Register sweep submitted."
echo "Monitor with: squeue -u \$USER"
