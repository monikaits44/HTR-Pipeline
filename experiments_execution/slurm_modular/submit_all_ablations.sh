#!/bin/bash
# =============================================================================
# Submit ALL ablation experiments (18 jobs)
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Submitting ALL Ablation experiments..."
echo "======================================="

for script in "${SCRIPT_DIR}/ablations/"*.slurm; do
    name=$(basename "$script")
    echo "  Submitting: ${name}"
    sbatch "$script"
done

echo "======================================="
echo "All Ablation experiments submitted."
echo "Monitor with: squeue -u \$USER"
