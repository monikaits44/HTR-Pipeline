#!/bin/bash
# =============================================================================
# Submit ALL synthetic-mode experiments (16 jobs)
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Submitting ALL Synthetic experiments..."
echo "========================================"

for script in "${SCRIPT_DIR}/synthetic/"*.slurm; do
    name=$(basename "$script")
    echo "  Submitting: ${name}"
    sbatch "$script"
done

echo "========================================"
echo "All Synthetic experiments submitted."
echo "Monitor with: squeue -u \$USER"
