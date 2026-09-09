#!/bin/bash
# =============================================================================
# Submit ALL READ2016 dataset experiments (5 jobs: baseline + register sweep)
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Submitting ALL READ2016 experiments..."
echo "======================================="

for script in "${SCRIPT_DIR}/read2016/"*.slurm; do
    name=$(basename "$script")
    echo "  Submitting: ${name}"
    sbatch "$script"
done

echo "======================================="
echo "All READ2016 experiments submitted."
echo "Monitor with: squeue -u \$USER"
