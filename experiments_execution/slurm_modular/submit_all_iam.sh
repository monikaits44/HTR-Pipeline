#!/bin/bash
# =============================================================================
# Submit ALL IAM-mode experiments (20 jobs: 4 architectures × with/without SWA)
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Submitting ALL IAM experiments..."
echo "================================="

for script in "${SCRIPT_DIR}/iam/"*.slurm; do
    name=$(basename "$script")
    echo "  Submitting: ${name}"
    sbatch "$script"
done

echo "================================="
echo "All IAM experiments submitted."
echo "Monitor with: squeue -u \$USER"
