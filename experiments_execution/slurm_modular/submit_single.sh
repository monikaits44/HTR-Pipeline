#!/bin/bash
# =============================================================================
# Submit a SINGLE experiment by name/pattern
# Usage: ./submit_single.sh <pattern>
# Example: ./submit_single.sh 07_vit_rgts_v2_4reg
#          ./submit_single.sh ablations/01
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <pattern>"
    echo ""
    echo "Available experiments:"
    echo "--- IAM ---"
    ls "${SCRIPT_DIR}/iam/"*.slurm 2>/dev/null | xargs -I{} basename {}
    echo ""
    echo "--- Synthetic ---"
    ls "${SCRIPT_DIR}/synthetic/"*.slurm 2>/dev/null | xargs -I{} basename {}
    echo ""
    echo "--- Ablations ---"
    ls "${SCRIPT_DIR}/ablations/"*.slurm 2>/dev/null | xargs -I{} basename {}
    exit 1
fi

PATTERN="$1"
FOUND=0

for dir in iam synthetic ablations; do
    for script in "${SCRIPT_DIR}/${dir}/"*"${PATTERN}"*.slurm; do
        if [[ -f "$script" ]]; then
            name=$(basename "$script")
            echo "Submitting: ${dir}/${name}"
            sbatch "$script"
            FOUND=$((FOUND + 1))
        fi
    done
done

if [[ $FOUND -eq 0 ]]; then
    echo "No scripts matching pattern: ${PATTERN}"
    exit 1
fi

echo "Submitted ${FOUND} job(s)."
