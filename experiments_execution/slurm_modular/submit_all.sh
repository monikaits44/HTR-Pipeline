#!/bin/bash
# =============================================================================
# Submit ALL experiments across all categories (54 total jobs)
# WARNING: This submits a large number of GPU jobs. Use with care.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "  SUBMITTING ALL HTR-PIPELINE EXPERIMENTS"
echo "  Total: 20 IAM + 16 Synthetic + 22 Ablation + 5 READ2016 = 63 jobs"
echo "============================================================"
echo ""

read -p "Are you sure you want to submit 63 GPU jobs? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "--- IAM Experiments (20) ---"
bash "${SCRIPT_DIR}/submit_all_iam.sh"

echo ""
echo "--- Synthetic Experiments (16) ---"
bash "${SCRIPT_DIR}/submit_all_synthetic.sh"

echo ""
echo "--- Ablation Experiments (22) ---"
bash "${SCRIPT_DIR}/submit_all_ablations.sh"

echo ""
echo "--- READ2016 Dataset Experiments (5) ---"
bash "${SCRIPT_DIR}/submit_all_read2016.sh"

echo ""
echo "============================================================"
echo "  ALL 63 EXPERIMENTS SUBMITTED"
echo "  Monitor: squeue -u \$USER"
echo "  Cancel:  scancel -u \$USER"
echo "============================================================"
