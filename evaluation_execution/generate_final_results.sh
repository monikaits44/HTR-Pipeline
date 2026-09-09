#!/bin/bash
# =============================================================================
# generate_final_results.sh — Generate all final result artifacts
# =============================================================================
# Runs after all experiments + evaluations are complete.
# Produces: unified CSV, LaTeX tables, training curves, comparison plots.
#
# Usage:
#   bash evaluation_execution/generate_final_results.sh
#   bash evaluation_execution/generate_final_results.sh --from 150  # only new runs
# =============================================================================
set -euo pipefail

# Derive PROJECT_ROOT from this script's location so the pipeline is portable
# across machines / user accounts. Override by exporting PROJECT_ROOT.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="${PROJECT_ROOT:-$( cd "${SCRIPT_DIR}/.." && pwd )}"
cd "${PROJECT_ROOT}"

source "${PROJECT_ROOT}/.venv/bin/activate"
export PYTHONUNBUFFERED=1

OUTPUT_DIR="outputs/tables"
FIGURES_DIR="outputs/report_figures"
mkdir -p "${OUTPUT_DIR}" "${FIGURES_DIR}"

FROM_RUN="${1:-}"

echo "============================================================"
echo "  GENERATING FINAL RESULTS"
echo "  Output: ${OUTPUT_DIR} / ${FIGURES_DIR}"
echo "============================================================"

# 1. Aggregate all experiment results into CSV + LaTeX tables
echo ""
echo "--- Step 1: Aggregate Results ---"
python scripts/postprocessing/aggregate_results.py \
    --runs-dir saved_models/experiments \
    --output-dir "${OUTPUT_DIR}"
echo "  Generated: unified_results.csv, *.tex tables"

# 2. Generate training curves (loss, CER, WER over epochs)
echo ""
echo "--- Step 2: Training Curves ---"
if [[ -f scripts/postprocessing/comparative/plot_training_metrics.py ]]; then
    python scripts/postprocessing/comparative/plot_training_metrics.py \
        --runs-dir saved_models/experiments \
        --output-dir "${FIGURES_DIR}" 2>/dev/null || echo "  WARN: training curves script returned non-zero"
    echo "  Generated: training curves"
else
    echo "  SKIP: plot_training_metrics.py not found"
fi

# 3. Attention quality metrics comparison
echo ""
echo "--- Step 3: Attention Quality Metrics ---"
if [[ -f scripts/postprocessing/comparative/attention_quality_metrics.py ]]; then
    python scripts/postprocessing/comparative/attention_quality_metrics.py \
        --runs-dir saved_models/experiments \
        --output-dir "${FIGURES_DIR}" 2>/dev/null || echo "  WARN: attention metrics script returned non-zero"
    echo "  Generated: attention quality metrics"
else
    echo "  SKIP: attention_quality_metrics.py not found"
fi

echo ""
echo "============================================================"
echo "  RESULTS GENERATION COMPLETE"
echo "  Tables:  ${OUTPUT_DIR}/"
echo "  Figures: ${FIGURES_DIR}/"
echo "============================================================"
echo ""
echo "Key outputs:"
ls -la "${OUTPUT_DIR}"/*.csv "${OUTPUT_DIR}"/*.tex 2>/dev/null || echo "  (no tables yet)"
echo ""
echo "Figures:"
ls "${FIGURES_DIR}"/ 2>/dev/null | head -20 || echo "  (no figures yet)"
