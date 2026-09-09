#!/bin/bash
# =============================================================================
# batch_evaluate.sh — Evaluate all completed experiment runs
# =============================================================================
# Scans saved_models/experiments/ for runs with model.pt and submits
# evaluation jobs for each. Also generates aggregate results table.
#
# Usage:
#   bash evaluation_execution/batch_evaluate.sh              # all runs
#   bash evaluation_execution/batch_evaluate.sh --from 150   # runs ≥ 150
#   bash evaluation_execution/batch_evaluate.sh --dry-run    # show what would run
# =============================================================================
set -euo pipefail

# Derive PROJECT_ROOT from this script's location so the pipeline is portable
# across machines / user accounts. Override by exporting PROJECT_ROOT.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="${PROJECT_ROOT:-$( cd "${SCRIPT_DIR}/.." && pwd )}"
EXPERIMENTS_DIR="${PROJECT_ROOT}/saved_models/experiments"
EVAL_SCRIPT="${PROJECT_ROOT}/evaluation_execution/model_evaluation.sh"

FROM_RUN=1
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM_RUN="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  BATCH EVALUATION — HTR-Pipeline"
echo "  Scanning: ${EXPERIMENTS_DIR}"
echo "  From run: ${FROM_RUN}"
echo "============================================================"

count=0
for run_dir in "${EXPERIMENTS_DIR}"/run_*; do
    [[ -d "$run_dir" ]] || continue
    run_num=$(basename "$run_dir" | sed 's/run_//')

    # Skip if below threshold
    [[ "$run_num" -lt "$FROM_RUN" ]] 2>/dev/null && continue

    # Skip if no model checkpoint
    [[ -f "${run_dir}/model.pt" ]] || continue

    # Skip if already evaluated
    if [[ -d "${run_dir}/evaluation" ]] && [[ -f "${run_dir}/evaluation/evaluation_summary_test.json" ]]; then
        echo "  SKIP run_${run_num} (already evaluated)"
        continue
    fi

    # Determine config from saved config.json
    if [[ ! -f "${run_dir}/config.json" ]]; then
        echo "  WARN run_${run_num}: no config.json, skipping"
        continue
    fi

    # Extract arch type to select correct config files
    arch_type=$(python3 -c "import json; c=json.load(open('${run_dir}/config.json')); print(c.get('arch',{}).get('type','unknown'))" 2>/dev/null || echo "unknown")
    data_path=$(python3 -c "import json; c=json.load(open('${run_dir}/config.json')); print(c.get('data',{}).get('path',''))" 2>/dev/null || echo "")

    case "$arch_type" in
        cnn_rnn)     CONFIGS="configs/config.yaml configs/baseline.yaml" ;;
        vit_rgts)    CONFIGS="configs/config.yaml configs/baseline_vit_rgts_v2.yaml" ;;
        torchvision_vit) CONFIGS="configs/config.yaml configs/torchvision_vit.yaml" ;;
        trocr)       CONFIGS="configs/config.yaml configs/trocr.yaml" ;;
        htrvt)       CONFIGS="configs/config.yaml configs/htrvt_read2016_r4.yaml" ;;
        *)           CONFIGS="configs/config.yaml"; echo "  WARN run_${run_num}: unknown arch '${arch_type}'" ;;
    esac

    # Build overrides from saved config
    OVERRIDES=""
    if [[ -n "$data_path" ]] && [[ "$data_path" != *"IAM"* ]]; then
        OVERRIDES="data.path=${data_path}"
    fi

    echo "  EVAL run_${run_num} (${arch_type})"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "    [dry-run] RUN_ID=${run_num} EVAL_CONFIGS=\"${CONFIGS}\" sbatch ${EVAL_SCRIPT}"
    else
        RUN_ID="${run_num}" EVAL_CONFIGS="${CONFIGS}" CONFIG_OVERRIDES="${OVERRIDES}" \
            sbatch "${EVAL_SCRIPT}"
    fi
    count=$((count + 1))
done

echo ""
echo "============================================================"
echo "  Submitted ${count} evaluation jobs"
echo "  Monitor: squeue -u \$USER"
echo "============================================================"
