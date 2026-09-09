#!/bin/bash
#SBATCH --job-name=batch_eval
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/batch_eval_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/batch_eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# Batch evaluation of all final experiment runs (143-205).
# Produces per-run evaluation_summary_*.json + evaluation_report_*.txt
# =============================================================================
set -euo pipefail
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline
source .venv/bin/activate
export PYTHONUNBUFFERED=1

RUNS_DIR="saved_models/experiments"
OK=0; SKIP=0; ERR=0

echo "================================================================"
echo "  BATCH EVALUATION — $(date)"
echo "================================================================"

for n in $(seq 143 205); do
    RUN="${RUNS_DIR}/run_${n}"
    [ -f "${RUN}/model.pt" ] || { echo "SKIP run_${n}: no model.pt"; SKIP=$((SKIP+1)); continue; }
    [ -f "${RUN}/evaluation/evaluation_summary_test.json" ] && {
        echo "SKIP run_${n}: already evaluated"; SKIP=$((SKIP+1)); continue
    }
    echo "--- Evaluating run_${n} ---"

    # Convert the run's own saved config.json (a complete, exact OmegaConf dump)
    # to YAML so evaluate.py reconstructs the EXACT architecture used at train
    # time (registers, head_type, dim, depth, use_cnn_stem, etc.) — using a
    # generic per-arch default config here would silently mismatch ablation
    # runs (e.g. head_type=cnn) and cause "Missing key(s) in state_dict" errors.
    TMP_CFG="${RUN}/_eval_config.yaml"
    python3 -c "
import json, yaml
c = json.load(open('${RUN}/config.json'))
c['resume'] = '${RUN}/model.pt'
c['eval_sets'] = 'test'
yaml.safe_dump(c, open('${TMP_CFG}', 'w'))
"
    python scripts/postprocessing/single_model/evaluate.py \
        "${TMP_CFG}" 2>&1 \
        && OK=$((OK+1)) \
        || { echo "  ERROR on run_${n}"; ERR=$((ERR+1)); }
    rm -f "${TMP_CFG}"
done

echo ""
echo "================================================================"
echo "  BATCH EVALUATION DONE — $(date)"
echo "  OK: ${OK}  SKIPPED: ${SKIP}  ERRORS: ${ERR}"
echo "================================================================"
