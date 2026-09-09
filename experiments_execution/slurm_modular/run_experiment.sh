#!/bin/bash
# =============================================================================
# run_experiment.sh — Modular experiment runner for HTR-Pipeline
# =============================================================================
#
# This is the SINGLE entry point for all experiments. Individual SLURM scripts
# set environment variables and then source this runner.
#
# Required env vars (set by caller):
#   EXPERIMENT_NAME   — human-readable name (e.g. "vit_rgts_v2_4reg_iam")
#   CONFIG_FILES      — space-separated YAML files (e.g. "configs/config.yaml configs/baseline.yaml")
#   CLI_OVERRIDES     — space-separated key=value pairs (e.g. "arch.num_registers=4 swa.enabled=true")
#
# Optional env vars:
#   PROJECT_ROOT      — defaults to /home/hpc/iwi5/iwi5369h/HTR-Pipeline
#   PYTHON_BIN        — defaults to $PROJECT_ROOT/.venv/bin/python
# =============================================================================

set -euo pipefail

# ---- Defaults ----
PROJECT_ROOT="${PROJECT_ROOT:-/home/hpc/iwi5/iwi5369h/HTR-Pipeline}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"

# ---- Environment setup ----
source ~/.bashrc
source "${PROJECT_ROOT}/.venv/bin/activate"
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export NO_ALBUMENTATIONS_UPDATE=1
export NLTK_DATA="${HOME}/nltk_data"

# ---- Validate inputs ----
if [[ -z "${EXPERIMENT_NAME:-}" ]]; then
    echo "ERROR: EXPERIMENT_NAME not set" >&2
    exit 1
fi
if [[ -z "${CONFIG_FILES:-}" ]]; then
    echo "ERROR: CONFIG_FILES not set" >&2
    exit 1
fi

# ---- Print banner ----
echo "========================================="
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Host:       $(hostname)"
echo "GPU:        ${CUDA_VISIBLE_DEVICES:-none}"
echo "Started at: $(date)"
echo "-----------------------------------------"
echo "Configs:    ${CONFIG_FILES}"
echo "Overrides:  ${CLI_OVERRIDES:-none}"
echo "========================================="

# ---- Build command ----
CMD="${PYTHON_BIN} scripts/trainer.py ${CONFIG_FILES}"
if [[ -n "${CLI_OVERRIDES:-}" ]]; then
    CMD="${CMD} ${CLI_OVERRIDES}"
fi

echo "Command: ${CMD}"
echo "========================================="

# ---- Run ----
eval ${CMD}
EXIT_CODE=$?

# ---- Finish banner ----
echo "========================================="
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Exit code:  ${EXIT_CODE}"
echo "Completed:  $(date)"
echo "========================================="

exit ${EXIT_CODE}
