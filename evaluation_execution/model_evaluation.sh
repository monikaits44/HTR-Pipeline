#!/bin/bash
#SBATCH --job-name=htr_eval
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --chdir=/home/hpc/iwi5/iwi5369h/HTR-Pipeline
#SBATCH --output=evaluation_execution/logs/eval_%j.log
#SBATCH --error=evaluation_execution/logs/eval_%j.log

# ── Configuration ───────────────────────────────────────────────────────────
# Override these via environment or command-line:
#   RUN_ID=107 CONFIG_OVERRIDES="arch.num_registers=0" sbatch model_evaluation.sh
RUN_ID="${RUN_ID:-107}"
EVAL_CONFIGS="${EVAL_CONFIGS:-configs/config.yaml configs/baseline.yaml}"
CONFIG_OVERRIDES="${CONFIG_OVERRIDES:-}"

# ── create log directory if needed ──────────────────────────────────────────
mkdir -p evaluation_execution/logs

LOG_START=$(date)
JOB_START_SEC=$(date +%s)

# ── Prologue ─────────────────────────────────────────────────────────────────
echo "### Job Prologue: ${SLURM_JOB_NAME} (ID: ${SLURM_JOB_ID}) on $(hostname) at ${LOG_START}"
echo "    Partition  : ${SLURM_JOB_PARTITION}"
echo "    Node list  : ${SLURM_JOB_NODELIST}"
echo "    CPUs       : ${SLURM_CPUS_PER_TASK}"
echo "    GPUs       : ${SLURM_GRES}"
echo ""

# ── GPU info ─────────────────────────────────────────────────────────────────
echo "=== GPU Status ==="
nvidia-smi
echo ""

# ── Activate environment ─────────────────────────────────────────────────────
echo "=== Environment ==="
source ~/HTR-Pipeline/.venv/bin/activate
export PYTHONUNBUFFERED=1
echo "Python   : $(which python)"
echo "Env      : ${VIRTUAL_ENV}"
echo ""

# ── Run evaluation ────────────────────────────────────────────────────────────
echo "================================================================================"
echo "Starting Evaluation: run_${RUN_ID}"
echo "Start Time: $(date)"
echo "================================================================================"
echo ""

python -u scripts/postprocessing/single_model/evaluate.py \
    ${EVAL_CONFIGS} \
    ${CONFIG_OVERRIDES} \
    resume=saved_models/experiments/run_${RUN_ID}/model.pt

EXIT_CODE=$?

echo ""
echo "================================================================================"
echo "Evaluation Finished"
echo "End Time  : $(date)"
echo "Exit Code : ${EXIT_CODE}"
echo "================================================================================"
echo ""

# ── Job statistics ────────────────────────────────────────────────────────────
JOB_END_SEC=$(date +%s)
ELAPSED=$(( JOB_END_SEC - JOB_START_SEC ))
ELAPSED_FMT=$(printf '%02d:%02d:%02d' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))

echo "=== JOB STATISTICS ==="
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Job Name     : ${SLURM_JOB_NAME}"
echo "  Partition    : ${SLURM_JOB_PARTITION}"
echo "  Node         : ${SLURM_JOB_NODELIST}"
echo "  Start        : ${LOG_START}"
echo "  End          : $(date)"
echo "  Elapsed      : ${ELAPSED_FMT}"
echo "======================"
echo ""

echo "=== GPU Utilization (final) ==="
nvidia-smi --query-accounted-apps=gpu_name,gpu_bus_id,pid,gpu_utilization,mem_utilization,max_memory_usage,time \
           --format=csv 2>/dev/null || nvidia-smi
echo ""

exit ${EXIT_CODE}
