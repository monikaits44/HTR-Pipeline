#!/bin/bash
# =============================================================================
# Submit ViT-RGTS v2 full register sweep (registers 0 – 16)
# =============================================================================
# Uses a SLURM array job so all 17 register variants are tracked under a
# single job ID.  Each array element index equals the number of registers.
#
# Usage: cd experiments_execution && bash submit_vit_rgts_v2_full_reg_sweep.sh
# =============================================================================

echo "================================================================"
echo "Submitting ViT-RGTS v2 (CNN Stem) — Full Register Sweep (0-16)"
echo "================================================================"

cd "$(dirname "$0")"

# Ensure the log directory exists
mkdir -p /home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/logs/vit_rgts_v2

echo "Submitting array job: registers 0, 1, 2, ..., 16"
JOB_ID=$(sbatch --parsable slurm_scripts/14_vit_rgts_v2_full_reg_sweep.slurm)

if [ $? -eq 0 ]; then
    echo ""
    echo "Array job submitted successfully!"
    echo "  Job ID   : ${JOB_ID}"
    echo "  Tasks    : 0_reg through 16_reg  (SLURM_ARRAY_TASK_ID = num_registers)"
    echo ""
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    echo "  squeue -j ${JOB_ID}"
    echo ""
    echo "Logs will appear in:"
    echo "  experiments_execution/logs/vit_rgts_v2/output_reg<N>_<jobid>.log"
    echo "  experiments_execution/logs/vit_rgts_v2/error_reg<N>_<jobid>.log"
    echo ""
    echo "Results will be saved in: saved_models/experiments/run_*/"
else
    echo "ERROR: sbatch submission failed."
    exit 1
fi
