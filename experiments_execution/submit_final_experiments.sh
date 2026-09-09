#!/bin/bash
# =============================================================================
# submit_final_experiments.sh — Submit all 63 final experiments
# Captures every SLURM job ID for tracking.
# Synthetic experiments depend on LMDB extraction completing first.
# =============================================================================
set -euo pipefail

SD="/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular"
LOG="/home/hpc/iwi5/iwi5369h/HTR-Pipeline/documents/final/SUBMISSION_LOG.txt"
mkdir -p "$(dirname "$LOG")"

exec > >(tee "$LOG") 2>&1

echo "================================================================"
echo "  FINAL EXPERIMENT SUBMISSION — $(date)"
echo "================================================================"
echo ""

sub() {
    local script="$1"
    local dep="${2:-}"
    local name=$(basename "$script" .slurm)
    local result
    if [[ -n "$dep" ]]; then
        result=$(sbatch --dependency=afterok:${dep} "$script" 2>&1)
    else
        result=$(sbatch "$script" 2>&1)
    fi
    local jid=$(echo "$result" | grep -oP '\d+$' || echo "FAILED")
    printf "  %-8s  %s\n" "$jid" "$name"
    echo "$jid" >&2
}

# Redirect sub's stderr (job IDs) to /dev/null for display; capture via $()
sub2() {
    local script="$1"
    local dep="${2:-}"
    local name=$(basename "$script" .slurm)
    local result
    if [[ -n "$dep" ]]; then
        result=$(sbatch --dependency=afterok:${dep} "$script" 2>&1)
    else
        result=$(sbatch "$script" 2>&1)
    fi
    local jid=$(echo "$result" | grep -oP '\d+$' || echo "FAILED")
    printf "  %-8s  %s\n" "$jid" "$name"
    # Return jid on stdout for capture
    echo "$jid"
}

# =========================================================================
# PHASE 1: Baselines (2 jobs, ~16h)
# =========================================================================
echo "--- PHASE 1: Baselines (2 jobs) ---"
sub "${SD}/iam/01_cnn_rnn.slurm"
sub "${SD}/iam/07_vit_rgts_v2_4reg.slurm"
echo ""

# =========================================================================
# PHASE 2: Core Register Sweep ± SWA (10 jobs, ~20h)
# =========================================================================
echo "--- PHASE 2: Register Sweep (10 jobs) ---"
sub "${SD}/iam/02_cnn_rnn_swa.slurm"
sub "${SD}/iam/03_vit_rgts_v2_0reg.slurm"
sub "${SD}/iam/04_vit_rgts_v2_0reg_swa.slurm"
sub "${SD}/iam/05_vit_rgts_v2_2reg.slurm"
sub "${SD}/iam/06_vit_rgts_v2_2reg_swa.slurm"
sub "${SD}/iam/08_vit_rgts_v2_4reg_swa.slurm"
sub "${SD}/iam/09_vit_rgts_v2_8reg.slurm"
sub "${SD}/iam/10_vit_rgts_v2_8reg_swa.slurm"
sub "${SD}/iam/11_vit_rgts_v2_16reg.slurm"
sub "${SD}/iam/12_vit_rgts_v2_16reg_swa.slurm"
echo ""

# =========================================================================
# PHASE 3: Pretrained Baselines (8 jobs, ~28h)
# =========================================================================
echo "--- PHASE 3: Pretrained Baselines (8 jobs) ---"
sub "${SD}/iam/13_torchvision_vit_b16.slurm"
sub "${SD}/iam/14_torchvision_vit_b16_swa.slurm"
sub "${SD}/iam/15_finetune_vit_b16.slurm"
sub "${SD}/iam/16_finetune_vit_b16_swa.slurm"
sub "${SD}/iam/17_trocr_base.slurm"
sub "${SD}/iam/18_trocr_base_swa.slurm"
sub "${SD}/iam/19_finetune_trocr.slurm"
sub "${SD}/iam/20_finetune_trocr_swa.slurm"
echo ""

# =========================================================================
# PHASE 4: Architecture Ablations (11 jobs, ~24h)
# =========================================================================
echo "--- PHASE 4: Architecture Ablations (11 jobs) ---"
sub "${SD}/ablations/01_no_cnn_stem.slurm"
sub "${SD}/ablations/02_rnn_head_only.slurm"
sub "${SD}/ablations/03_cnn_head_only.slurm"
sub "${SD}/ablations/04_gru_head.slurm"
sub "${SD}/ablations/05_vit_augmentation.slurm"
sub "${SD}/ablations/06_depth4.slurm"
sub "${SD}/ablations/07_depth8.slurm"
sub "${SD}/ablations/08_dim128.slurm"
sub "${SD}/ablations/09_dim512.slurm"
sub "${SD}/ablations/10_rnn1_layer.slurm"
sub "${SD}/ablations/11_dropout03.slurm"
echo ""

# =========================================================================
# PHASE 5: SWA / LR / Seed / FT Ablations (11 jobs, ~28h)
# =========================================================================
echo "--- PHASE 5: SWA/LR/Seed Ablations (11 jobs) ---"
sub "${SD}/ablations/12_swa_early.slurm"
sub "${SD}/ablations/13_swa_late.slurm"
sub "${SD}/ablations/14_batch16.slurm"
sub "${SD}/ablations/15_ft_vit_gradual_unfreeze.slurm"
sub "${SD}/ablations/16_ft_trocr_gradual_unfreeze.slurm"
sub "${SD}/ablations/17_vit_rgts_v1.slurm"
sub "${SD}/ablations/18_lr_5e4.slurm"
sub "${SD}/ablations/19_swa_lr_high.slurm"
sub "${SD}/ablations/20_swa_lr_low.slurm"
sub "${SD}/ablations/21_seed_123.slurm"
sub "${SD}/ablations/22_seed_456.slurm"
echo ""

# =========================================================================
# PHASE 6: READ2016 Dataset Variation (5 jobs, ~20h)
# =========================================================================
echo "--- PHASE 6: READ2016 (5 jobs) ---"
sub "${SD}/read2016/01_cnn_rnn.slurm"
sub "${SD}/read2016/02_vit_rgts_v2_0reg.slurm"
sub "${SD}/read2016/03_vit_rgts_v2_4reg.slurm"
sub "${SD}/read2016/04_vit_rgts_v2_8reg.slurm"
sub "${SD}/read2016/05_vit_rgts_v2_16reg.slurm"
echo ""

# =========================================================================
# PHASE 7: Synthetic Extraction + Training (1 + 16 = 17 jobs)
# =========================================================================
echo "--- PHASE 7A: Synthetic Extraction (1 job, ~3h) ---"
EXTRACT_JID=$(sub2 "${SD}/extract_synthetic.slurm")
echo ""

echo "--- PHASE 7B: Synthetic Training (16 jobs, depend on extraction) ---"
for script in "${SD}/synthetic/"*.slurm; do
    sub "${script}" "${EXTRACT_JID}"
done
echo ""

# =========================================================================
echo "================================================================"
echo "  ALL 64 JOBS SUBMITTED — $(date)"
echo "  (1 extraction + 47 IAM/ablation/READ2016 + 16 synthetic)"
echo ""
echo "  Monitor:  squeue -u \$USER"
echo "  Logs:     tail -f logs/modular/<name>_<jobid>.out"
echo "  Results:  saved_models/experiments/"
echo "================================================================"
