#!/bin/bash
#SBATCH --job-name=postprocess
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/postprocess_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/logs/modular/postprocess_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
# =============================================================================
# Post-processing pipeline: attention visualization, GradCAM, quantitative figs
# Requires GPU for forward passes on 4 ViT-RGTS v2 models.
# Runs: register sweep (0,4,8,16 reg) = run_144, run_146, run_151, run_152
# =============================================================================
set -uo pipefail   # NOTE: no -e — one step's failure must not skip the rest
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export NO_ALBUMENTATIONS_UPDATE=1

DEVICE="cuda:0"
RUNS_DIR="saved_models/experiments"
IAM_TEST="data/IAM/processed_lines/test"

# Pick 5 visually rich sample images (used across all single-image scripts)
SAMPLE_IMG=$(find ${IAM_TEST} -name "*.png" | shuf -n 1 | head -1)

echo "================================================================"
echo "  POST-PROCESSING PIPELINE — $(date)"
echo "  Sample image: ${SAMPLE_IMG}"
echo "================================================================"

# ── Step 1: Attention quality metrics (entropy, Gini, sparsity) ──────────────
echo ""
echo "--- Step 1: Attention Quality Metrics (100 test images) ---"
mkdir -p outputs/report_figures/attention_quality
python scripts/postprocessing/comparative/attention_quality_metrics.py \
    configs/config.yaml configs/baseline_vit_rgts_v2.yaml \
    --model-path ${RUNS_DIR}/run_144/model.pt \
    --model-path ${RUNS_DIR}/run_146/model.pt \
    --model-path ${RUNS_DIR}/run_151/model.pt \
    --model-path ${RUNS_DIR}/run_152/model.pt \
    --save-dir outputs/report_figures/attention_quality \
    -- $(find ${IAM_TEST} -name "*.png" | shuf -n 100 | tr '\n' ' ') \
    || echo "[WARN] Step 1 failed, continuing"

# ── Step 2: Publication Fig 2 — register comparison attention grid ───────────
echo ""
echo "--- Step 2: Publication Fig 2 (Register Comparison) ---"
python scripts/pub_fig2_register_comparison.py \
    --device ${DEVICE} \
    --output outputs/pub_figures \
    --runs run_144 run_146 run_151 run_152 \
    || echo "[WARN] Step 2 failed, continuing"

# ── Step 3: Quantitative register analysis bar charts ───────────────────────
echo ""
echo "--- Step 3: Quantitative Register Analysis ---"
python scripts/pub_register_quantitative.py \
    --device ${DEVICE} \
    --output_dir outputs/pub_figures \
    --n_images 100 \
    --runs run_144 run_146 run_151 run_152 \
    || echo "[WARN] Step 3 failed, continuing"

# ── Step 4: GradCAM + quantitative ──────────────────────────────────────────
echo ""
echo "--- Step 4: GradCAM Analysis ---"
python scripts/pub_fig3_gradcam_quantitative.py \
    --device ${DEVICE} \
    --output outputs/pub_figures \
    --runs run_144 run_146 run_151 run_152 \
    || echo "[WARN] Step 4 failed, continuing"

# ── Step 5: Beyond-Memorization per-character attention (single model) ───────
echo ""
echo "--- Step 5: Beyond-Memorization Attention (single model, 4reg) ---"
python scripts/postprocessing/beyond_memorization_viz.py \
    --config configs/config.yaml \
    --model ${RUNS_DIR}/run_146/model.pt \
    --image-dir ${IAM_TEST} \
    --num-samples 10 \
    --save-dir outputs/report_figures/attention_bm_single \
    || echo "[WARN] Step 5 failed, continuing"

# ── Step 6: Beyond-Memorization comparison (0 vs 4 vs 16 registers) ─────────
echo ""
echo "--- Step 6: Beyond-Memorization Comparison (0/4/16 reg) ---"
python scripts/postprocessing/beyond_memorization_viz.py \
    --config configs/config.yaml \
    --models-dir ${RUNS_DIR} \
    --runs run_144 run_146 run_152 \
    --image-dir ${IAM_TEST} \
    --num-samples 5 \
    --save-dir outputs/report_figures/attention_bm_comparison \
    || echo "[WARN] Step 6 failed, continuing"

# ── Step 7: Head specialization (which heads attend where) ───────────────────
echo ""
echo "--- Step 7: Head Specialization (4reg model) ---"
python scripts/postprocessing/attention_viz/head_specialization.py \
    --image ${SAMPLE_IMG} \
    --runs run_144 run_146 run_151 run_152 \
    --runs-dir ${RUNS_DIR} \
    --device ${DEVICE} \
    --save-dir outputs/report_figures/head_specialization \
    || echo "[WARN] Step 7 failed, continuing"

# ── Step 8: Layerwise attention evolution ────────────────────────────────────
echo ""
echo "--- Step 8: Layerwise Attention Evolution (4reg model) ---"
python scripts/postprocessing/attention_viz/layerwise_evolution.py \
    --image ${SAMPLE_IMG} \
    --runs run_146 \
    --runs-dir ${RUNS_DIR} \
    --device ${DEVICE} \
    --save-dir outputs/report_figures/layerwise_evolution \
    || echo "[WARN] Step 8 failed, continuing"

echo ""
echo "================================================================"
echo "  POST-PROCESSING COMPLETE — $(date)"
echo "  Figures in outputs/report_figures/ and outputs/pub_figures/"
echo "================================================================"
