# Systematic Experimentation Guide

## Final Experiment Design — Development Frozen 2026-06-14

---

## 1. Experimental Framework

### 1.1 Research Hypothesis

**H₁**: Register tokens improve attention map interpretability (measured by entropy, 
Gini coefficient, character localization accuracy) without degrading CER/WER.

**H₀**: Register tokens have no statistically significant effect on attention quality.

### 1.2 Independent Variable
- Number of register tokens: {0, 2, 4, 8, 16}

### 1.3 Dependent Variables
- **Primary**: CER (%), WER (%)
- **Secondary**: Attention entropy, Gini sparsity, peak sharpness, character localization
- **Tertiary**: Token norms, register specialization, head diversity

### 1.4 Controlled Variables
- Architecture: ViT-RGTS v2 (CNN stem, dim=256, depth=6, heads=8)
- Dataset: IAM Aachen splits (6,161/900/1,861 lines)
- Training: 80 epochs, AdamW, warmup+cosine, batch_size=8, accum=2
- Augmentation: `aug_transforms_cnn` (moderate)
- Random seed: Same across all conditions

---

## 2. Experiment Matrix

### 2.1 Complete Experiment Set (16 Runs)

| ID | Config | Architecture | Registers | Purpose |
|----|--------|-------------|-----------|---------|
| **01** | baseline.yaml | CNN-RNN | N/A | Performance floor |
| 02 | baseline_vit_rgts.yaml | ViT-RGTS v1 | 0 | v1 baseline |
| 03 | baseline_vit_rgts.yaml | ViT-RGTS v1 | 2 | v1 register effect |
| 04 | baseline_vit_rgts.yaml | ViT-RGTS v1 | 4 | v1 register effect |
| 05 | baseline_vit_rgts.yaml | ViT-RGTS v1 | 8 | v1 register effect |
| 06 | baseline_vit_rgts.yaml | ViT-RGTS v1 | 16 | v1 register effect |
| 07 | torchvision_vit.yaml | ViT-B/16 | 0 | Transfer learning upper bound |
| 08 | trocr.yaml | TrOCR-base | 0 | Transfer learning upper bound |
| **09** | baseline_vit_rgts_v2.yaml | **ViT-RGTS v2** | **0** | **v2 baseline** |
| **10** | baseline_vit_rgts_v2.yaml | **ViT-RGTS v2** | **2** | **Main experiment** |
| **11** | baseline_vit_rgts_v2.yaml | **ViT-RGTS v2** | **4** | **Main experiment** |
| **12** | baseline_vit_rgts_v2.yaml | **ViT-RGTS v2** | **8** | **Main experiment** |
| **13** | baseline_vit_rgts_v2.yaml | **ViT-RGTS v2** | **16** | **Main experiment** |
| 14 | baseline_vit_rgts_v2.yaml | ViT-RGTS v2 | 0-16 | Repeat for significance |
| 15 | finetune_torchvision_vit.yaml | ViT-B/16 | 0 | LLRD fine-tuning |
| 16 | finetune_trocr.yaml | TrOCR-base | 0 | LLRD fine-tuning |

### 2.2 Primary Comparisons

```
Comparison 1: Register Sweep (Runs 09-13)
  Goal: Isolate effect of register count on attention quality
  Control: Same architecture, optimizer, data, augmentation
  Variable: num_registers ∈ {0, 2, 4, 8, 16}

Comparison 2: Architecture Classes (Runs 01 vs 11 vs 07 vs 15)
  Goal: Compare from-scratch vs pretrained vs fine-tuned
  CNN-RNN (01) vs ViT-RGTS-v2 (11) vs ViT-B/16 (07) vs Fine-tuned ViT (15)

Comparison 3: v1 vs v2 Architecture (Runs 02-06 vs 09-13)
  Goal: Validate CNN stem improvement over raw patch embedding
  Matched register counts, different backbone design
```

---

## 3. Execution Protocol

### 3.1 Single Experiment Execution

```bash
# Navigate to project root
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline

# Activate environment
source .venv/bin/activate

# Run single experiment
python scripts/trainer.py configs/baseline_vit_rgts_v2.yaml arch.num_registers=4

# Via SLURM
sbatch experiments_execution/slurm_scripts/11_vit_rgts_v2_4reg.slurm
```

### 3.2 Batch Execution

```bash
# All 16 experiments
bash experiments_execution/submit_all_experiments.sh

# ViT-RGTS v2 register sweep only (runs 09-13)
bash experiments_execution/submit_vit_rgts_v2_sweep.sh

# Single SLURM job
bash experiments_execution/submit_single.sh experiments_execution/slurm_scripts/11_vit_rgts_v2_4reg.slurm
```

### 3.3 Monitoring

```bash
# Check SLURM job status
squeue -u $USER

# Watch training progress
tail -f saved_models/experiments/run_XX/training.log

# Interactive dashboard (if available)
streamlit run backup/dashboard/app.py
```

---

## 4. Output Collection Protocol

### 4.1 Per-Run Outputs

Each `saved_models/experiments/run_XX/` contains:

| File | Content | Used For |
|------|---------|----------|
| `config.json` | Full experiment configuration | Reproducibility |
| `model.pt` | Best model checkpoint | Inference, visualization |
| `model_swa.pt` | SWA-averaged checkpoint | Comparison with base |
| `results.csv` | Per-epoch metrics | Training curves |
| `evaluation_details.csv` | Per-sample predictions | Error analysis |
| `training.log` | Full training transcript | Debugging |
| `attention_weights/` | Attention maps at epochs 1,5,10,...,80 | Evolution analysis |

### 4.2 Aggregation

```bash
# Generate unified results table
python scripts/postprocessing/aggregate_results.py

# Output: CSV table + LaTeX-formatted comparison
```

### 4.3 Analysis Workflow

```
Step 1: Training Curves
  → scripts/postprocessing/comparative/plot_training_metrics.py
  → Output: Loss/CER/WER convergence comparison

Step 2: Register Impact
  → scripts/postprocessing/comparative/analyze_register_impact.py
  → Output: 6-panel bar chart (CER, WER, entropy, Gini, localization, overlap)

Step 3: Attention Quality
  → scripts/postprocessing/comparative/attention_quality_metrics.py
  → Output: Per-register quantitative metrics

Step 4: Publication Figures
  → scripts/pub_fig1_paper_fig5.py (character attention grid)
  → scripts/pub_fig2_register_comparison.py (register comparison)
  → scripts/pub_register_quantitative.py (quantitative panels)
  → scripts/postprocessing/beyond_memorization_viz.py (blob-thresholded maps)
```

---

## 5. Statistical Analysis Plan

### 5.1 Performance Comparison

| Metric | Statistical Test | Rationale |
|--------|-----------------|-----------|
| CER between register counts | Paired t-test (or Wilcoxon) | Same test set, per-sample pairing |
| Attention entropy vs registers | Spearman rank correlation | Ordinal register count |
| CER improvement significance | Bootstrap CI (95%) | Robust for non-normal distributions |

### 5.2 Effect Size Reporting

For each comparison, report:
1. **Mean difference** (Δ CER, Δ entropy)
2. **95% confidence interval** (bootstrap or t-distribution)
3. **Effect size** (Cohen's d for CER; Spearman ρ for attention metrics)

### 5.3 Ablation Analysis

| Ablation | What It Tests |
|----------|--------------|
| Remove registers (0 vs 4) | Core hypothesis |
| Remove CNN stem (v1 vs v2) | Architectural contribution |
| Remove dual head (CTCtopR vs CTCtopB) | Dual supervision value |
| Remove warmup (direct vs warmup+cosine) | Scheduler importance |
| Change augmentation (CNN vs ViT) | Augmentation strategy |

---

## 6. Visualization Protocol

### 6.1 Required Figures for Publication

| Figure | Script | Content |
|--------|--------|---------|
| Fig. 1 | `pub_fig1_paper_fig5.py` | Character-aligned attention heatmaps (no-reg vs 4-reg) |
| Fig. 2 | `pub_fig2_register_comparison.py` | Register sweep (0/4/8/16) with entropy annotations |
| Fig. 3 | `pub_register_quantitative.py` | 4-panel quantitative: entropy, ρ, aligned-heads, norms |
| Fig. 4 | `pub_gradcam_analysis.py` | GradCAM + gradient-weighted attention |
| Fig. 5 | `beyond_memorization_viz.py` | Blob-thresholded per-character maps (ICDAR style) |

### 6.2 Supplementary Analysis

| Analysis | Script | Output |
|----------|--------|--------|
| Training curves | `plot_training_metrics.py` | Convergence comparison |
| Attention rollout | `rollout_comparison.py` | Layer-accumulated attention |
| Head specialization | `head_specialization.py` | Per-head character focus |
| t-SNE registers | `tsne_register_tokens.py` | Register embedding space |
| Writer ID | `writer_identification/evaluate.py` | Retrieval accuracy |

---

## 7. Reproducibility Checklist

- [x] All configs versioned in `configs/`
- [x] Full config.json saved per run
- [x] SLURM scripts capture environment (module loads, paths)
- [x] File locking prevents race conditions
- [x] Per-sample evaluation enables exact result reproduction
- [x] requirements.txt pins major package versions
- [x] Data preparation scripts deterministic
- [x] Augmentation uses Albumentations (reproducible with seed)
- [x] Models saved with complete state_dict

---

## 8. Active vs Archived Components

### ACTIVE (Use These)

| Component | Path |
|-----------|------|
| Training | `scripts/trainer.py` + `configs/*.yaml` |
| Models | `models.py` |
| Data utilities | `utils/` |
| Preprocessing | `scripts/preprocessing/` |
| Postprocessing | `scripts/postprocessing/` |
| Publication figures | `scripts/pub_*.py` |
| Experiments | `experiments_execution/slurm_scripts/` |
| Synthetic data | `synthetic_data_generation/` |
| Attention analysis | `scripts/visualization/` |
| Writer ID | `scripts/writer_identification/` |

### ARCHIVED (Reference Only)

| Component | Path | Reason |
|-----------|------|--------|
| Early visualizations | `archive/legacy_visualization/` | Superseded by scripts/postprocessing/ |
| v1 postprocessing | `archive/legacy_postprocessing_v1/` | Superseded by current scripts |
| Experimental notebooks | `archive/experimental_notebooks/` | Development artifacts |
| Old documentation | `archive/legacy_documents/` | Replaced by documents/latest/ |
| Reference papers | `archive/papers/` | Not source code |

---

## 9. Quick Reference Commands

```bash
# Validate environment
python scripts/preprocessing/validate_setup.py

# Run primary experiment (ViT-RGTS v2, 4 registers)
python scripts/trainer.py configs/baseline_vit_rgts_v2.yaml

# Run register sweep
for r in 0 2 4 8 16; do
    python scripts/trainer.py configs/baseline_vit_rgts_v2.yaml arch.num_registers=$r
done

# Evaluate specific run
python scripts/postprocessing/single_model/evaluate.py --run saved_models/experiments/run_11

# Generate all publication figures
python scripts/pub_fig1_paper_fig5.py
python scripts/pub_fig2_register_comparison.py
python scripts/pub_register_quantitative.py
python scripts/postprocessing/beyond_memorization_viz.py

# Aggregate all results
python scripts/postprocessing/aggregate_results.py

# Attention analysis
python scripts/postprocessing/attention_viz/generate_all.py
```
