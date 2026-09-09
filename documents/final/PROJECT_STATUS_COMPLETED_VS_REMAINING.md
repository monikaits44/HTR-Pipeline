# Project Status: Improving HTR Attention Maps

## Project Overview

- **Student**: Monika Radhakisan Chavan
- **Supervisor**: Prof. Vincent Christlein
- **Project Period**: July 2025 – Present (Aug 2026)
- **Core Goal**: Improve attention map interpretability in HTR systems using Vision Transformer register tokens, and benchmark against CNN-RNN baselines
- **Key Deliverable**: 8+ page report with experiments and results (per Vincent's email, 2026-05-18)

---

## 1. COMPLETED WORK (Verified Against Codebase)

### 1.1 Core Training Infrastructure ✅

| Item | Evidence | Status |
|------|----------|--------|
| CNN-RNN-CTC baseline model | `models.py` — HTRNet with `cnn_rnn` arch type | **Complete** |
| ViT-RGTS v1 (register transformer) | `models.py` — `vit_rgts` with patch embedding | **Complete** |
| ViT-RGTS v2 (CNN stem + transformer) | `models.py` — `use_cnn_stem=True`, dim=256, depth=6 | **Complete** |
| TorchVision ViT-B/16 integration | `models.py` — `torchvision_vit` arch | **Complete** |
| TrOCR integration | `models.py` — `trocr` arch | **Complete** |
| HTR-VT (2D grid attention) | `models.py` — `htrvt` arch for READ2016 | **Complete** |
| Unified trainer with YAML configs | `scripts/trainer.py` (1324 lines) + 10 YAML configs | **Complete** |
| Reproducibility seeding | `set_seed()` in trainer.py, `seed: 42` in config | **Complete** |
| SLURM experiment infrastructure | 54 modular scripts in `experiments_execution/slurm_modular/` | **Complete** |
| Auto-incrementing run directories | `saved_models/experiments/run_<N>/` with file locking | **Complete** |

### 1.2 Data Pipelines ✅

| Item | Evidence | Status |
|------|----------|--------|
| IAM dataset preparation (Aachen split) | `data/IAM/processed_lines/` — 6,482 train / 976 val / 2,915 test lines | **Complete** |
| Image preprocessing (aspect-ratio + pad 128×1024) | `utils/preprocessing.py`, `scripts/preprocessing/prepare_iam.py` | **Complete** |
| Data augmentation (CNN + ViT variants) | `utils/transforms.py` — `aug_transforms_cnn`, `aug_transforms_vit` | **Complete** |
| READ2016 dataset integration | `scripts/prepare_read2016.py`, `data/READ2016/processed/` | **Complete** |
| Synthetic data generation pipeline (7 steps) | `synthetic_data_generation/` — 16 scripts, fonts→CC100→LMDB | **Complete** |
| Synthetic-to-IAM format converter | `scripts/extract_synthetic_to_iam_format.py` | **Complete** |
| Synthetic training mode in trainer | `data.mode='synthetic'` in config, implemented in `HTRTrainer.prepare_dataloaders()` | **Complete** |

### 1.3 Training Experiments ✅ (83 total runs)

| Experiment Category | Runs | Key Results | Status |
|-------------------|------|-------------|--------|
| **CNN-RNN baseline** | run_107 (SWA), run_115 (no SWA) | **4.87% CER** (best overall), 5.0% CER | **Complete** |
| **ViT-RGTS v2 register sweep** (0/2/4/8/16 regs) | runs 116, 105, 111, 123, 122 | 6.0–6.8% CER; 8-reg best (5.92%) | **Complete** |
| **ViT-RGTS v2 extended (120 epochs)** | run_135 | 5.86% CER (best ViT-RGTS) | **Complete** |
| **Full register sweep (0–16)** | runs 60–92 | Comprehensive 17-point sweep at 50 & 80 epochs | **Complete** |
| **TorchVision ViT-B/16** | run_121 | 12.7% CER (out-of-domain pretrained) | **Complete** |
| **TrOCR-base** | run_120 | 15.1% CER (50 epochs) | **Complete** |
| **Fine-tuned TrOCR** | run_112 | 14.8% CER (LLRD improved from 66.5% → 14.8%) | **Complete** |
| **CNN stem ablation** | run_126 | 76.1% CER without stem (catastrophic) | **Complete** |
| **Head type / depth / dim ablations** | runs 127–134, 136 | rnn_only, cnn_only, GRU, depth 4/8, dim 128/512 | **Complete** |
| **SWA ablations** | multiple ± SWA pairs | SWA helps CNN-RNN (−0.15%); hurts ViT-RGTS | **Complete** |
| **HTR-VT on READ2016** | run_142 | 4.84% CER / 20.6% WER (2D grid + registers) | **Complete** |

### 1.4 Fine-Tuning Module ✅

| Item | Evidence | Status |
|------|----------|--------|
| LLRD (Layer-wise Learning Rate Decay) | `utils/finetuning.py` — `build_finetune_optimizer()` | **Complete** |
| Gradual unfreezing | `utils/finetuning.py` — `GradualUnfreezer` class | **Complete** |
| Differential weight decay per layer group | Implemented in trainer for stem/transformer/head | **Complete** |
| TrOCR fine-tuning (LLRD) | CER improved 66.5% → 14.8% (run_112) | **Complete** |
| TorchVision ViT-B/16 fine-tuning | Configs + SLURM scripts ready | **Complete** |

**Key finding documented**: LLRD benefits in-domain pretrained (TrOCR) but degrades out-of-domain (ViT-B/16).

### 1.5 Attention Map Visualization ✅

| Item | Evidence | Status |
|------|----------|--------|
| `forward_explain()` in ViT-RGTS | Returns attention maps `[B,H,S,S]`, register tokens, norms | **Complete** |
| Attention weight extraction per epoch | `utils/attention_extractor.py` + saved in `run_<N>/attention_weights/` | **Complete** |
| Character-level attention (CTC-aligned) | `scripts/postprocessing/beyond_memorization_viz.py` — blob-thresholded per-char heatmaps | **Complete** |
| Beyond-Memorization style visualization | Per-character heatmaps matching ICDAR 2025 Fig. 5 aesthetic | **Complete** |
| Register comparison visualizations | `scripts/pub_fig2_register_comparison.py` | **Complete** |
| GradCAM for ViT-RGTS | `scripts/postprocessing/single_model/gradcam_vit_rgts.py` | **Complete** |
| Attention rollout | `scripts/visualization/attn_rollout.py` | **Complete** |
| Quantitative attention metrics | `utils/attention_metrics.py` — entropy, Gini, peak sharpness, localization accuracy | **Complete** |
| Token norm heatmaps | `scripts/postprocessing/single_model/visualize_token_norms.py` | **Complete** |
| t-SNE of register tokens | `scripts/postprocessing/single_model/tsne_register_tokens.py` | **Complete** |
| Patch-aligned overlay fix (Nov 2025) | Replaced incorrect bilinear upscale with patch-aligned expansion | **Complete** |
| HTR-VT BM-style attention with CTC anchoring | `scripts/htrvt_bm_attention.py` — Gaussian col prior fixes sink-token issue | **Complete** |

### 1.6 Publication Figure Scripts ✅

| Script | Output | Status |
|--------|--------|--------|
| `scripts/pub_fig1_paper_fig5.py` | Character attention grid | **Complete** — PDF+PNG in `outputs/pub_figures/` |
| `scripts/pub_fig2_register_comparison.py` | Register sweep comparison grid | **Complete** — PDF+PNG |
| `scripts/pub_fig3_gradcam_quantitative.py` | GradCAM + quantitative metrics | **Complete** — PDF+PNG |
| `scripts/pub_register_quantitative.py` | Attention quality bar charts | **Complete** — PDF+PNG |
| `scripts/pub_gradcam_analysis.py` | GradCAM analysis | **Complete** |
| `scripts/pub_fig5_character_attention.py` | Attention rollout with register effect | **Complete** — PDF+PNG |

### 1.7 Evaluation & Result Aggregation ✅

| Item | Evidence | Status |
|------|----------|--------|
| Per-sample evaluation (CER/WER) | `scripts/postprocessing/single_model/evaluate.py` | **Complete** |
| Unified results CSV (all runs) | `outputs/tables/unified_results.csv` — 21 key runs | **Complete** |
| LaTeX results table | `outputs/tables/unified_results.tex` | **Complete** |
| All-runs raw CSV | `outputs/tables/all_runs.csv` | **Complete** |
| Experiment results analysis | `outputs/EXPERIMENT_RESULTS_AND_INSIGHTS.md` | **Complete** |
| SLURM evaluation harness | `evaluation_execution/model_evaluation.sh` (parameterized) | **Complete** |

### 1.8 Writer Identification Pipeline ✅

| Item | Evidence | Status |
|------|----------|--------|
| VLAC feature extraction | `scripts/writer_identification/vlac.py` | **Complete** |
| Embedding extraction | `scripts/writer_identification/embeddings.py` | **Complete** |
| Retrieval evaluation (mAP/Top-K) | `scripts/writer_identification/retrieval.py`, `evaluate.py` | **Complete** |
| t-SNE / embedding visualization | `scripts/writer_identification/visualize.py` | **Complete** |

### 1.9 Beyond-Memorization Integration ✅

| Item | Evidence | Status |
|------|----------|--------|
| Official repo cloned & patched | `external/Beyond-Memorization/` with local fixes | **Complete** |
| Config + SLURM integration | `configs/beyond_memorization.yaml`, `slurm_modular/beyond_memorization.slurm` | **Complete** |
| IAM data prep (64×256 word images) | `data/IAM/words_64x256/` preprocessed | **Complete** |
| Generated outputs (charIndex 0–3) | `outputs/beyond_memorization/{noChange,charIndex_0..3}/` — 16 word PNGs + attention maps each | **Complete** |
| Understanding of codebase documented | `documents/most_most_latest/BEYOND_MEMORIZATION_CODEBASE_UNDERSTANDING.md` | **Complete** |

### 1.10 Repository & Infrastructure ✅

| Item | Evidence | Status |
|------|----------|--------|
| GitHub repository | Created and structured | **Complete** |
| HPC cluster access | Multiple SLURM jobs executed (runs 60–142) | **Complete** |
| requirements.txt (pinned, UTF-8) | 103 packages, all version-pinned | **Complete** |
| Modular repository structure | Organized scripts/, utils/, configs/, experiments_execution/ | **Complete** |
| Archive of legacy work | `archive/` with 13 organized subdirectories | **Complete** |
| Scientific documentation | `documents/most_most_latest/SCIENTIFIC_DOCUMENTATION.md` (36K) | **Complete** |

---

## 2. KEY FINDINGS & OBSERVATIONS (Verified)

### 2.1 CNN Stem is Critical for ViT-RGTS
- Without CNN stem: **76.1% CER** (run_126) — catastrophic failure
- With CNN stem: **5.86–6.8% CER** — functional models
- Reason: raw patch embedding destroys left-to-right order needed for CTC

### 2.2 Register Token Sweet Spot
- **8 registers** best at 80 epochs (5.92% CER, run_123)
- **4 registers** best at 120 epochs (5.86% CER, run_135)
- 0 registers still functional (6.03% CER) — registers improve but aren't essential for CER
- Primary benefit is attention map quality, not raw CER

### 2.3 SWA Asymmetry
- SWA **helps** CNN-RNN: 5.0% → 4.87% CER (−0.13%)
- SWA **hurts** ViT-RGTS: degrades all register configs except 16-reg
- SWA with pretrained ViT-B/16: catastrophic (72.5% CER) due to scheduler mismatch

### 2.4 Pretrained Models Underperform
- TorchVision ViT-B/16: 12.7% CER (out-of-domain, no fine-tuning helps much)
- TrOCR-base: 14.8% CER (after LLRD fine-tuning; frozen encoder: 74% CER)
- Both significantly worse than CNN-RNN baseline (4.87%)

### 2.5 Attention Map Quality
- Register models show higher diagonality for short words
- 4-reg has sharpest attention (peak sharpness 7.5)
- Sink-token phenomenon in HTR-VT: CTC anchoring fixes localization (error 38.2 → 2.5 cols)

---

## 3. REMAINING / NOT YET COMPLETED

### 3.1 The 8-Page Report 🔴 CRITICAL

| Item | Status | Priority |
|------|--------|----------|
| **8+ page project report** (per Vincent's requirement, 2026-05-18) | **NOT STARTED** — no .tex or .docx found in repo | **CRITICAL** |

Vincent explicitly stated: *"For the project finalization, I need an 8 page (can of course also be larger) report, where you show your experiments and results."*

**Report should include** (based on project scope and email discussions):
1. Introduction — HTR problem, attention maps, register tokens motivation
2. Related Work — HTR-best-practices, Vision Transformers Need Registers, Beyond Memorization
3. Methodology — Architecture (CNN-RNN vs ViT-RGTS v2), register tokens, CNN stem design
4. Experimental Setup — IAM dataset, Aachen splits, hyperparameters, hardware
5. Results — CER/WER tables (use `outputs/tables/unified_results.tex`), ablation analysis
6. Attention Map Analysis — Register effect on attention quality, BM-style visualization, quantitative metrics
7. Discussion — CNN stem criticality, SWA asymmetry, register sweet spot, pretrained model gap
8. Conclusion

**Available assets for the report:**
- LaTeX table: `outputs/tables/unified_results.tex`
- Publication figures: 15+ PDF/PNG pairs in `outputs/pub_figures/`
- Attention map figures: 50+ files in `outputs/attention_maps/`
- GradCAM figures: `outputs/viz_gradcam/`
- BM-style per-char maps: `outputs/beyond_memorization/`, `outputs/htrvt_bm_attention/`
- Experiment analysis: `outputs/EXPERIMENT_RESULTS_AND_INSIGHTS.md`

### 3.2 Synthetic Data Experiments 🟡

| Item | Status | Notes |
|------|--------|-------|
| Synthetic data generation pipeline | **Complete** | 7-step pipeline, 9,367 fonts, 1M images |
| Synthetic training mode in trainer.py | **Complete** | `data.mode=synthetic`, unified charset handling |
| 16 SLURM scripts validated | **Complete** | All source `run_experiment.sh`, all set `data.mode=synthetic` |
| Training on synthetic + IAM evaluation | **Ready to submit** — 16 scripts in `slurm_modular/synthetic/` | Data on HPC scratch (`/home/woody/`), requires `sbatch` |

### 3.3 Consolidated Result Analysis ✅ (IMPLEMENTED)

| Item | Status | Notes |
|------|--------|-------|
| Unified results table (all valid runs) | **Complete** — 43 runs in `all_runs.csv`, 10 canonical in `unified_results.csv` | Enhanced with SWA, mode, head_type columns |
| Register sweep table | **Complete** — `outputs/tables/register_sweep.tex` | 10 rows: 0/2/4/8/16 regs × ±SWA |
| Ablation study table | **Complete** — `outputs/tables/ablations.tex` | 12 variants: stem, heads, depth, dim, epochs, augmentation |
| Architecture comparison table | **Complete** — `outputs/tables/architecture_comparison.tex` | Best per architecture: CNN-RNN, ViT-RGTS, HTR-VT, ViT-B/16, TrOCR |
| Statistical significance testing | **Complete** — paired bootstrap (10K resamples) | `significance_vs_cnn_rnn.tex`, `significance_register_sweep.tex` |

**Key statistical finding**: CNN-RNN+SWA significantly better than all ViT variants (p<0.001). Within ViT-RGTS register sweep, 8-reg is significantly better than 2-reg (p=0.008) and 4-reg/80ep (p<0.001), but NOT significantly different from 0-reg (p=0.088) or 4-reg/120ep (p=0.761).

### 3.4 Final Attention Map Figures for Report ✅ (IMPLEMENTED)

| Item | Status | Notes |
|------|--------|-------|
| Curated figure set | **Complete** — 31 figures in `outputs/report_figures/` | PDF+PNG, standardized naming |
| Figure index | **Complete** — `outputs/report_figures/FIGURE_INDEX.txt` | LaTeX `\includegraphics` ready |
| BM-style per-char samples | **Complete** — 6 BM figures included | Both Beyond-Memorization and HTR-VT styles |

---

## 4. TIMELINE OF KEY MILESTONES (from emails)

| Date | Milestone |
|------|-----------|
| Jul 2025 | Project initiated, literature review started |
| Aug 2025 | Papers reviewed (ViT Registers, VLAC, Beyond Memorization), repo cloned |
| Oct 2025 | IAM dataset prepared, first CNN-RNN training (10 epochs local) |
| Nov 2025 | ViT-RGTS integrated, first attention maps generated, patch-aligned fix |
| Dec 2025 | ViT convergence issue identified (CER=100%), CNN-RNN baseline: 4.7% CER |
| Jan 2026 | Aachen split confirmed, 30-epoch experiments, augmentation studied |
| Feb 2026 | ViT-RGTS v2 (CNN stem) resolved convergence, 80-epoch runs complete |
| Mar 2026 | Register sweep 0–16, attention analysis, exams (ADL, SMAI) |
| Apr 2026 | Writer identification pipeline (VLAC), attention map challenges discussed |
| May 2026 | Fine-tuning (LLRD), synthetic data pipeline complete (1M images), TrOCR experiments |
| Jun 2026 | Beyond-Memorization code obtained, BM integration, HTR-VT on READ2016, codebase modularized |
| Jun 30, 2026 | Last progress email — traveling for thesis, consolidated update planned |
| Aug 28, 2026 | Repository finalized (seed, encoding, archiving, validation) |

---

## 5. SUMMARY

### What is Done (ready for thesis/report)
- **All training experiments completed** — 83 runs across 5 architectures, register sweeps, ablations, SWA variants
- **All visualization pipelines operational** — 6 publication scripts, BM-style attention, GradCAM, rollout
- **All result tables generated** — unified CSV + LaTeX, per-run CSVs
- **All code infrastructure finalized** — reproducible seeds, modular SLURM, clean configs
- **All auxiliary pipelines done** — synthetic data generation, writer identification, Beyond-Memorization integration

### What Remains
1. **🔴 Write the 8+ page report** — the single critical remaining deliverable
2. **🟡 Run synthetic data experiments** — 16 SLURM scripts validated and ready, needs `sbatch` on HPC

### Blockers
- None for report writing — all data, figures, tables, and statistical tests are available
- Synthetic experiments require HPC GPU time (estimated 16 × 4–24 hours)
