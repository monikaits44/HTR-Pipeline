# Project Status & Action Plan — Improving HTR Attention Maps

> **Created**: 2026-06-14  
> **Based on**: Complete email thread (Jul 2025 – Jun 2026) cross-referenced against repo state  
> **Deliverable**: 8-page report (per Vincent, May 18 2026)  
> **Core interest**: Attention maps and interpretability via register tokens

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Timeline Reconstruction from Email Thread](#2-timeline-reconstruction)
3. [What Is Done — Verified in Repo](#3-what-is-done)
4. [What Is Remaining — Open Items](#4-what-is-remaining)
5. [Architecture Decision Record](#5-architecture-decision-record)
6. [Action Items — Detailed & Step-by-Step](#6-action-items)
7. [Recommended Execution Sequence](#7-recommended-execution-sequence)
8. [Report Structure Proposal](#8-report-structure-proposal)
9. [Key Quotes from Vincent](#9-key-quotes-from-vincent)

---

## 1. Executive Summary

### Project Goal
Improve attention map quality and interpretability in Handwritten Text Recognition (HTR) using **register tokens** from "Vision Transformers Need Registers" (Darcet et al. 2023), applied to a CNN-stem + Transformer architecture trained on the IAM dataset.

### Current State (June 14, 2026)

| Category | Status |
|---|---|
| Model architectures (4 types) | ✅ Complete |
| Training pipeline (all features) | ✅ Complete |
| Register experiments (0/2/4/8/16) | ✅ Complete (~100 runs) |
| Attention visualization (35+ scripts) | ✅ Complete |
| Writer identification (VLAC pipeline) | ✅ Complete |
| Synthetic data generation (1M images) | ✅ Complete |
| CER/WER evaluation pipeline | ✅ Complete |
| Beyond-Memorization code integration | ❌ Not started |
| Unified results comparison table | ❌ Not created |
| **8-page report (THE deliverable)** | ❌ Only Introduction started |

### Bottom Line
Engineering is ~90% done. The remaining 10% is what Vincent actually asked for: **(1)** the Beyond-Memorization attention-map comparison and **(2)** the 8-page report.

---

## 2. Timeline Reconstruction from Email Thread

### Phase 1: Foundation (Jul – Oct 2025)
| Date | What was done |
|---|---|
| Jul 14 | Project started — literature review of 3 core papers |
| Aug 11–18 | Paper reviews complete: ViT Need Registers, VLAC, Beyond-Memorization |
| Aug 25 | Studied HTR Best Practices repo, cloned it locally |
| Oct 13 | Resumed after exam break. IAM dataset downloaded, data prep complete |
| Oct 20 | First 10-epoch local run. CNN-RNN baseline running |
| Oct 27 | CNN/RNN attention visualization implemented (spatial + temporal heatmaps) |

### Phase 2: ViT Integration (Nov – Dec 2025)
| Date | What was done |
|---|---|
| Nov 3 | HPC access configured. Repo modularization started |
| Nov 10 | XAI exploration (LIME, SHAP, LRP). Vincent redirected: **"focus on register tokens"** |
| Nov 17 | Register-based transformer implementation in progress |
| Nov 24 | ViT-RGTS integrated. Feature extraction pipeline done. Token-norm heatmaps |
| Nov 24 | Vincent: overlay y-axis distortion bug identified |
| Dec 1 | Bug fixed: patch-aligned expansion replaced bilinear upscaling |
| Dec 22 | Resumed after presentation break |
| Dec 29 | ViT models fail to learn: CER=100%, CTC loss 130+. CNN baseline CER=4.7% |

### Phase 3: ViT Fix & Experiments (Jan – Mar 2026)
| Date | What was done |
|---|---|
| Jan 5 | Vincent: **"create character-based attention maps as in Fig 5"** |
| Jan 12 | ViT attention extraction working. 4 arch types finalized |
| Jan 19 | Asymmetric patches (4×64) tested. Enhanced augmentation. ViT still underperforms |
| Jan 25 | Experiment format defined. 30-epoch uniform runs across models |
| Feb 9 | Synthetic data pipeline executed (font crawling → CC100 → LMDB rendering) |
| Feb 16 | 50-epoch runs with augmentation/patch modifications. Character viz inconsistencies |
| Feb 23 | **ViT convergence fixed** — CNN-stem v2. All register models trained 80 epochs |
| Feb 23 | Vincent: **"I am mostly interested in attention maps"** |
| Feb 24 | Vincent: **"Why not visualize the attention maps? Please orientate on the paper"** |
| Mar 2 | Register sweep analysis: short words → strong diagonal attention. Reg > no-reg |
| Mar 9 | 0–16 register experiments collected. Attention rollout analysis |
| Mar 16 | Exams (ADL, SMAI). Paused until April 4 |

### Phase 4: Advanced Features (Apr – May 2026)
| Date | What was done |
|---|---|
| Apr 13 | Resumed work |
| Apr 20 | Register attention analysis in progress |
| Apr 27 | **Writer identification** (VLAC) implemented — mAP, Top-1/Top-5 |
| Apr 27 | Key insight: paper uses cross-attention; our model uses self-attention (CTC encoder-only) |
| May 4 | Fine-tuning pipeline: LLRD, gradual unfreezing. TrOCR 66.5%→13.2% CER |
| May 11 | Synthetic data integration complete (1M segment). Full pipeline end-to-end |
| May 18 | Vincent: **"I need an 8 page report where you show experiments and results"** |
| May 25 | Contacted Beyond-Memorization author. Code release pending |

### Phase 5: Finalization (Jun 2026)
| Date | What was done |
|---|---|
| Jun 1 | Author follow-up. Repo restructuring continues |
| Jun 8 | **Code received from Beyond-Memorization author**. Report Introduction started |
| Jun 14 | Architecture frozen (CNN-stem + ViT-RGTS v2). Augmentation trimmed. CTCtopLinear removed |

---

## 3. What Is Done — Verified in Repo

### 3.1 Model Architectures ✅

| Architecture | Class | File | Status |
|---|---|---|---|
| CNN-RNN (baseline) | `CNN` + `CTCtopR` | `models.py` | ✅ CER ~4.7% (best baseline) |
| **ViT-RGTS v2** (primary) | `ViTRGTSBackbone` (CNN stem) | `models.py` | ✅ Frozen — 6M params |
| TorchVision ViT | `TorchVisionViTBackbone` | `models.py` | ✅ ImageNet ViT-B/16 fine-tuned |
| TrOCR | `TrOCREncoderBackbone` | `models.py` | ✅ Encoder-only, CER 13.2% with LLRD |

**CTC Heads** (after cleanup June 14):
- `CTCtopC` — Conv projection (CNN-RNN only)
- `CTCtopR` — BiLSTM/GRU (single supervision)
- `CTCtopB` — BiLSTM + CNN shortcut (dual supervision, **primary for ViT-RGTS v2**)
- ~~`CTCtopLinear`~~ — Removed (unused, no advantage)

### 3.2 Training Pipeline ✅

| Feature | File | Status |
|---|---|---|
| Main trainer | `scripts/trainer.py` | ✅ Supports all 4 architectures |
| Fine-tuning (LLRD, gradual unfreeze) | `utils/finetuning.py` | ✅ |
| Synthetic data mode | `scripts/trainer.py` (data.mode='synthetic') | ✅ |
| Augmentation | `utils/transforms.py` | ✅ Trimmed to 2 strategies: `cnn` + `vit` |
| Auto experiment tracking | `scripts/trainer.py` | ✅ File-locked auto-increment runs |
| Attention extraction | `models.py` `forward_explain()` | ✅ Per-layer [B,H,S,S] attention maps |

**Augmentation cleanup (June 14):**
- Kept 5 proven transforms (Affine, Elastic/Grid, Morphological, BrightnessContrast, Gamma)
- `aug_transforms_vit` adds GaussNoise and higher probabilities
- Removed `aug_transforms_vit_strong` entirely (overkill for 6M-param model on 6.5K data)
- Removed CoarseDropout (destroys characters), Blur/Sharpen/Perspective (marginal)

### 3.3 Register Experiments ✅

| Experiment Set | Register Counts | Config | SLURM Scripts | Runs |
|---|---|---|---|---|
| ViT-RGTS v2 sweep | 0, 2, 4, 8, 16 | `baseline_vit_rgts_v2.yaml` | `slurm_scripts/09–13` | ~run_50–54 (main) |
| ViT-RGTS v1 sweep | 0, 2, 4, 8, 16 | `baseline_vit_rgts.yaml` | `slurm_scripts/02–06` | Early runs |
| Full sweep script | All | `submit_vit_rgts_v2_sweep.sh` | Combined | Batch |

~100+ completed runs in `saved_models/experiments/run_60/` through `run_101/`.

### 3.4 Attention Map Visualization ✅

**35+ scripts across 4 locations:**

| Location | Scripts | Purpose |
|---|---|---|
| `scripts/postprocessing/single_model/` | 14 scripts | Per-model attention: character, register, GradCAM, token norms, t-SNE |
| `scripts/postprocessing/comparative/` | 8 scripts | Register comparison, attention quality metrics (entropy, diagonality) |
| `scripts/postprocessing/` (top-level) | `character_attention_viz.py` + 3 others | Beyond-Memorization-style CTC-aligned viz |
| `scripts/` (publication) | `pub_fig1_paper_fig5.py` through `pub_fig3_*.py` | Publication-ready figure generation |

**Generated outputs (40+ PDFs/PNGs):**
- `outputs/pub_figures/` — fig5 character attention, register comparison, gradient saliency
- `outputs/viz_gradcam/` — GradCAM + attention rollout per sample
- `outputs/attention_maps/` — per-head, per-layer, combined, register comparison

**Key script** (fixed June 14): `scripts/postprocessing/character_attention_viz.py`
- Approach A: CTC-aligned per-character heatmaps (Beyond-Memorization style)
- Approach B: Register token spatial attention (per head)
- Approach C: Self-attention flow matrix (all layers)

### 3.5 Writer Identification (VLAC) ✅

| Component | File | Status |
|---|---|---|
| VLAC encoding | `scripts/writer_identification/vlac.py` | ✅ |
| Feature extraction | `scripts/writer_identification/embeddings.py` | ✅ |
| Retrieval (cosine dist) | `scripts/writer_identification/retrieval.py` | ✅ |
| Evaluation (mAP, CMC) | `scripts/writer_identification/evaluate.py` | ✅ |
| Visualization | `scripts/writer_identification/visualize.py` | ✅ |

### 3.6 Synthetic Data Generation ✅

7-step pipeline in `synthetic_data_generation/`:
1. Font crawling (Google Fonts, 9367 valid) → 2. Download → 3. IAM length distribution matching → 4. CC100 text preprocessing → 5. Font verification → 6. LMDB rendering (~1M images)

Full documentation: `synthetic_data_generation/SYNTHETIC_DATA_PIPELINE_DOCUMENTATION.md`

### 3.7 Evaluation Pipeline ✅

- `utils/metrics.py` — CER class, WER class (Levenshtein distance)
- `scripts/postprocessing/single_model/evaluate.py` — Per-run evaluation
- `evaluation_execution/model_evaluation.sh` — SLURM batch evaluation
- Per-run output: `results.csv` + `evaluation_details.csv`

### 3.8 Notebooks (20+) ✅

`notebook/` contains 20+ Jupyter notebooks for attention exploration, GradCAM, character visualization, register analysis. These are development/exploration artifacts.

---

## 4. What Is Remaining — Open Items

### 4.1 ❌ Beyond-Memorization Code Integration
- **Status**: Code received June 8, understanding documented, NOT integrated
- **The gap**: Their code uses cross-attention (UNet, char slots × image spatial), producing direct per-char spatial maps. Our model uses self-attention (encoder-only, CTC). Cannot copy directly.
- **What's needed**: Port their *rendering style* (inferno colormap, blob thresholding via mean+std, connected-components bounding box) onto our existing CTC-aligned Approach A output. Produce side-by-side comparison figure.

### 4.2 ❌ Unified Results Comparison Table
- **Status**: Each run has its own `results.csv`. No aggregated comparison exists.
- **What's needed**: One CSV/LaTeX table: rows = all architectures × register counts, columns = params, train CER, val CER, test CER, test WER.

### 4.3 ❌ 8-Page Report (THE Deliverable)
- **Status**: Only Introduction started (per June 8 email)
- **Vincent's requirement** (May 18): "I need an 8 page (can of course also be larger) report, where you show your experiments and results"
- **No .tex, .pdf, or .docx exists in the repo**

### 4.4 ⚠️ Final Figure Curation
- **Status**: 40+ PDFs generated but not curated for report inclusion
- **What's needed**: Select 5–6 best figures, verify they use the correct best-run model, ensure consistent formatting

### 4.5 ⚠️ Register Effect Narrative
- **Status**: Findings exist in scattered markdown docs (Mar 2026 emails: short words → strong diagonal, registers improve diagonality)
- **What's needed**: Consolidate into a quantified narrative with specific metrics (entropy, diagonality scores)

---

## 5. Architecture Decision Record

### Final Architecture: ViT-RGTS v2

```
Input [B, 1, 128, 1024] (grayscale IAM line)
    │
    ▼
CNN Stem (4 conv layers)
    Conv1 stride(4,2): [B,  32, 32, 512]
    Conv2 stride(2,2): [B,  64, 16, 256]
    Conv3 stride(2,2): [B, 128,  8, 128]
    Conv4 stride(2,1): [B, 256,  4, 128]
    MaxPool(H→1):      [B, 256,  1, 128]  → 128 column tokens, left-to-right
    │
    ▼
Transformer Encoder (6 layers, d=256, 8 heads, mlp=1024)
    + 4 Register Tokens (learnable, capture global style/writer info)
    Self-attention over [4 registers + 128 column tokens] = 132 tokens
    │
    ▼
CTCtopB Head (3-layer BiLSTM, hidden=256 + CNN shortcut)
    Dual supervision during training (RNN + CNN paths)
    │
    ▼
CTC Loss → Greedy decode → text
```

**Parameters**: ~6M backbone + ~2.9M head = ~8.9M total

### Why CNN Stem (Not Raw Patches)

| Approach | Token sequence | CTC result |
|---|---|---|
| v1: Raw 16×16 patches | 8×64 grid flattened row-major → columns interleaved across 8 rows | **CER stuck at 75%** (Dec 2025) |
| v2: CNN stem | 128 column tokens in strict left→right order | **CER competitive** (Feb 2026) |

The CTC alignment constraint requires monotonically left-to-right tokens. CNN stem's height collapse to 1 guarantees this. Every production HTR system (CRNN, TrOCR, PARSeq) collapses height before the decoder.

### Why Not U-Net + ViT
- U-Net is for pixel-level reconstruction/segmentation, not sequence recognition
- Skip connections re-inject spatial details the CTC decoder doesn't need
- 2–3× more parameters → overfitting on 6.5K samples
- Wrong tool for discriminative CTC-based HTR on pre-segmented line images

### CTC Head Selection

| Head | Use case | Rationale |
|---|---|---|
| `CTCtopC` | CNN-RNN arch | Simple conv projection, original baseline |
| `CTCtopR` | Single supervision | BiLSTM adds sequential context |
| **`CTCtopB`** | **ViT-RGTS v2 (primary)** | Dual CTC loss prevents transformer from ignoring sequence |
| ~~`CTCtopLinear`~~ | Removed June 14 | No advantage over CTCtopR, unused by any config |

### Augmentation (After Cleanup June 14)

**Two strategies only:**

| Strategy | Used by | Transforms |
|---|---|---|
| `aug_transforms_cnn` (moderate) | CNN-RNN, pretrained ViT/TrOCR | Affine(±1°), Elastic/Grid, Morph, BrightContrast, Gamma |
| `aug_transforms_vit` (strong) | ViT-RGTS (from scratch) | Affine(±10°), Elastic/Grid, Morph, BrightContrast, Gamma, GaussNoise |

Removed: `aug_transforms_vit_strong` (overkill), CoarseDropout (destroys chars), Blur/Sharpen/Perspective (marginal).

---

## 6. Action Items — Detailed & Step-by-Step

### ACTION 1 🔴 [CRITICAL] — Integrate Beyond-Memorization Visualization Style
**Priority**: Highest — this is the recurring ask from Vincent since Jan 5, 2026  
**Effort**: ~1 day  
**Depends on**: Nothing (code received)

**Steps:**
1. Clone the released Beyond-Memorization repo into `external/beyond_memorization/`:
   ```bash
   mkdir -p external && cd external
   git clone https://github.com/aniketntnu/Beyond-Memorization.git beyond_memorization
   ```
2. Study `utils/saveAttentionMaps.py` → function `save_Attention2_with_blobs()`.
3. Port their rendering style into `scripts/postprocessing/character_attention_viz.py` Approach A:
   - Inferno colormap (already done ✅)
   - Blob thresholding: `threshold = mean + std` (already in doc, need to apply)
   - Connected components via `scipy.ndimage.label()` → strongest blob → bounding box
   - Per-character PNG: image with heatmap overlay + character label + bounding box
4. Add a **Beyond-Memorization comparison figure**: their generated-image char attention vs our CTC-aligned recognition attention, side by side.
5. Run on 3–5 test samples to produce publication figures.
6. Output to `outputs/pub_figures/beyond_memorization_comparison/`.

### ACTION 2 🔴 [CRITICAL] — Build Unified Results Table
**Priority**: High — required for report Section 5  
**Effort**: ~2 hours  
**Depends on**: Nothing (data exists)

**Steps:**
1. Create script `scripts/postprocessing/aggregate_results.py`:
   ```python
   # Walk saved_models/experiments/run_*/
   # Parse config.json for arch type, num_registers, params
   # Parse results.csv for best val CER, final test CER, test WER
   # Output: unified_results.csv + LaTeX table
   ```
2. Identify the canonical runs for each configuration:
   - CNN-RNN baseline (best run)
   - ViT-RGTS v2 reg-0, reg-2, reg-4, reg-8, reg-16
   - TorchVision ViT-B/16
   - TrOCR (best LLRD run)
   - Synthetic-pretrained (if run exists)
3. Produce two outputs:
   - `outputs/tables/unified_results.csv`
   - `outputs/tables/unified_results.tex` (LaTeX tabular)

### ACTION 3 🔴 [CRITICAL] — Draft the 8-Page Report
**Priority**: Highest — this IS the deliverable  
**Effort**: ~3–5 days  
**Depends on**: Actions 1 and 2

**Steps:**
1. Create `report/` directory with LaTeX scaffold:
   ```
   report/
   ├── main.tex           # Master document (Springer LNCS or IEEE format)
   ├── references.bib     # BibTeX entries for all 8 papers
   ├── figures/            # Symlink or copy selected pub_figures
   └── tables/             # Symlink or copy from outputs/tables/
   ```
2. Draft sections in order:

   | Section | Pages | Content | Source material |
   |---|---|---|---|
   | **1. Introduction** | ~1 | HTR + interpretability problem. Register tokens as the solution lever. | *(already started)* |
   | **2. Related Work** | ~1 | HTR Best Practices, ViT Need Registers, VLAC, Beyond-Memorization, TrOCR | 8 PDFs in `papers/` |
   | **3. Method** | ~1.5 | Architecture: CNN-stem + ViT + registers + CTCtopB. v1 failure → v2 fix story. | Section 5 of this document |
   | **4. Experimental Setup** | ~1 | IAM Aachen split, 128×1024, augmentation, training schedule, register sweep design | `configs/`, `experiments_execution/` |
   | **5. Results** | ~1.5 | Unified table + CER/WER curves + register sweep analysis | Action 2 output |
   | **6. Attention Analysis** | ~1.5 | **THE core section Vincent wants.** CTC-aligned char maps, register specialization, Beyond-Memorization comparison | Action 1 output + existing viz |
   | **7. Conclusion** | ~0.5 | Summary. Registers improve interpretability, not CER. Future: cross-attention HTR | — |

3. Populate `references.bib` from the 8 papers in `papers/`.
4. Write each section, embedding figures from Actions 1 and 4.

### ACTION 4 🟡 [IMPORTANT] — Curate Final Figures
**Priority**: Medium — needed for report  
**Effort**: ~2 hours  
**Depends on**: Action 1

**Steps:**
1. From `outputs/pub_figures/`, select 5–6 best:
   - **Fig 1**: Architecture diagram (draw or adapt from existing)
   - **Fig 2**: `fig5_char_attention_run_84.pdf` — CTC-aligned character attention
   - **Fig 3**: `fig5_register_comparison.pdf` — reg-0 vs reg-4 attention quality
   - **Fig 4**: `fig4_register_attention.pdf` — what each register attends to
   - **Fig 5**: Beyond-Memorization comparison (from Action 1)
   - **Fig 6**: CER/WER training curves across register counts
2. Verify each figure uses the correct best-run model (cross-check with Action 2).
3. Ensure consistent formatting: inferno colormap, matching aspect ratios, readable labels.
4. Copy selected figures to `report/figures/`.

### ACTION 5 🟡 [IMPORTANT] — Write Register-Effect Narrative
**Priority**: Medium — feeds into report Section 6  
**Effort**: ~3 hours  
**Depends on**: Action 2

**Steps:**
1. Consolidate findings from:
   - `documents/latest/EXPLAINABILITY_INSIGHTS_REGISTER_SWEEP.md`
   - Email thread Mar 2 entry: "short words → strong diagonal, registers → higher diagonality"
   - `scripts/postprocessing/comparative/attention_quality_metrics.py` output
2. Quantify with metrics:
   - **Attention entropy**: registers → lower entropy (more focused)?
   - **Diagonality score**: registers → higher diagonal alignment?
   - **Spatial sparsity**: registers → sparser, cleaner attention maps?
3. Key narrative: "Registers do not significantly change CER (~5% for all), but they produce cleaner, more localized attention patterns. Register tokens absorb global/positional information, freeing patch tokens to attend locally."
4. This directly answers Vincent's Feb 23 question.

### ACTION 6 🟢 [LOW PRIORITY] — Final Repo Hygiene
**Priority**: Low — mostly done  
**Effort**: ~1 hour  

**Steps:**
1. ✅ Architecture frozen (done June 14)
2. ✅ `CTCtopLinear` removed (done June 14)
3. ✅ Augmentation trimmed to 2 strategies (done June 14)
4. ✅ `character_attention_viz.py` bugs fixed (done June 14)
5. Verify `requirements.txt` is current:
   ```bash
   pip freeze | grep -E "torch|albumentations|einops|omegaconf|scipy|matplotlib|scikit" > /tmp/reqs_check.txt
   diff requirements.txt /tmp/reqs_check.txt
   ```
6. Add a short `REPRODUCE.md` with exact commands: train → evaluate → visualize.

---

## 7. Recommended Execution Sequence

```
Week 1 (June 14–20):
  ├── ACTION 2: Build unified results table (2 hrs)
  ├── ACTION 1: Integrate Beyond-Memorization viz style (1 day)
  └── ACTION 4: Curate final figures (2 hrs)

Week 2 (June 21–27):
  ├── ACTION 5: Write register-effect narrative (3 hrs)
  └── ACTION 3: Draft report Sections 1–4 (2 days)

Week 3 (June 28–Jul 4):
  ├── ACTION 3: Draft report Sections 5–7 (2 days)
  ├── ACTION 3: Review and polish (1 day)
  └── ACTION 6: Repo hygiene (1 hr)
```

**Dependencies:**
- Action 1 (Beyond-Mem integration) → needed for report Section 6 figures
- Action 2 (results table) → needed for report Section 5
- Actions 1+2+4+5 → all feed into Action 3 (report)
- Action 6 → independent, lowest priority

---

## 8. Report Structure Proposal

### Suggested Format
- **Template**: Springer LNCS (single-column, 8+ pages) or IEEE two-column
- **Length**: 8 pages minimum (Vincent: "can of course also be larger")

### Outline

```
1. Introduction (1 page)
   - HTR state of the art (CTC-based, attention-based)
   - The interpretability gap: attention maps in HTR are noisy
   - Register tokens (Darcet et al. 2023) as a solution
   - Contributions: (1) hybrid CNN-stem + ViT for CTC-based HTR,
     (2) register token impact on attention map quality,
     (3) CTC-aligned character attention visualization

2. Related Work (1 page)
   - HTR Best Practices (Retsi et al.) — CNN+RNN+CTC baseline
   - Vision Transformers Need Registers (Darcet et al.) — register tokens
   - Beyond Memorization (Gurav et al.) — character-level attention in diffusion
   - VLAC (Raven et al.) — writer identification via character aggregation
   - TrOCR (Li et al.) — transformer-based OCR

3. Method (1.5 pages)
   3.1 Architecture: CNN Stem + Transformer + Registers + CTCtopB
   3.2 Why CNN Stem: the v1 failure story (row-major → CTC incompatible)
   3.3 Register Tokens: absorb global info, clean patch attention
   3.4 CTC-aligned attention visualization (bridging self-attn → char maps)

4. Experimental Setup (1 page)
   4.1 Dataset: IAM Aachen split (6,161 train, 900 val, 1,861 test)
   4.2 Preprocessing: invert → aspect-ratio resize → pad 128×1024
   4.3 Augmentation: 5 proven transforms (Affine, Elastic, Morph, BC, Gamma)
   4.4 Training: AdamW, warmup+cosine, 80 epochs, batch 32 (accum 4)
   4.5 Register sweep: 0, 2, 4, 8, 16 tokens

5. Results (1.5 pages)
   5.1 Unified comparison table (all architectures × register counts)
   5.2 CER/WER curves across training
   5.3 Register count vs performance (conclusion: marginal CER difference)
   5.4 Fine-tuning results (TrOCR: LLRD helps; ViT-B: hurts)

6. Attention Map Analysis (1.5 pages)  ← CORE SECTION
   6.1 CTC-aligned per-character heatmaps (our Fig 5)
   6.2 Register token spatial specialization
   6.3 Comparison with Beyond-Memorization (cross-attn vs self-attn)
   6.4 Quantitative: entropy, diagonality, sparsity across register counts
   6.5 Key finding: registers clean attention without hurting CER

7. Conclusion (0.5 pages)
   - Registers improve attention interpretability, not recognition
   - CNN-stem is critical for CTC compatibility
   - Future: cross-attention HTR for direct char→space mapping

References (~20 entries)
```

---

## 9. Key Quotes from Vincent (Decision Anchors)

These quotes from the email thread define what matters:

| Date | Quote | Implication |
|---|---|---|
| **Nov 11, 2025** | "I am actually not interested in other explainable AI but more on a way to improve the attention maps and thus the interpretability with these register tokens. Please focus on that." | Drop LIME/SHAP/LRP. Only register tokens matter. |
| **Jan 5, 2026** | "Can you create such character-based attention maps as in the paper I sent you? (Figure 5)" | Beyond-Memorization Fig 5 is the target output format. |
| **Feb 23, 2026** | "I am mostly interested in [the attention maps]" | Performance (CER) is secondary. Attention quality is primary. |
| **Feb 24, 2026** | "This looks quite confusing to me. Why not visualize the attention maps? Please orientate on the paper" | Existing viz was too complex. Simplify. Match the paper. |
| **May 18, 2026** | "For the project finalization, I need an 8 page (can of course also be larger) report, where you show your experiments and results." | THE deliverable. Non-negotiable. |

---

## Appendix: Key File Paths

### Configs
- `configs/baseline_vit_rgts_v2.yaml` — Primary architecture config
- `configs/finetune_torchvision_vit.yaml` — ViT-B fine-tuning
- `configs/finetune_trocr.yaml` — TrOCR fine-tuning

### Core Code
- `models.py` — All 4 architectures + CTC heads + forward_explain()
- `scripts/trainer.py` — Training loop with auto-selection
- `utils/transforms.py` — Augmentation (2 strategies after cleanup)
- `utils/finetuning.py` — LLRD + gradual unfreezing
- `utils/metrics.py` — CER/WER computation
- `utils/preprocessing.py` — Image loading + preprocessing

### Visualization
- `scripts/postprocessing/character_attention_viz.py` — 3-approach viz (primary)
- `scripts/pub_fig1_paper_fig5.py` — Publication Fig 5 reproduction
- `scripts/pub_fig2_register_comparison.py` — Register comparison figure
- `scripts/postprocessing/comparative/attention_quality_metrics.py` — Quantitative

### Experiment Infrastructure
- `experiments_execution/slurm_scripts/` — 16 SLURM scripts
- `experiments_execution/submit_vit_rgts_v2_sweep.sh` — Register sweep launcher
- `saved_models/experiments/run_*/` — ~100 completed runs

### Documentation
- `documents/most_most_latest/` — 10 comprehensive guides (including this one)
- `helper/QUICK_START_GUIDE.md` — Command recipes
- `papers/` — 8 reference PDFs
