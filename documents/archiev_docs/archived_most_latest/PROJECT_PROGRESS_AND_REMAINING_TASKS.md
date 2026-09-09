# HTR Explainability Project — Progress & Remaining Tasks

**Project**: Improving HTR Attention Maps via Register Tokens  
**Objective**: Make Handwritten Text Recognition explainable for downstream tasks (writer identification) using fine-grained, meaningful attention maps  
**Deadline**: ~10 days remaining  
**Date**: April 21, 2026

---

## 1. Project Overview (In Brief)

This project integrates **register tokens** (from "Vision Transformers Need Registers", Darcet et al. 2023) into an HTR system based on [HTR-best-practices](https://github.com/georgeretsi/HTR-best-practices/) to produce cleaner, more interpretable attention maps. The improved attention maps are intended for downstream **writer identification** (inspired by VLAC — "Interpretable Writer Recognition via Vectors of Locally Aggregated Characters").

**Core Pipeline**:
1. Train HTR models (CNN-RNN baseline + ViT-RGTS with 0–16 register tokens)
2. Extract and visualize attention maps (character-level, register-level)
3. Demonstrate explainability benefits for downstream writer identification
4. Compare architectures and register configurations

---

## 2. What Has Been Completed ✅

### 2.1 Architecture & Code Implementation
| Component | Status | Details |
|-----------|--------|---------|
| CNN-RNN-CTC Baseline | ✅ Done | 7.4M params, CER ~4.3% |
| ViT-RGTS v1 (from scratch) | ✅ Done | Small model, 16×16 patches |
| ViT-RGTS v2 (CNN stem) | ✅ Done | Critical fix: left-to-right token order for CTC |
| Register token support (0–16) | ✅ Done | Configurable via YAML |
| TorchVision ViT (pretrained) | ✅ Done | ImageNet weights, register support |
| TrOCR encoder config | ✅ Done | Config exists, NOT yet trained |
| CTC heads (CNN/RNN/Both) | ✅ Done | Dual supervision (CTCtopB) |
| `forward_explain()` method | ✅ Done | Per-layer attention, token norms, entropy |

### 2.2 Training & Experiments
| Item | Status | Details |
|------|--------|---------|
| 25 experiment runs (run_47–71) | ✅ Done | All 80 epochs, full metrics |
| Register sweep (0,2,4,8,16 regs) | ✅ Done | Runs 50–54 and 55–71 |
| TorchVision ViT experiments | ✅ Done | Run 47 (vit_b_16, 4 regs) |
| Data augmentation pipeline | ✅ Done | CNN (moderate), ViT (strong), ViT-strong |
| IAM dataset prep (Aachen split) | ✅ Done | Train/val/test splits |
| SLURM execution scripts (8 jobs) | ✅ Done | Automated submission |
| Training curves & plots | ✅ Done | Loss, CER, WER, LR plots |
| EDA (dataset analysis) | ✅ Done | Character freq, writer dist, image dims |

**Best Results**:
| Run | Architecture | Registers | Test CER | Test WER |
|-----|-------------|-----------|----------|----------|
| 71 | ViT-RGTS v2 | 16 | **6.10%** | **19.73%** |
| 55 | ViT-RGTS v2 | 0 | 6.14% | 20.58% |
| 60 | ViT-RGTS v2 | 13 | 6.15% | 20.10% |
| CNN-RNN (historic) | Baseline | N/A | ~4.3% | ~16% |

### 2.3 Attention Visualization & Explainability
| Component | Status | Details |
|-----------|--------|---------|
| Character-level attention maps | ✅ Done | Paper Fig.5-style grids |
| Register token attention analysis | ✅ Done | Register-to-patch, patch-to-register |
| Attention rollout | ✅ Done | Accumulated multi-layer attention |
| GradCAM visualization | ✅ Done | Gradient-based attribution |
| Token norm heatmaps | ✅ Done | L2 norm importance maps |
| t-SNE register embeddings | ✅ Done | Embedding space visualization |
| Cross-model comparison (run 55 vs 71) | ✅ Done | Full comparison with plots |
| Attention quality metrics | ✅ Done | Entropy, focus analysis |
| CLS/register attention comparison | ✅ Done | Impact analysis |
| 12+ visualization directories | ✅ Done | All populated with outputs |

### 2.4 Infrastructure
| Component | Status |
|-----------|--------|
| Streamlit dashboard (app.py) | ✅ Done |
| Attention explorer (interactive) | ✅ Done |
| 15+ postprocessing scripts | ✅ Done |
| Experiment tracker notebook | ✅ Done |
| Comprehensive documentation (21 docs) | ✅ Done |

### 2.5 Synthetic Data Generation Pipeline
| Component | Status |
|-----------|--------|
| Font crawling script | ✅ Code done |
| Text extraction (CC100) | ✅ Code done |
| LMDB rendering pipeline | ✅ Code done |
| **Actual data generation** | ❌ Not executed |

---

## 3. What Remains To Be Done ❌

### Priority 1 — CRITICAL (Must complete)

| # | Task | Why | Effort |
|---|------|-----|--------|
| **R1** | Write project report/thesis | No .tex or .docx exists yet | 4–5 days |
| **R2** | Final attention map analysis & interpretation | Vincent's primary interest; need clear comparative conclusions about register impact on attention quality | 1 day |
| **R3** | Character attention maps aligned with Fig.5 (Beyond Memorization paper) — final polished version | Explicitly requested by supervisor multiple times | 0.5 day |

### Priority 2 — IMPORTANT (Should complete)

| # | Task | Why | Effort |
|---|------|-----|--------|
| **R4** | Writer identification segment | Mentioned repeatedly in email updates as upcoming; core downstream task | 1.5 days |
| **R5** | TrOCR fine-tuning on IAM Aachen split | Config exists but never trained; mentioned in emails | 0.5 day (training) |
| **R6** | CNN-RNN baseline re-run with current code | Historical run_32 doesn't exist in saved_models | 0.5 day |

### Priority 3 — NICE TO HAVE (If time permits)

| # | Task | Why | Effort |
|---|------|-----|--------|
| **R7** | Synthetic data integration training | Pipeline code exists, data not generated | 1 day |
| **R8** | Linear probing execution | Guide complete, code in documents/ not scripts/ | 0.5 day |
| **R9** | Pretrained model fine-tuning experiments | Mentioned in emails | 1 day |

---

## 4. 10-Day Implementation Plan

### Day 1–2: Final Experiments & Baseline

**Day 1**: Run missing critical experiments
- [ ] Submit CNN-RNN baseline training (SLURM, ~4h)
- [ ] Submit TrOCR fine-tuning on IAM (SLURM, ~4h)
- [ ] While training: consolidate all existing results into a single comparison table
- [ ] Verify attention extraction works for all completed runs

**Day 2**: Collect results & finalize attention analysis
- [ ] Gather final CER/WER from all runs into master results table
- [ ] Generate polished Fig.5-style character attention grids (run `paper_fig5_ctc.py` for best models)
- [ ] Create side-by-side attention comparison: Reg-0 vs Reg-4 vs Reg-16 (same input samples)
- [ ] Document clear observations: does register token count improve attention map quality?

### Day 3: Writer Identification Segment

- [ ] Implement simple writer identification pipeline:
  1. Extract ViT features (patch tokens) from all test images using `extract_vit_rgts_features.py`
  2. Aggregate features per writer (mean pooling over lines)
  3. Train a simple classifier (KNN or linear SVM) on extracted features
  4. Evaluate: Top-1 accuracy, mAP
  5. Compare: features from Reg-0 vs Reg-4 vs Reg-16 models → show register tokens improve writer ID
- [ ] Generate writer ID results table and basic plots

### Day 4: Writer Identification + Attention Quality Analysis

- [ ] Complete writer ID evaluation across register variants
- [ ] Quantitative attention quality comparison:
  - Attention entropy per layer (lower = more focused)
  - Character alignment score (diagonality of CTC attention)
  - Register token absorption of background attention (artifact reduction)
- [ ] Create summary figures for report

### Day 5–9: Report Writing

**Day 5**: Report structure & introduction
- [ ] Create report template (.tex or .docx)
- [ ] Write: Abstract, Introduction, Related Work
- [ ] Introduction covers: HTR explainability problem, register tokens motivation, VLAC reference

**Day 6**: Method & Implementation
- [ ] Write: Methodology section
  - Architecture descriptions (CNN-RNN, ViT-RGTS, register mechanism)
  - CNN stem fix (V1 → V2 critical improvement)
  - Training setup, augmentation, IAM dataset, Aachen split
- [ ] Include architecture diagrams

**Day 7**: Experiments & Results
- [ ] Write: Experimental Setup section
  - Model configurations table, hyperparameters
  - Register sweep design (0, 2, 4, 8, 16)
- [ ] Write: Results section
  - CER/WER comparison table
  - Attention map quality analysis (entropy, focus, diagonality)
  - Figure 5-style attention visualizations
  - Register impact analysis

**Day 8**: Explainability & Writer ID
- [ ] Write: Explainability Analysis section
  - Character attention maps interpretation
  - Register token behavior (what they learn)
  - Attention rollout and GradCAM findings
- [ ] Write: Writer Identification section
  - Feature extraction approach
  - Writer ID results with register comparison
- [ ] Include all key figures

**Day 9**: Discussion, Conclusion, Polish
- [ ] Write: Discussion (register tokens' effect on interpretability)
- [ ] Write: Conclusion & Future Work
- [ ] References, formatting, figure captions
- [ ] Review and polish entire document

### Day 10: Final Review & Submission

- [ ] Proofread report
- [ ] Verify all figures are correct and properly referenced
- [ ] Final code cleanup (remove debug prints, organize outputs)
- [ ] Prepare submission package

---

## 5. Key Supervisor Requests (from Email Thread)

| Date | Request | Status |
|------|---------|--------|
| Jan 5, 2026 | Create character-based attention maps like Fig.5 of 145 Submission.pdf | ✅ Done (paper_fig5_ctc.py) |
| Feb 23, 2026 | "Have you had a look at the attention maps? I am mostly interested in those" | ✅ Done (12+ viz dirs) |
| Feb 24, 2026 | "Why not visualize the attention maps? Orientate on the paper" (Fig.5 style) | ✅ Done |
| Nov 11, 2025 | "Not interested in other XAI. Focus on improving attention maps with register tokens" | ✅ Focus maintained |
| Jan 20, 2026 | Fine-tune TrOCR on IAM Aachen split | ❌ Not done |
| Multiple dates | Writer identification implementation | ❌ Not done |
| Multiple dates | Synthetic data integration | ❌ Not done |
| Mar 9, 2026 | Attention map analysis for register variations (0–16) | ✅ Done |

**Supervisor's core interest**: Clean, interpretable attention maps. Register tokens' impact on attention quality. Character-level attribution.

---

## 6. File Quick Reference

| Purpose | Path |
|---------|------|
| Main model code | `models.py` |
| Training script | `scripts/trainer.py` |
| Postprocessing (15 scripts) | `scripts/postprocessing/single_model/` |
| Comparative analysis | `scripts/postprocessing/comparative/` |
| All configs | `configs/` |
| Experiment results | `saved_models/experiments/run_47–71/` |
| Visualizations output | `visualizations/` |
| Dashboard | `dashboard/app.py` |
| Attention explorer | `dashboard/attention_explorer.py` |

---

## 7. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| TrOCR training fails | Config validated; freeze encoder, only train CTC head |
| Writer ID results are weak | Even negative results are reportable; focus on feature quality comparison |
| Report takes too long | Use existing 21 documentation files as drafting source |
| GPU queue delays | Submit jobs early (Day 1); use existing 25 runs if new ones don't finish |
| Synthetic data too slow | Skip (Priority 3); sufficient results without it |
