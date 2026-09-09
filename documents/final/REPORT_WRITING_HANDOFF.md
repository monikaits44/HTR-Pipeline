# Report-Writing Handoff & Tracking Document

> **Purpose**: Single reference to carry into the report-writing thread. Contains
> (1) live execution status of this experiment thread, (2) the Step-0 deliverables
> required by `report_writing_instruction/report_writing_instrcution_set.txt`
> (problem statement summary, paper→section map, title options, artifact inventory),
> and (3) a section-by-section pointer to exactly which file/table/figure/run backs
> each claim. Paste/reference this at the start of the report-writing thread.
> **Last updated**: 2026-09-07

---

## 1. Live Execution Status (check before writing Results/Discussion)

| Item | Status |
|------|--------|
| Training runs (63 total, run_143–run_205) | ✅ **100% complete** — 47 to full 150 epochs, 16 synthetic runs to 24h walltime (valid partial checkpoints, disclosed) |
| Aggregate CSV/LaTeX tables (`outputs/tables/`) | ✅ Done — `unified_results.csv/.tex`, `register_sweep.tex`, `ablations.tex`, `architecture_comparison.tex`, `final_results_report.csv` |
| Training curves (6 experiment groups) | ✅ Done — `outputs/report_figures/training_curves*/` |
| Critical findings analysis | ✅ Done — see `KEY_FINDINGS_AND_OBSERVATIONS.md` (this is your primary Discussion-section source) |
| Per-sample batch evaluation (job **1804973**, `batch_eval`, 3rd attempt) | ⏳ **Queued** — 1st attempt failed (evaluate.py path bug). 2nd attempt (job 1803920) ran but only **23/63 runs succeeded, 40 errored** — root cause: the script picked a *generic default config* per architecture type instead of each run's own exact saved config, so every ablation/pretrained-variant run with non-default architecture (different `head_type`, `dim`, `depth`, `use_cnn_stem`, etc.) failed with `Missing key(s) in state_dict`. **Fixed**: script now converts each run's own `config.json` directly to a temporary YAML and evaluates against that exact architecture — bulletproof against any variant. The 23 already-successful evaluations are skipped (not re-run); only the 40 previously-failed runs are retried. |
| Attention visualization pipeline (job **1804972**, `postprocess`, 2nd attempt) | ⏳ **Queued** — 1st attempt (job 1803919) completed Steps 1–3 successfully (attention quality metrics, Fig 2 register comparison, Fig 3 quantitative analysis — **all with REAL data from runs 144/146/151/152**) but Step 4 (GradCAM) crashed with `RuntimeError: cudnn RNN backward can only be called in training mode`, and because the script used `set -e`, this killed Steps 5–8 (Beyond-Memorization single/comparison, head specialization, layer-wise evolution) before they ran. **Fixed**: (a) wrapped the GradCAM backward call in `torch.backends.cudnn.flags(enabled=False)`, (b) removed `set -e` and added `\|\| echo "[WARN] ... continuing"` after every step so one failure can no longer skip the rest. |
| Statistical significance tests (bootstrap CI) | ⚠️ **Stale** — `outputs/tables/statistical_tests.csv` references an **old run set (105–142)**, not the final 143–205 runs. Must be regenerated after `batch_eval` completes (needs `evaluation_details.csv` from every run, which the fixed batch_eval job produces). Do not cite these numbers as final. |
| Real Fig 2 / Fig 3 in `outputs/pub_figures/` | ✅ **Regenerated 2026-09-07** — `fig2_register_comparison.{png,pdf}` and `fig3_register_quantitative.{png,pdf}` now reflect runs 144/146/151/152 (the final register sweep), overwriting the old May-16 (`run_84`-era) placeholders. Safe to cite. |
| Attention quality metrics (entropy/diagonality/peak-sharpness) | ✅ **Ready** — `outputs/report_figures/attention_quality/attention_quality_metrics.csv` + 2 plots, computed over 100 IAM test images × 4 register configs (0/4/8/16). Safe to cite in §5.3. |
| GradCAM figure, BM-style maps, head-specialization, layer-wise evolution | ⏳ **Queued (job 1804972)** — pending completion; check timestamps in `outputs/pub_figures/` and `outputs/report_figures/{attention_bm_single,attention_bm_comparison,head_specialization,layerwise_evolution}/` before citing. |

**Action before starting Results (§5) and Discussion (§6) drafting**: run `squeue -u $USER`. If empty, check `logs/modular/postprocess_1804972.out` and `logs/modular/batch_eval_1804973.out` for the final "DONE" summary line and OK/ERROR counts before citing any attention figure, GradCAM result, or per-sample significance number.

---

## 2. Step 0 Deliverables (per report_writing_instruction Step 0)

### 2.1 Problem statement summary (3–5 sentences)

Handwritten Text Recognition (HTR) models are typically evaluated only on Character/Word Error Rate, leaving their internal attention mechanisms unexamined and their outputs unusable for downstream interpretability tasks such as writer identification. Vision Transformer HTR models exhibit "attention sink" pathology — a small number of patch tokens absorb disproportionate, semantically meaningless attention, degrading the quality of per-character attention maps. This project adapts the register-token mechanism (Darcet et al., 2023, originally proposed for classification ViTs) into a from-scratch, CTC-based sequence-labelling HTR architecture (ViT-RGTS v2: CNN stem + Transformer + register tokens + dual BiLSTM/CNN CTC head), building on the CNN-RNN-CTC baseline codebase (HTR-best-practices, Retsinas et al.) already used in the lab. The core research question is whether adding 0/2/4/8/16 register tokens improves attention interpretability (and does or does not harm CER/WER) relative to a no-register baseline, a traditional CNN-RNN baseline, and pretrained baselines (TorchVision ViT-B/16, TrOCR). The project also validates these findings across two datasets (IAM, READ2016) and investigates supporting factors (SWA, fine-tuning strategy, synthetic pretraining, architectural ablations) needed to make the from-scratch ViT competitive at IAM's small-data scale (6,482 training lines).

*(Confirm this matches your intent before the report-writing thread proceeds — this is the Step-0 gate per your own instructions.)*

### 2.2 Paper → Report-Section Map

Source: `archive/papers/` (9 PDFs, numbered 0–8) and `archive/problem statement/` (2 PDFs + 2 txt files — the PDFs are duplicates of papers #3 and #8, the txts are the problem statement itself).

| # | Paper | Core claim / method | Dataset used in paper | Report section(s) it supports | What is borrowed / contrasted |
|---|-------|---------------------|------------------------|-------------------------------|-------------------------------|
| 4 | **Vision Transformers Need Registers** (Darcet et al.) | ViTs develop high-norm "sink" patches that absorb global attention at the cost of local feature quality; adding learnable register tokens removes the sinks and improves attention map interpretability without hurting (sometimes improving) downstream accuracy | ImageNet classification | §1.1 Motivation, §2.2 Related Work, §3.3 Architecture (register tokens), §5.3 Attention Analysis, §6 Discussion (register "sweet spot" / neutrality-on-CER finding) | **Directly borrowed**: the register-token mechanism itself. **Contrasted**: this project applies it to a sequence-labelling CTC task instead of classification, and confirms attention-only (not accuracy) benefit — same qualitative conclusion, new task domain |
| 6 | **TrOCR: Transformer-based OCR** | Encoder-decoder Transformer pretrained for OCR via a ViT image encoder + text decoder, fine-tuned end-to-end | Large-scale synthetic + real OCR corpora | §2.1/2.2 Related Work, §3.5 Baselines, §5.1/5.2 Results (TrOCR baseline + fine-tuning ablations) | **Borrowed**: `trocr-base-handwritten` as a pretrained baseline. **Contrasted**: TrOCR is encoder-decoder, this project bolts a CTC head onto its encoder only — result (frozen encoder fails at ~72% CER, fine-tuned reaches ~12%) is a novel finding specific to this repurposing |
| 5 | **Best Practices for a Handwritten Text Recognition** (Retsinas et al., presumably HTR-best-practices repo paper) | CNN-RNN-CTC design choices, augmentation strategy, dual CTC head (CNN shortcut + RNN) | IAM | §3.3 Architecture (CNN-RNN baseline, dual-head design), §3.6 Implementation, §4.3 Ablation design, §5.2 Ablations (head-type ablation) | **Directly borrowed**: the CNN-RNN baseline architecture and the "both" (dual CNN+RNN) CTC head design that this project's ViT-RGTS v2 also adopts |
| 1 | **Handwritten Text Recognition: A Survey** | Landscape of CNN-RNN, attention, and Transformer HTR approaches | Survey (multiple) | §2.1 Related Work (context/landscape paragraph) | Context only — no specific numeric claim borrowed |
| 0 | **Advancements and Challenges in Handwritten Text** | Contemporary HTR bottlenecks/frontiers | Survey | §1.3 Research Gap, §7 Future Work | Motivates the interpretability gap framing in §1.3 |
| 8 | **HTR-VT: HTR with Vision Transformer** | ResNet stem + ViT + span-masking for HTR; achieves strong CER on READ2016 | READ2016, IAM | §2.2 Related Work, §3.3 (contrast architecture), mention in §6 Discussion as a design alternative | **Contrasted, not used as final baseline**: an `htrvt` architecture variant exists in `models.py`/`configs/htrvt_read2016_r4.yaml` (run_142 from a prior iteration, CER 4.82%/WER 20.58% on READ2016) but is **not part of the final 63-run experiment matrix** — mention only as related/exploratory work, do not present as a core final result unless you deliberately fold it in |
| 2 | **Interpretable Writer Recognition via VLAC** | Vector of Locally Aggregated Characters for writer-ID from character-level features | IAM (writer-ID task) | §1.1 Motivation (why attention interpretability matters — downstream writer-ID use case), §7 Future Work | Motivates *why* attention-map quality matters (not directly evaluated in this project's final experiment matrix — no writer-ID numbers were produced in the 63 runs) |
| 3 / 7 | **Beyond Memorization (style mixing)** / **WordStylist** | Training-free style mixing via writer-embedding injection into a pretrained diffusion model; cross-attention localizes characters | IAM | §2.2 Related Work (brief), NOT §5 Results — this is a separate reproduction effort (`external/Beyond-Memorization/`, `configs/beyond_memorization.yaml`) outside the 63-run register-sweep matrix | Only cite if you choose to include the Beyond-Memorization reproduction as a secondary contribution; the visualization *style* (blob-thresholded per-character heatmaps, `scripts/postprocessing/beyond_memorization_viz.py`) was borrowed for **this project's own attention figures**, which is a legitimate methodological citation even without discussing the diffusion-model reproduction itself |

**Rule enforced**: every claim in the report must trace to a row in this table (or to the KEY_FINDINGS document, which is itself derived only from measured files). If a claim isn't here, don't include it without adding a row first.

### 2.3 Title Options

1. **"Register Tokens for Interpretable Handwritten Text Recognition: A CTC-Compatible Vision Transformer Study on IAM and READ2016"**
   *Justification: names both the mechanism (register tokens) and the specific technical contribution (making them work in a CTC/sequence-labelling context), plus both datasets used for generalization.*

2. **"Attention Without Accuracy Cost: Register Tokens in From-Scratch Vision Transformers for Handwriting Recognition"**
   *Justification: leads with the actual finding (registers don't hurt CER, improve attention) rather than a generic architecture description — matches the "state the claim, then prove it" instruction style for the whole report.*

3. **"Beyond CER: Evaluating Register-Token Attention Quality in a CNN-Stem Vision Transformer for HTR"**
   *Justification: signals the paper's methodological stance (CER alone is insufficient; attention-quality metrics are the real contribution) and highlights the CNN-stem design as a named architectural component, which is the paper's second-most defensible finding (Section 4.2 of KEY_FINDINGS — the two independent catastrophic-failure ablations).*

*(Pick one, or request revisions, in the report-writing thread — do not proceed past Step 0 there without confirming.)*

### 2.4 Available Result Artifacts (file → report section)

| Artifact | Path | Report section | Status |
|----------|------|-----------------|--------|
| Unified results table (CSV) | `outputs/tables/unified_results.csv` | §5.1 Main Quantitative Results | ✅ Ready |
| Unified results table (LaTeX) | `outputs/tables/unified_results.tex` | §5.1 (drop-in table) | ✅ Ready |
| Register-sweep LaTeX table | `outputs/tables/register_sweep.tex` | §5.1 / §5.3 | ✅ Ready |
| Ablation LaTeX table | `outputs/tables/ablations.tex` | §5.2 Ablation Results | ✅ Ready |
| Architecture comparison LaTeX table | `outputs/tables/architecture_comparison.tex` | §5.1 | ✅ Ready |
| Human-readable labeled CSV | `outputs/tables/final_results_report.csv` | §5.1/§5.2 (cross-check numbers) | ✅ Ready |
| Full corrected ablation table + mechanistic explanations | `documents/final/KEY_FINDINGS_AND_OBSERVATIONS.md` §4 | §5.2, §6 Discussion | ✅ Ready — **use this over any older ablation table**, it fixes 5 mislabeled runs |
| True SWA metrics (`[SWA] Test CER`) | `documents/final/KEY_FINDINGS_AND_OBSERVATIONS.md` §3 | §5.1, §6 Discussion (SWA asymmetry) | ✅ Ready — **do not use `results.csv` final-epoch row for SWA runs**, it's the wrong (live, non-averaged) model |
| Training curves (6 groups: baseline, register sweep, pretrained, ablations, READ2016, synthetic, SWA-ablation) | `outputs/report_figures/training_curves*/` | §5.1, §5.4 Qualitative Results | ✅ Ready |
| Register comparison attention figure | `outputs/pub_figures/fig2_register_comparison*` | §5.3 Attention Map Analysis | ✅ Regenerated 2026-09-07 for runs 144/146/151/152 |
| GradCAM quantitative figure | `outputs/pub_figures/fig3_register_quantitative*` | §5.3 | ✅ Regenerated 2026-09-07 for runs 144/146/151/152 |
| Beyond-Memorization per-character maps | `outputs/report_figures/attention_bm_single/`, `attention_bm_comparison/` | §5.3, §5.4 | ⏳ Pending (job 1804972) |
| Attention quality metrics (entropy/Gini/sharpness) | `outputs/report_figures/attention_quality/` | §5.3 (quantitative interpretability claim) | ✅ Ready — 100 IAM test images × 4 register configs |
| GradCAM per-character figure | `outputs/pub_figures/fig3_gradcam*` (if produced) | §5.3 | ⏳ Pending (job 1804972 — cuDNN eval-mode backward bug fixed, resubmitted) |
| Head specialization / layer-wise evolution figures | `outputs/report_figures/head_specialization/`, `layerwise_evolution/` | §5.3, §5.4 | ⏳ Pending (job 1804972) |
| Per-sample evaluation summaries (CER/WER/exact-match) | `saved_models/experiments/run_N/evaluation/evaluation_summary_test.json` | §5.1 (exact-match rate), §5.4 (worst/best samples) | ⏳ Pending (job 1804973) — 23/63 already done, 40 retrying with config-round-trip fix |
| Per-sample bootstrap significance tests | `outputs/tables/statistical_tests.csv` (needs regeneration) | §5.1 (statistical rigor caveat), §6 Discussion | ⚠️ Stale, must regenerate after job 1804973 |
| Reproducibility (seed variance) numbers | `KEY_FINDINGS_AND_OBSERVATIONS.md` §6 | §4.2 Experimental Design (justify noise floor), §5.1 | ✅ Ready |

---

## 3. Section-by-Section Quick Reference (for the /writer pass in the other thread)

Use this as the fast lookup when drafting each section — it points to the exact source, not just the section name.

- **§1 Introduction** → problem statement summary above (§2.1) + `archive/problem statement/Project problem statement and information.txt` + `archive/problem statement/Instruction set.txt` (architecture requirements, e.g. "use ViT-RGTS from kyegomez/Vit-RGTS", register sweep 0/2/4/8/16).
- **§2 Related Work** → paper map (§2.2 above), cite papers #4, #6, #5, #1, #0, #8, #2 as scoped in the table; do not discuss #3/#7 (Beyond-Memorization/WordStylist) unless including that reproduction as secondary content.
- **§3 Methodology** → `documents/final/FINAL_EXPERIMENT_EXECUTION_GUIDE.md` §2–4 (architecture spec, dataset stats, hyperparameters); repo memory `/memories/repo/htr-pipeline.md` has the exact CTC head types, dataset preprocessing pipeline, and augmentation policy if you need more implementation detail than the guide covers.
- **§4 Evaluation** → `documents/final/FINAL_EXPERIMENT_EXECUTION_GUIDE.md` §4 (experiment matrix table: 20 IAM + 16 synthetic + 22 ablations + 5 READ2016) for §4.2/§4.3; CER/WER definitions are in repo memory (`utils/metrics.py` — edit-distance based, `wer_mode: tokenizer`).
- **§5.1 Main Quantitative Results** → `outputs/tables/unified_results.tex` + `register_sweep.tex` + `architecture_comparison.tex`; narrative numbers in `KEY_FINDINGS_AND_OBSERVATIONS.md` §1.1, §2.1, §6.
- **§5.2 Ablation Results** → `KEY_FINDINGS_AND_OBSERVATIONS.md` §4.1 (**use this corrected table**, not `outputs/tables/ablations.tex` alone — cross-check run labels against §0.1 of that document first).
- **§5.3 Attention Map Analysis** → pending job 1803919 outputs; mechanistic reasoning already written in `KEY_FINDINGS_AND_OBSERVATIONS.md` §1.2 (why registers don't move CER — the interpretability argument goes here once figures land).
- **§5.4 Qualitative Results** → training curves (ready) + BM-style attention maps (pending) + worst/best sample reports (pending, job 1803920).
- **§6 Discussion** → `KEY_FINDINGS_AND_OBSERVATIONS.md` in full — it is written specifically as mechanistic discussion material (why CNN-RNN wins, why naive ViT-B/16 fails, why the two CNN-stem-removal ablations converge, SWA asymmetry, dataset generalization, reproducibility noise floor). Section 10 of that document ("Recommendations for the Final Report") is literally a discussion-section outline.
- **§7 Conclusion & Future Work** → `KEY_FINDINGS_AND_OBSERVATIONS.md` §9 ("What Remains Pending") + §7.4 (synthetic pretraining is disclosed as time-truncated, a natural future-work item) + writer-ID (paper #2) as an unrealized downstream application.
- **Appendix** → full hyperparameter tables already in `FINAL_EXPERIMENT_EXECUTION_GUIDE.md` Appendix A/B; SLURM script inventory in `experiments_execution/slurm_modular/README.md` if reproducibility appendix is required.

---

## 4. Known Caveats to Carry Into the Report-Writing Thread

1. **Never cite a register-count CER difference as a finding without noting it's within the ±0.08–0.10pp seed-noise floor** (§6 of KEY_FINDINGS). The honest claim is "CER-neutral," not "CER-improving."
2. **Never cite `results.csv`'s final-epoch row for an SWA-enabled run.** Use the `[SWA] Test CER` value from `training.log`, documented per-run in KEY_FINDINGS §3.1.
3. **Never reuse run→ablation labels from any document dated before 2026-09-06** without cross-checking against KEY_FINDINGS §0.1 (5 runs were mislabeled in earlier drafts: 163, 164, 165, 175, 179).
4. **Synthetic pretraining numbers (§7 of KEY_FINDINGS) are walltime-truncated** (2–11 of 15 target epochs) — report as a lower-bound/domain-gap demonstration, not a converged result.
5. **HTR-VT (`htrvt` arch, run_142) is NOT part of the final 63-run matrix** — it's prior exploratory work on READ2016 only. Mention as related work (paper #8), not as a final baseline, unless you deliberately decide to fold it in and rerun it under the same 150-epoch standard.
6. **Before writing §5.3, re-check job 1804972/1804973 status** (`squeue -u $USER`) and the freshness of `outputs/pub_figures/` and `outputs/report_figures/attention_*` timestamps — the old May-16-dated placeholder figures for Fig 2/3 have already been overwritten (2026-09-07) with real runs 144/146/151/152 data, but GradCAM, Beyond-Memorization maps, head-specialization, and layer-wise-evolution figures were still pending at last check.
7. **Two more pipeline bugs were found and fixed on 2026-09-07** (after KEY_FINDINGS_AND_OBSERVATIONS.md was written): (a) `batch_eval.slurm` was reconstructing each run's architecture from a *generic default config per arch type* instead of that run's own exact saved `config.json`, causing 40/63 evaluations to fail with `Missing key(s) in state_dict` for every non-default architecture (ablations, pretrained fine-tune variants); fixed by converting each run's `config.json` directly to a temporary YAML. (b) `pub_fig3_gradcam_quantitative.py`'s backward pass crashed with `cudnn RNN backward can only be called in training mode` because the model is in `.eval()` mode; fixed by wrapping the backward call in `torch.backends.cudnn.flags(enabled=False)`. Both fixes are already applied and jobs resubmitted — no action needed unless `squeue` shows further errors.
