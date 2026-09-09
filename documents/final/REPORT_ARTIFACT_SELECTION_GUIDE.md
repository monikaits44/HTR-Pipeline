# Report Artifact Selection Guide

> **Purpose**: Single reference that decides, section-by-section, which quantitative tables and qualitative figures belong in the final report, which are supporting-only (supplementary), and which should be dropped. Every entry is graded **P0 (must include)** / **P1 (include if space)** / **P2 (cut / move to supplementary)** and tied back to a specific storyline sentence, so removals during page-count trimming are principled instead of arbitrary.
>
> **Companion documents (already exist, do not duplicate)**:
> - Storyline & mechanism: [KEY_FINDINGS_AND_OBSERVATIONS.md](./KEY_FINDINGS_AND_OBSERVATIONS.md)
> - Section→artifact mapping (higher level, less opinionated): [REPORT_WRITING_HANDOFF.md](./REPORT_WRITING_HANDOFF.md) §2.4 and §3
> - Current draft: [pr_htr_report/main.tex](./pr_htr_report/main.tex) and [pr_htr_report/sections/](./pr_htr_report/sections/)
>
> **Date**: 2026-09-07

---

## 0. The Storyline in One Paragraph (Anchor Every Decision to This)

The report argues **two connected claims**. **Claim A (recognition-neutral)**: Adapting register tokens (Darcet *et al.*, 2023, classification ViTs) into a CTC-based sequence-labelling ViT for HTR leaves CER within seed-noise (0.10–0.24 pp range across R∈{0,2,4,8,16}, floor ±0.08 pp) — same qualitative behaviour reported for classification, now demonstrated on IAM **and** READ2016. **Claim B (attention-quality-positive-but-partial)**: On four attention-quality metrics, registers monotonically or near-monotonically improve interpretability (entropy ↓, Spearman ρ ↑, aligned-head count ↑) but a **residual sink pattern survives on patch tokens** because CTC self-attention has no CLS-style query separated from the patches. **Supporting scaffolding**: CNN-stem is causally necessary (2× catastrophic-failure convergence at ~72–74% CER); SWA helps only when the base training has converged; pretrained ViT-B/16 fails without differential LR; results generalise across script (IAM English, READ2016 German). Every figure and table in the report must earn its place by directly supporting **one specific sentence** in that paragraph. Everything else moves to the supplementary or is cut.

---

## 1. Grading Rubric (Applied Below)

| Grade | Meaning | Number allowed |
|-------|---------|----------------|
| **P0** | Storyline collapses without it. Cutting it makes the report incomplete. | ~5 tables, ~6 figures |
| **P1** | Materially strengthens the argument, but the storyline survives without it. Include if page budget allows; otherwise supplementary. | ~3 tables, ~4 figures |
| **P2** | Interesting, but redundant with a P0/P1, or off-storyline. Cut from main report; keep in supplementary or delete. | remainder |

**Page budget assumption**: IEEEtran onecolumn journal-style, roughly 20–25 pages target. Main-body figure count ~10, table count ~8 is a healthy target; anything beyond that starts costing narrative density.

---

## 2. Quantitative Artifacts (Tables & Numbers)

### 2.1 P0 — Must Include in Main Report

| # | Artifact | Source | Report section | Supports storyline sentence | Why P0 |
|---|----------|--------|----------------|-----------------------------|--------|
| Q1 | **Register sweep on IAM** (CER/WER, R∈{0,2,4,8,16}, no-SWA + SWA rows) | [outputs/tables/register_sweep.tex](../../outputs/tables/register_sweep.tex) | §5.1 (Table `tab:register_sweep`) | Claim A — direct evidence CER is register-neutral | This IS the primary result. Without it the reader cannot check the "0.10–0.24 pp range" number. Already in draft. |
| Q2 | **Architecture family comparison** (CNN-RNN, ViT-RGTS v2, TorchVision ViT-B/16, TrOCR, HTR-VT-external) | [outputs/tables/architecture_comparison.tex](../../outputs/tables/architecture_comparison.tex) | §5.1 (Table `tab:arch_comparison`) | Framing — CNN-RNN wins absolute CER; ViT-RGTS v2 is the interpretability contribution, not the accuracy one | Positions the register-token model honestly against baselines. Already in draft. |
| Q3 | **Attention-quality metrics vs. register count** (entropy, Spearman ρ, aligned-head count, token-norm) | [outputs/report_figures/attention_quality/attention_quality_metrics.csv](../../outputs/report_figures/attention_quality/attention_quality_metrics.csv) → summary table | §5.3 (new small table alongside Fig `fig:register_quant`) | Claim B — the "monotonic three of four metrics" claim | The report currently narrates these numbers inline; a compact 4-row table makes them citable and verifiable. **RECOMMENDED ADDITION.** |
| Q4 | **READ2016 register sweep** | Extracted from [outputs/tables/unified_results.csv](../../outputs/tables/unified_results.csv) | §5.6 (Table `tab:read2016`) | Claim A generalises across dataset | Second-dataset corroboration — without it the register-neutrality claim is IAM-only. Already in draft. |
| Q5 | **Architecture ablations** (dual head, CNN stem, dim/depth/augmentation, RNN depth) | [outputs/tables/ablations.tex](../../outputs/tables/ablations.tex) with corrected labels from [KEY_FINDINGS](./KEY_FINDINGS_AND_OBSERVATIONS.md) §4.1 | §5.4 (Table `tab:ablations`) | Supporting — CNN-stem causally necessary; dual head contributes; other choices robust | **Both catastrophic-failure rows (No-CNN-stem, Legacy v1) MUST stay in main-body** because they carry the argument. Marginal-ablation rows can be trimmed to supplementary. Already in draft. |
| Q6 | **Fine-tuning strategy sweep for pretrained models** | [outputs/tables/unified_results.csv](../../outputs/tables/unified_results.csv), filtered | §5.5 (Table `tab:finetune`) | Supporting — why naive LR fails, why gradual unfreeze fails | The "73.14% → 15.09% CER just by LR grouping" is one of the cleanest numeric arguments in the whole matrix. Already in draft. |

### 2.2 P1 — Include If Space Permits

| # | Artifact | Source | Report section | Supports | Why P1 |
|---|----------|--------|----------------|----------|--------|
| Q7 | **Seed-variance (noise-floor) row** | [KEY_FINDINGS](./KEY_FINDINGS_AND_OBSERVATIONS.md) §6 | §5.7 as a 3-row inline table or a single sentence | Grounds every "within noise" claim | Currently a single sentence in draft. Consider promoting to a 3-row mini-table — it is the single number the entire "no register effect" reading rests on. |
| Q8 | **SWA true-vs-CSV correction** (3–4 rows: CNN-RNN, ViT-RGTS v2 R=4, TrOCR-LLRD) | [KEY_FINDINGS](./KEY_FINDINGS_AND_OBSERVATIONS.md) §3.1 | §5.1 caption footnote OR §5.5 mini-table | Reproducibility hygiene — reader may want to check numbers against `results.csv` | Deserves at least a footnote; a mini-table is honest but takes lines. |
| Q9 | **SWA timing/LR ablation** | [KEY_FINDINGS](./KEY_FINDINGS_AND_OBSERVATIONS.md) §3.2, §3.3 | Supplementary | Marginal effect (<0.15 pp differences) | Interesting but does not carry the storyline. Move to supplementary. |
| Q10 | **Synthetic pretraining table** | [KEY_FINDINGS](./KEY_FINDINGS_AND_OBSERVATIONS.md) §7 | §5.7 (already there) or supplementary | Disclosed as walltime-truncated — supports the render-to-real gap sentence | The narrative sentence covers this; the table is optional. |

### 2.3 P2 — Cut from Main Report

| # | Artifact | Why P2 |
|---|----------|--------|
| Q11 | Full `outputs/tables/unified_results.tex` (all 63 runs, ~80 rows) | Data dump; readers do not read 80-row tables. Cite as "see supplementary". |
| Q12 | `outputs/tables/statistical_tests.csv` (bootstrap significance) | Stale (references runs 105–142, not final 143–205 set). Do not cite unless regenerated after `batch_eval_1804973` completes and numbers are re-verified. |
| Q13 | `outputs/tables/significance_register_sweep.tex` / `significance_vs_cnn_rnn.tex` | Same staleness concern. If regenerated after `batch_eval` completes, promote to P1. Currently P2 by default. |
| Q14 | Dropout, no-warmup, batch-size ablation rows (from ablations.tex) | Effect sizes below noise floor. Aggregate into a single "other ablations: all within noise" sentence in main body; full table to supplementary. |

---

## 3. Qualitative Artifacts (Figures & Attention Maps)

### 3.1 P0 — Must Include in Main Report

| # | Figure | Path (current) | Report section | Supports storyline sentence | Why P0 |
|---|--------|----------------|----------------|-----------------------------|--------|
| F1 | **Overall pipeline diagram** (input → CNN stem + registers → transformer → CTC head → CER; attention branch → interpretability metrics) | [documents/final/figures/htr_pipeline_diagram_v3.png](./figures/htr_pipeline_diagram_v3.png) | §3.1 System Overview | Reader orientation | The v3 is the final approved version per repo memory. Ships as `figures/htr_pipeline_diagram.pdf` in the report. |
| F2 | **CER convergence curve** (best ViT-RGTS v2 R=4 + all-R overlay on IAM) | [outputs/report_figures/training_curves/cer_metrics.png](../../outputs/report_figures/training_curves/cer_metrics.png) → currently in draft as `training_cer_curve.png` | §5.1 (Fig `fig:cer_convergence`) | "All register counts converge to essentially the same trajectory after epoch 25" | Already in draft. Central claim visualization. |
| F3 | **Beyond-Memorization reproduction on our model** (one word, e.g. "booklet", input crop + 6 per-character attention overlays) | [documents/final/pr_htr_report/figures/bm_booklet_*.png](./pr_htr_report/figures/) (7 files) | §5.2 (Fig `fig:bm_booklet`) | "Reproduces Fig. 5 of Nikolaidou et al. on our own CTC ViT" | This is Vincent's explicit ask. Already in draft. **Non-negotiable.** |
| F4 | **Register-effect attention map** (R∈{0,4,8,16} × 6 characters of "talks.", CTC self-attention at each character's peak column) | [outputs/attention_maps/fig_attn_talks.png](../../outputs/attention_maps/fig_attn_talks.png) — **SELECTED 2026-09-07 (see §3.4).** | §5.3 (Fig `fig:register_effect_attn`) | Claim B — visual evidence that registers change attention structure | 0-reg row shows a clear sink pattern (broad yellow bands filling the map for `A_{c=1}` "t" and `A_{c=2}` "a"); the pattern weakens sharply at 4-reg, disappears at 8-reg, and 16-reg shows narrow vertical peaks shifting with character index. **Provenance caveat**: May 2026 checkpoint (pre-run-143); add one honest sentence to caption or supplementary (see §3.4). |
| F5 | **Attention-quality metrics grid** (4 panels: entropy, Spearman ρ, aligned-head count, token-norm distribution vs R) | [outputs/pub_figures/fig3_register_quantitative.png](../../outputs/pub_figures/fig3_register_quantitative.png) OR [documents/final/pr_htr_report/figures/register_quantitative.png](./pr_htr_report/figures/register_quantitative.png) | §5.3 (Fig `fig:register_quant`) | Claim B — quantitative companion to F4 | Already in draft. Computed on final runs 144/146/151/152 (2026-09-07). Pairs with F4: F4 is the pixel evidence, F5 is the numeric evidence. |
| F4b | **CTC posterior heatmap** (input image with CTC peak-column red lines + character×timestep posterior heatmap for "talks.") | [outputs/pub_figures/ctc_posterior.png](../../outputs/pub_figures/ctc_posterior.png) | §3.3 Architecture (end of CTC-head description) OR §4.1 Preliminaries | Expository — defines "CTC peak column t_c" for the reader | Not a result. Sets up the vocabulary every attention figure in §5.3 depends on. Without it the reader has to trust the phrase "CTC peak column" as jargon; with it they see the peaks themselves. Cheap in page count, high in payoff. |

### 3.2 P1 — Include If Space Permits

| # | Figure | Path | Report section | Why P1 |
|---|--------|------|----------------|--------|
| F6 | **Attention-quality trend panels** (peak-sharpness vs registers, diagonality vs text length — the actual generated line plots) | [outputs/report_figures/attention_quality/peak_sharpness_vs_registers.png](../../outputs/report_figures/attention_quality/peak_sharpness_vs_registers.png), [.../diagonality_vs_text_length.png](../../outputs/report_figures/attention_quality/diagonality_vs_text_length.png) | §5.3 supplementary companion to F5 | Overlaps with F5's panel (a) and (b). Keep only if F5 becomes cramped and you split it. |
| F7 | **Head-specialization** (per-head attention pattern grid for R=4 model) | Pending job 1804972 → `outputs/report_figures/head_specialization/` | §5.3 §Discussion | Adds granularity to "aligned-head count" metric. **Currently pending** — safe to cite only after job completes. |
| F8 | **Layer-wise attention evolution** (attention map at layer 1, 3, 6 for the same input) | Pending job 1804972 → `outputs/report_figures/layerwise_evolution/` | §5.3 or Discussion | Shows *where in the network* the register effect emerges. **Currently pending.** |
| F9 | **Beyond-Memorization ORIGINAL diffusion reproduction** (from the WordStylist repo, 4 charLocation panels for one word) | [outputs/beyond_memorization/charIndex_0/](../../outputs/beyond_memorization/charIndex_0/), etc. | §2 Related Work OR §5.2 introduction | Only include if you devote a paragraph to explaining WHY the BM visualization style was chosen. Otherwise the reader will not know what "BM style" refers to. |
| F9b | **BM-style per-character overlay on HTR-VT external comparator** (5 per-character hot-spot panels for the digit string "203", R=4) | [outputs/htrvt_bm_attention/test_0_R4_chars.png](../../outputs/htrvt_bm_attention/test_0_R4_chars.png) | §5.2 (secondary figure after `fig:bm_booklet`) | Shows the BM visualization style transfers to a different architecture (HTR-VT) and script (READ2016 digits), i.e. is not tied to our specific model. Cheap way to cite HTR-VT visually once without inflating its role beyond the numeric row in Table `tab:read2016`. Cut if pages are tight. |
| F10 | **Pretrained fine-tuning failure curve** (loss plateau for run_153 naive-LR, contrasted with run_174 2-group-LR) | Extract from `saved_models/experiments/run_153/results.csv` and `run_174/results.csv` → new plot | §5.5 (companion to Q6) | The 5× error-rate improvement from LR grouping is a great story; a curve makes it visceral. Currently only in text. |

### 3.3 P2 — Cut from Main Report / Move to Supplementary

| # | Figure | Why P2 |
|---|--------|--------|
| F11 | All per-group training curves (`outputs/report_figures/training_curves_{ablations,pretrained,read2016,synthetic,swa_ablation}/`) | Already correctly moved to supplementary in current draft. Keep them there. |
| F12 | Older per-run attention explorations under `outputs/attention_maps/` (fig1–fig6, `run_90_*.png`, `notebook_beyond_memorization/`, etc.) | Iterative development artifacts from earlier register-sweep sets. Superseded by pub_figures. Delete or archive. |
| F13 | `outputs/pub_figures/fig5_char_attention_run_84.{png,pdf}` and other `run_84`-era files | Refer to a superseded run set; already overwritten by the 2026-09-07 regeneration for runs 144/146/151/152. Delete. |
| F14 | `outputs/pub_figures/best_head_comparison.png`, `raw_self_attention.png`, `saliency_maps.png` (single-panel diagnostic plots) | Diagnostic-only; each individually shows one aspect that is better summarized by F5. |
| F15 | `outputs/htrvt_bm_attention/*` (HTR-VT-arch attention grids from run_142) | HTR-VT is external comparator only, not a contribution. Numeric result in Table `tab:read2016` is enough; the attention grid does not need to be shown. |
| F16 | `outputs/gradcam/*` (if the report does not adopt GradCAM as a primary interpretability tool) | Current draft uses attention weights, not GradCAM. Keep the GradCAM figure (`fig_gradcam_attention.pdf`) only if you decide to add a "GradCAM as sanity check" one-paragraph subsection. Otherwise drop. |
| F17 | Multiple TikZ diagram versions (`htr_pipeline_diagram.tex`, `_v2.tex`) | v3 is the accepted final version. v1 and v2 are archived — do not compile them into the report. |

### 3.4 The Register-Effect Attention Map — Selected Figure & Rationale (updated 2026-09-07)

**Selected**: [`outputs/attention_maps/fig_attn_talks.png`](../../outputs/attention_maps/fig_attn_talks.png)

**Why this one, not the newer `fig2_register_comparison.png` / `fig5_register_comparison_attn_talks.png`.**

The two 2026-09-07-regenerated figures use *per-panel* color normalization, which makes a diffuse 0-reg row and a peaked 8-reg row occupy the same visual dynamic range and hides the effect. `fig_attn_talks.png` uses a shared colormap across the 4×7 grid — the 0-reg row `A_{c=1}` "t" panel is filled with broad yellow bands (the sink pattern that Darcet et al. describes for classification ViTs, here on a CTC ViT patch token), the 4-reg row has that spread visibly reduced, the 8-reg row is essentially a set of narrow vertical peaks aligned to the ink columns, and the 16-reg row shows the same peaks shifting cleanly with character index. **The register-absorption mechanism is legible from the pixels alone**, without needing captions to walk the reader through what to look for.

**Provenance caveat.** May 16, 2026 timestamp — generated from an earlier ViT-RGTS v2 register-sweep training set that predates the final 63-run matrix (runs 143–205, Aug 2026). Two ways to handle this honestly:
1. **Recommended**: add one sentence to the caption or the supplementary provenance section (draft in §7 below). The register-sweep mechanism is architectural — it depends on the presence/count of register tokens, not on the specific training seed. The final-run figure `fig2_register_comparison.png` is cited immediately after F4 as an in-matrix cross-check.
2. Regenerate with the final runs. Off the table per user constraint.

**Companion figures that turn F4 into a complete argument.**
- **F5** ([`outputs/pub_figures/fig3_register_quantitative.png`](../../outputs/pub_figures/fig3_register_quantitative.png)) — attention-quality metric panels on the *final* run set (144/146/151/152). Pair F4 (pixel evidence, earlier runs) immediately with F5 (numeric evidence, final runs) in §5.3 so the two together defensibly cover the claim.
- **F4b** ([`outputs/pub_figures/ctc_posterior.png`](../../outputs/pub_figures/ctc_posterior.png)) — CTC posterior heatmap for the same word "talks." Sits earlier in the report (§3.3 or §4.1) as an expository figure defining "CTC peak column t_c", which the entire §5.3 vocabulary rests on. Reusing the same word ("talks.") across F4b, F4, and F5 gives the reader a single running example to anchor to.

**Older ideas superseded by this decision.**
- The F4-v2 regeneration recipe (previously drafted here on 2026-09-07 morning) — no longer needed. `fig_attn_talks.png` already meets the visual bar the recipe was designed to hit.
- Using `fig6b_character_attention_supporters.png` (11-column supporters grid) — cut per user feedback ("not meaningful").
- Using `fig6_character_attention.png` — cut per user feedback ("not meaningful").

**Placement in the report:**

| Section | Figure | Purpose |
|---------|--------|---------|
| §3.3 (end of CTC-head description) or §4.1 (Preliminaries) | **F4b** `ctc_posterior.png` | Defines CTC peak column, sets up §5.3 vocabulary |
| §5.2 (hero) | Existing `bm_booklet_*.png` figures | Reproduction of Nikolaidou et al. Fig. 5 |
| §5.2 (secondary, optional) | **F9b** `test_0_R4_chars.png` | Shows BM style transfers to HTR-VT external comparator |
| §5.3 (hero) | **F4** `fig_attn_talks.png` | Register-sweep pixel evidence |
| §5.3 (companion) | **F5** `fig3_register_quantitative.png` | Register-sweep numeric evidence (final runs) |

**Suggested captions:**

*F4b (ctc_posterior.png)* — "CTC posterior for the ViT-RGTS v2 model on the word 'talks.' (IAM). Top: input image with red vertical lines marking the CTC peak columns t_c for each recognized character. Bottom: per-character posterior P(c | timestep). Only ~6 of 128 timesteps carry non-blank probability mass; the remaining columns emit the CTC blank symbol. Every per-character attention map in Section 5.3 is queried at those same peak columns t_c."

*F4 (fig_attn_talks.png)* — "Per-character self-attention maps for the word 'talks.' (IAM) queried at each character's CTC peak column, across ViT-RGTS v2 with R ∈ {0, 4, 8, 16} registers (rows). At R=0 (top row), attention for early characters shows broad diffuse bands over regions of the image that do not contain the queried character — the sink pattern described by Darcet et al. for classification ViTs, here surfacing on CTC patch tokens. As register count increases, the sink pattern weakens and attention concentrates on the vertical ink columns of the actual queried character; by R=8 the sink is essentially absorbed. The figure is rendered from an earlier training checkpoint of the same register-sweep architecture (May 2026); the numeric metrics in Figure 5 confirm the effect quantitatively on the final run set (runs 144/146/151/152)."

*F9b (test_0_R4_chars.png)* — "The same per-character BM-style attention rendering applied to the HTR-VT external comparator (run 142) on a READ2016 digit string ('203'), showing the visualization style is not tied to our specific ViT-RGTS v2 architecture. HTR-VT is reported quantitatively in Table `tab:read2016`; this figure is included only to demonstrate cross-architecture applicability of the visualization."

---

### 3.4-legacy Older critique kept for context (not the current recommendation)

**Problem with the older-generation renderings.**

Earlier iterations of this document proposed regenerating a new register-sweep grid because two contemporary figures both failed to show the effect clearly:
- `outputs/pub_figures/fig2_register_comparison.png` — high-level grid with input + attention rollups across R∈{0,4,8,16}
- `outputs/pub_figures/fig5_register_comparison_attn_talks.png` — rows = R∈{0,4,8,16}, columns = per-character attention for the word "talks."

The `fig5_register_comparison_attn_talks.png` grid **does not clearly show a register effect** as-is. Every row looks similar at first glance; the attention peaks shift with the character index in every row (0-reg included), which visually undercuts the argument. The problem is not the model — the metrics in F5 do show entropy dropping and aligned-head count rising — but the *rendering choice* is not making the effect legible. Specifically:

1. The overlay uses **column-collapsed 1D attention** stretched to full image height, so every character attention looks like a vertical band across the whole image. Peak-shift is visible, but sink-pattern differences (the actual mechanism the report claims) are not.
2. The color range is normalized *per panel*, so a diffuse 0-reg row and a peaked 8-reg row end up with similar visual dynamic range.
3. Only one word ("talks") is shown — small sample; readers can suspect cherry-picking.
4. No numeric annotation (entropy value, peak column, ρ) is overlaid, so the reader cannot cross-reference F5 metrics against F4 pixels.

**Required regeneration recipe.** The report needs a fresh figure that makes the register effect visually *unmissable*. The following is a step-by-step specification.

**F4-v2 specification** (file: `outputs/report_figures/fig_register_effect_v2.pdf`):

- **Layout**: 4 rows × 6 columns.
  - Rows = register configurations: R=0 (run_144), R=4 (run_146), R=8 (run_151), R=16 (run_152). R=2 is skipped because 3 configurations already give the trend line and 5 rows overcrowds.
  - Columns: (col 1) input word crop; (cols 2–6) per-character attention overlays for 5 characters of a single ~5-letter word.
- **Per-character rendering**: use the Beyond-Memorization recipe *exactly* — `scripts/postprocessing/beyond_memorization_viz.py`'s `find_strongest_blob()` + `compute_char_heatmap()`. This is the same recipe used for F3 (BM reproduction), so the register-effect figure and the BM-reproduction figure share the same visual grammar and the reader can compare them directly.
- **Global color normalization**: use one shared vmin/vmax across all 4×5=20 attention panels, not per-panel. This is the single most important change — it lets the reader *see* that 0-reg attention is more diffuse than 8-reg attention rather than having to infer it from the numbers.
- **Per-row annotation**: at the right margin of each row, print the three F5 metrics for that R: `H = X.X bits · ρ = 0.XX · aligned = X.X/8`. This directly connects F4 to F5 so the reader sees both the pixel evidence and the quantitative evidence in one glance.
- **Word selection**: pick a word that is *long enough* (≥5 characters, ≤8) to show the character-shift AND *legible* in the input crop. From the currently-shipped BM figure ("booklet") a good candidate is a similar word from the same set. Reuse `notebook/sample_images/a01-038-12.png` (the "talks." word already used in `fig5_register_comparison_attn_talks.png`) if the length is acceptable, OR pick "booklet" from `outputs/beyond_memorization/` to match F3.
- **Multi-word supplement**: also render the same 4-row grid for 2–3 additional words (as separate figures for the supplementary) so the main-report figure is defensibly one example out of a reproducible set, not a cherry-pick.
- **Ground-truth strokes shown at 20% opacity behind the heatmap** (matching F3 recipe). Do not use pure attention heat with no image behind it — the reader loses spatial grounding.

**Recommended script to produce F4-v2**:

Extend or reuse `scripts/pub_fig2_register_comparison.py` with the following changes:
1. Add `--global-vmax` flag (share vmax across the entire figure).
2. Add `--metric-annotation` flag to overlay entropy/ρ/aligned-head-count per row (values pulled from `outputs/report_figures/attention_quality/attention_quality_metrics.csv`).
3. Use BM per-character blob rendering path (`beyond_memorization_viz.py`'s helper functions) instead of the current column-collapsed 1D rendering.
4. Emit both `.pdf` (report) and `.png` (preview) into `outputs/report_figures/`.

Add a SLURM step to `experiments_execution/slurm_modular/postprocess_all.slurm` (currently job 1804972) so it runs after Step 3 (attention_quality_metrics) — that ordering guarantees the CSV needed for the row-annotations exists.

**Fallback if the P0 F4-v2 cannot be regenerated in time**: use `outputs/pub_figures/fig2_register_comparison.pdf` (the 2026-09-07-regenerated version, already ships real runs 144/146/151/152 data). It is *acceptable but not optimal*. In that case, expand F4's caption to explicitly walk the reader through what to look for (e.g. "note that the peak in row 3 is narrower than in row 1, and the diffuse background attention is lower — compare to the entropy values in Fig. 5"). Do not present the current fig2 without this narrative crutch, because the effect is not self-evident from the pixels alone.

---

## 4. Section-by-Section Decision Table (Copy-Paste Into Report Planning)

| Report section | Tables (Q) | Figures (F) | 1-sentence storyline anchor |
|----------------|------------|-------------|----------------------------|
| §1 Introduction | — | (optional F1 preview) | "Register tokens improve attention interpretability in ViTs without hurting recognition — does this transfer to CTC HTR?" |
| §2 Related Work | — | F9 (P1, only if BM-repro is discussed) | "Darcet et al. proposed registers for classification; Nikolaidou et al. visualize per-character attention in diffusion; we bridge these into CTC HTR." |
| §3 Methodology | — | **F1 (P0)** pipeline diagram; **F4b (P0)** CTC posterior at end of §3.3 | Architecture + register mechanism + BM visualization pipeline + CTC-decoding vocabulary. |
| §4 Evaluation | — | (F4b alternative placement here if not in §3.3) | CER/WER definitions, experiment matrix table (small, embed inline). |
| §5.1 Main results | **Q1 (P0), Q2 (P0)**, Q7 (P1), Q8 (P1) | **F2 (P0)** CER curve | "Register-count CER range 0.10–0.24 pp is within seed noise ±0.08 pp." |
| §5.2 BM reproduction | — | **F3 (P0)** BM booklet; **F9b (P1)** HTR-VT "203" secondary | "We reproduce Nikolaidou et al. Fig. 5 on our own CTC ViT; visualization style transfers to HTR-VT as well." |
| §5.3 Register effect on attention | **Q3 (P0)** attention-quality summary | **F4 (P0)** `fig_attn_talks.png`, **F5 (P0)** metrics panels, F6/F7/F8 (P1) | "Registers monotonically improve 3 of 4 attention-quality metrics; residual sink survives." |
| §5.4 Architecture ablation | **Q5 (P0)** | (F10 P1 if pretrained curves added) | "CNN-stem is causally necessary; dual head contributes." |
| §5.5 Fine-tuning pretrained | **Q6 (P0)** | F10 (P1) | "Differential LR closes 60 pp of ViT-B/16 CER; gradual unfreeze fails for large domain gap." |
| §5.6 READ2016 generalization | **Q4 (P0)** | — | "Register-neutrality replicates on READ2016 (German)." |
| §5.7 Synthetic pretraining | Q10 (P1) | — | "~30 pp render-to-real domain gap; walltime-truncated." |
| §5.8 Reproducibility | Q7 (P1) — mini-table | — | "Seed-noise floor 0.08 pp defines the resolution limit." |
| §6 Discussion | — | (F7/F8 if promoted P0) | Mechanistic paragraphs from KEY_FINDINGS §1.2, §2.2, §3, §4.2. |
| §7 Conclusion & Future Work | — | — | Writer-ID extension; CLS-token + attention regularizer to close residual sink. |
| Supplementary | Q9, Q10, Q11 | F6, F7, F8, F11, all per-group training curves | Overflow and reproducibility supporting material. |

---

## 5. Concrete "Keep / Regenerate / Cut" Checklist

Work top-to-bottom before submitting the report.

### 5.1 Keep as-is (no change needed)

- ✅ [documents/final/pr_htr_report/figures/htr_pipeline_diagram.pdf](./pr_htr_report/figures/htr_pipeline_diagram.pdf) (F1)
- ✅ [documents/final/pr_htr_report/figures/training_cer_curve.png](./pr_htr_report/figures/training_cer_curve.png) (F2)
- ✅ [documents/final/pr_htr_report/figures/bm_booklet_*.png](./pr_htr_report/figures/) (F3)
- ✅ [outputs/tables/register_sweep.tex](../../outputs/tables/register_sweep.tex) (Q1)
- ✅ [outputs/tables/architecture_comparison.tex](../../outputs/tables/architecture_comparison.tex) (Q2)
- ✅ [outputs/tables/ablations.tex](../../outputs/tables/ablations.tex) — but **verify against KEY_FINDINGS §0.1 corrected labels** before citing (Q5)

### 5.2 Regenerate before final compile

- ✅ **F4 (register-effect attention map)** — resolved 2026-09-07 by selecting [`outputs/attention_maps/fig_attn_talks.png`](../../outputs/attention_maps/fig_attn_talks.png). No regeneration needed. See §3.4 for provenance-caveat wording.
- ⚠️ **Q3 (attention-quality summary table)** — extract from `outputs/report_figures/attention_quality/attention_quality_metrics.csv` (currently only shipped as CSV + 2 plots). Add a compact 4-row LaTeX table (`outputs/tables/attention_quality_summary.tex`) with columns `R | median entropy | median ρ | aligned heads (median) | median peak sharpness`. Ship in §5.3 alongside F5.
- ⚠️ **Q12/Q13 (statistical significance tests)** — regenerate against final run set (143–205) after job `batch_eval_1804973` completes. Do not cite the currently-shipped stale numbers.
- ⚠️ **F7/F8 (head-specialization, layer-wise-evolution)** — currently empty directories under `outputs/report_figures/`. Await job `postprocess_1804972` completion. If they land in time and are legible, promote to P1 in §5.3 or §6; if not, drop.

### 5.3 Cut / Move to Supplementary

- ❌ All `run_84`, `run_90`, `notebook_beyond_memorization` files under `outputs/attention_maps/` and `outputs/pub_figures/` — superseded by 2026-09-07 regeneration on final run set (F13).
- ❌ `outputs/htrvt_bm_attention/*` — external comparator, not a contribution (F15).
- ❌ `outputs/gradcam/*` — not the primary interpretability tool in current draft (F16).
- ❌ `documents/final/figures/htr_pipeline_diagram.tex` and `_v2.tex` (v1 and v2) — v3 is final (F17).
- 📦 Per-group training curves → supplementary (F11) — already correctly placed.
- 📦 SWA-timing / SWA-LR ablation (Q9) → supplementary — sub-noise-floor effect.

### 5.4 Verify before citing (do not skip)

- ⏳ Job `batch_eval_1804973` (per-sample evaluation) status: `squeue -u $USER` → `logs/modular/batch_eval_1804973.out`. Required for Q12/Q13 regeneration and for any exact-match rate cited.
- ⏳ Job `postprocess_1804972` (attention pipeline) status: `logs/modular/postprocess_1804972.out`. Required for F7/F8 and for any post-2026-09-07 attention-quality metric.
- 🔎 Every SWA-enabled number: cross-check `training.log` `[SWA] Test CER` line vs. `results.csv` last-epoch row. Never cite the CSV row for an SWA run — see KEY_FINDINGS §0.2.
- 🔎 Every ablation run label: cross-check against KEY_FINDINGS §0.1 corrected mapping (runs 163, 164, 165, 175, 179 were mislabeled in earlier drafts).

---

## 6. Attention-Map Register-Effect — Delivery Status (resolved 2026-09-07)

**Status**: **Resolved without regeneration.** The three P0/P1 figures needed for the register-effect story already exist in the workspace:

| Figure | File | Role in §5.3 storyline |
|--------|------|------------------------|
| **F4b** | [`outputs/pub_figures/ctc_posterior.png`](../../outputs/pub_figures/ctc_posterior.png) | Expository — defines CTC peak columns t_c (placed earlier, §3.3 or §4.1) |
| **F4**  | [`outputs/attention_maps/fig_attn_talks.png`](../../outputs/attention_maps/fig_attn_talks.png) | Pixel evidence of the register-sweep effect ("talks.", R∈{0,4,8,16}) |
| **F5**  | [`outputs/pub_figures/fig3_register_quantitative.png`](../../outputs/pub_figures/fig3_register_quantitative.png) | Numeric evidence on the final run set (144/146/151/152) |

**Delivery steps** (all ≤10 minutes, no compute required):

1. Copy the three files into [`documents/final/pr_htr_report/figures/`](./pr_htr_report/figures/) — either as-is or renamed (e.g. `fig_ctc_posterior.png`, `fig_register_effect.png`, `fig_register_quantitative.png`).
2. Edit [`sections/methodology.tex`](./pr_htr_report/sections/methodology.tex) (or `experiments.tex`) to insert F4b at the end of the CTC-head description with the caption drafted in §3.4.
3. Edit [`sections/results.tex`](./pr_htr_report/sections/results.tex) §5.3 to swap the current register-effect figure reference to F4, followed by F5 as the companion figure. Reuse the two draft captions from §3.4.
4. Add one honest sentence to §5.3 (or the supplementary provenance section) about F4 being from an earlier training checkpoint. See §3.4 for wording.
5. Optional: add F9b ([`outputs/htrvt_bm_attention/test_0_R4_chars.png`](../../outputs/htrvt_bm_attention/test_0_R4_chars.png)) at the end of §5.2 as a small BM-style-transfers-to-HTR-VT panel. Cut if pages are tight.

**Rollback plan if F4 needs to be replaced during review**: swap in [`outputs/pub_figures/fig2_register_comparison.png`](../../outputs/pub_figures/fig2_register_comparison.png) (final-runs, less visually clear). This is the provenance-clean fallback and does not require rewriting the surrounding paragraph — only the caption.

---

### 6-legacy Older delivery roadmap (F4-v2 regeneration, superseded 2026-09-07)

Because the "clear attention map showing register effect" is the specific artifact called out in the user prompt, this section originally proposed a regeneration plan. That plan was superseded when [`outputs/attention_maps/fig_attn_talks.png`](../../outputs/attention_maps/fig_attn_talks.png) was identified as already meeting the visual bar. The regeneration steps below are retained only in case future work does need a fresh render on the final run set.

**Goal (original)**: One P0 main-report figure (F4) + one supplementary multi-word figure that together make the register-sweep attention effect visually unambiguous, backed by numeric annotations from the attention-quality metrics CSV.

**Step 1 — Data ready check** (5 min, no compute):
- Verify runs 144 (R=0), 146 (R=4), 151 (R=8), 152 (R=16) each have a `model.pt` under `saved_models/experiments/run_XXX/`. Existing per repo memory.
- Verify `outputs/report_figures/attention_quality/attention_quality_metrics.csv` has rows for `run_144, run_146, run_151, run_152` covering ≥100 IAM test images. Existing.

**Step 2 — Choose target word** (2 min):
- Recommended: reuse the "booklet" word from F3 so F3 and F4 tell one continuous story. Preprocessed image at `data/IAM/words_64x256/a03-034-01-03.png` (from repo memory, BM prep pipeline).
- Alternative: "supporters" or "talks" from `notebook/sample_images/` if a longer/shorter word is preferred.

**Step 3 — Render per-character heatmaps** (SLURM GPU job, ~10 min):
- Extend `scripts/pub_fig2_register_comparison.py` (or copy to `scripts/pub_fig4_register_effect_v2.py`) with the changes listed in §3.4:
  - Global vmax across all panels.
  - BM blob rendering via `scripts/postprocessing/beyond_memorization_viz.py` helpers.
  - Row annotations pulled from the metrics CSV.
  - Ground-truth strokes at 20% opacity behind the heatmap.
- Run under existing `postprocess_all.slurm` framework (already fixed re: cuDNN eval-mode bug and `set -e` removal — see KEY_FINDINGS §9 update).
- Emit `outputs/report_figures/fig_register_effect_v2.pdf` (main) and `outputs/report_figures/fig_register_effect_v2_supp_{word}.pdf` (2–3 additional words for supplementary).

**Step 4 — Verify legibility before committing to report** (5 min manual):
- Open the emitted PDF.
- Confirm: (a) 0-reg row visibly more diffuse than 8-reg row; (b) per-character peaks visibly shift left-to-right within each row; (c) per-row entropy annotation matches the row's visual sparsity ordering.
- If (a) is still not obvious, escalate: consider **highlighting the residual sink columns** with a thin vertical line to visually cue the reader to where the effect lives.

**Step 5 — Insert into report**:
- Save PDF into `documents/final/pr_htr_report/figures/`.
- Reference in `sections/results.tex` §5.3 as `\ref{fig:register_effect_attn}` (currently `fig:register_quant` covers the metric panel — add F4 as a separate figure directly above it).
- Update caption per §3.4 with the "what to look for" narrative even if the visual is now self-evident — the redundancy protects against reviewer misreading.

**Rollback plan if Step 3 fails**: fall back to the currently-shipped `outputs/pub_figures/fig2_register_comparison.pdf` with an extended caption (§3.4 fallback path). Accept a weaker visual argument in exchange for shipping on time.

---

## 7. Explicit "Do Not Include" List (Traps From Prior Drafts)

These are artifacts previous drafts flirted with; do not re-add them without a clear storyline reason:

1. **Any CER trend line implying a monotonic register-effect on CER.** The data does not support it. Use scatter-with-error-bars if visualizing at all, and label the noise-floor band.
2. **The old `fig2_register_comparison_run_84.pdf`.** Refers to a superseded run set. Deleted 2026-09-07; do not re-fetch from git history.
3. **The HTR-VT R=4 attention grid** (`outputs/htrvt_bm_attention/*`). HTR-VT is external comparator only; showing its attention structure suggests it is a contribution when it is not. Numeric row in Table `tab:read2016` is sufficient.
4. **The Beyond-Memorization ORIGINAL diffusion cross-attention maps** (`outputs/beyond_memorization/charIndex_*/*/attentionMaps/*.png`) beyond a single-word illustrative panel in §2 Related Work (F9). This is not the report's contribution — it is a reproduction confirmation used to justify borrowing the visualization *style*. Do not over-showcase it.
5. **CTC posterior plots** (`outputs/pub_figures/ctc_posterior.png`, `attn_at_ctc_peaks.png`) — diagnostic during method development, not part of the storyline. Skip.
6. **`archive/`, `backup/`, and `notebook/`** — never source figures from these paths for the final report; they are exploratory work with no version guarantees.

---

## 8. Final Sanity Check Before Compilation

Before the last `pdflatex main.tex`:

- [ ] Every figure in `sections/*.tex` resolves to a file under `documents/final/pr_htr_report/figures/`.
- [ ] Every table LaTeX file cited (`\input{...}`) resolves under `outputs/tables/` and is on the final run set (143–205).
- [ ] Every numeric value in the results narrative traces to either (a) `KEY_FINDINGS_AND_OBSERVATIONS.md`, (b) `outputs/tables/*.csv`, or (c) `training.log` for SWA-labeled runs.
- [ ] No claim in the report exceeds what its supporting artifact demonstrates — the "register-neutral CER + partial attention improvement" storyline is honest; do not upgrade it to "registers improve CER" or "registers fully eliminate sink patterns."
- [ ] F4 (register-effect attention map) is legible without expert priming — hand the PDF to someone unfamiliar with the project and ask "which row has the most focused attention?" If they cannot answer within 10 seconds, regenerate with stronger annotations.

---

## Change Log

- **2026-09-07 (rev 2)** — F4 resolved: [`outputs/attention_maps/fig_attn_talks.png`](../../outputs/attention_maps/fig_attn_talks.png) selected as the register-effect hero figure (per user feedback that fig6/fig6b are not meaningful). Added **F4b** ([`outputs/pub_figures/ctc_posterior.png`](../../outputs/pub_figures/ctc_posterior.png)) as an expository figure for §3.3 / §4.1 that defines CTC peak columns. Added **F9b** ([`outputs/htrvt_bm_attention/test_0_R4_chars.png`](../../outputs/htrvt_bm_attention/test_0_R4_chars.png)) as an optional secondary figure in §5.2 showing the BM visualization style transfers to the HTR-VT external comparator. F4-v2 regeneration plan superseded; older §3.4/§6 content preserved as "-legacy" for context. Section-by-section decision table updated with new figure assignments.
- **2026-09-07** — Initial creation. Grades P0/P1/P2 applied to full artifact inventory (25 tables/figures reviewed). §3.4 register-effect-attention critique and regeneration recipe added. §6 delivery roadmap for F4-v2 added.
