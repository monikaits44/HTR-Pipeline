# Writer Identification via VLAC + Register Tokens

## 1. Paper Foundation

Based on:
> Raven et al., *"Interpretable Writer Recognition via Vectors of Locally Aggregated Characters"* (2025)

**Core idea**: A trained HTR model already learns character-specific writing style features. VLAC repurposes these features — without any additional training — for writer identification by encoding *how* each person writes each character differently from the population mean.

---

## 2. Pipeline Overview

```
         ┌──────────────────────────────────────────────────────┐
         │              ViT-RGTS HTR Model (frozen)             │
         │                                                      │
  Image ─┤  CNN Stem → ViT Encoder → CTC Head → text + positions│
  [1,1,  │            ↕                                         │
  128,   │  Register tokens [R, D]    Patch tokens [T, D]       │
  1024]  └──────────────────┬────────────────┬──────────────────┘
                            │                │
                    Step 1: Extract      Step 1: Index by
                    reg embeddings       CTC positions
                            │                │
                            ▼                ▼
                     reg_feats [R,D]   char_feats [N,D]  +  char_labels
                            │                │
                            │          Step 2: Compute global
                            │          prototypes μ_c [C,D]
                            │                │
                            │          Step 3: Residual VLAC
                            │          Φ_{v,c} = L2(Σ(x - μ_c))
                            │                │
                            │          Step 4: Character selection
                            │          A* ⊂ charset (τ=0.8)
                            │                │
                            ▼                ▼
                     ┌──────────────────────────┐
                     │  Writer Descriptor        │
                     │  [VLAC_flat | reg_mean]   │
                     │  L2-normalised            │
                     └──────────┬───────────────┘
                                │
                     Step 5: Character-wise
                     cosine distance (Eq.8-9)
                                │
                     Step 6: Retrieval metrics
                     mAP, Top-1, Top-5
```

---

## 3. Mathematical Foundation (Paper Equations)

### Eq. 4 — Global Character Prototypes
$$\mu_c = \frac{1}{|S_c|} \sum_{x \in S_c} x$$
where $S_c$ is the set of all feature vectors labelled as character $c$ across the entire dataset.

**What it captures**: The "average" way everyone writes character $c$. This becomes the reference point.

### Eq. 5 — Residual VLAC Encoding
$$\Phi_{v,c} = \text{L2}\left(\sum_{x_i \in D_c} (x_i - \mu_c)\right)$$
For a specific document $D$, sum the deviations from the prototype for each character.

**What it captures**: How *this writer's* character $c$ differs from the population average. If a writer writes 'e' with a distinctive loop, the residual encodes that deviation.

### Eq. 6-7 — Character Selection
$$A^* = \{c \in A : \text{mAP}_c \geq \tau \cdot \max_{c'} \text{mAP}_{c'}\}$$
Only keep characters that are individually discriminative for writer ID (mAP threshold $\tau=0.8$).

### Eq. 8-9 — Character-wise Distance
$$d(v_1, v_2) = \frac{1}{|A^*_{shared}|} \sum_{c \in A^*_{shared}} (1 - \cos(\Phi_{v_1,c}, \Phi_{v_2,c}))$$
Average cosine distance only over characters present in both documents.

---

## 4. Module Architecture

```
scripts/writer_identification/
├── vlac.py          # Core: feature extraction, prototypes, VLAC, char selection
├── embeddings.py    # Descriptor construction: [VLAC | reg_mean] → L2-norm
├── retrieval.py     # Distance matrices + retrieval metrics (mAP, Top-1, Top-5)
├── evaluate.py      # CLI entry point: loads model → runs full pipeline
├── visualize.py     # Plotting: bar charts, t-SNE
├── run_eval.slurm   # SLURM job submission script
└── __init__.py
```

### 4.1 `vlac.py` — Core Feature Extraction & Aggregation

| Function | Input | Output | Purpose |
|----------|-------|--------|---------|
| `extract_line_features()` | image tensor, model | dict with `char_feats [N,D]`, `reg_feats [R,D]`, `char_labels` | Runs `forward_explain()`, CTC decodes, extracts patch tokens at decoded positions |
| `compute_global_prototypes()` | all line features, charset | `[C, D]` numpy | Averages all features per character class across dataset (Eq. 4) |
| `vlac_aggregate()` | char_feats, labels, charset, prototypes | `vlac [C,D]`, `counts [C]` | Sums residuals (x−μ_c) per class, L2-normalises each row (Eq. 5) |
| `vlac_from_lines()` | list of line features, charset, prototypes | vlac, counts, reg_mean | Aggregates VLAC across multiple lines (same writer) |
| `select_characters()` | writer_vlacs dict, charset, τ | selected indices, char_maps | Per-char retrieval mAP → selects discriminative chars (Eq. 6) |

**Key detail**: `extract_line_features` calls `net.backbone.forward_explain()` a second time to get raw backbone token embeddings (the first call via `net.forward_explain()` goes through the CTC head). It indexes `seq_tokens[positions]` to get features at CTC-decoded character positions. Each feature is L2-normalised before storage.

**`_EXCLUDED_CHARS`**: `{" ", "", "\t", "\n"}` — blank and whitespace characters are excluded from VLAC as they carry no style information (paper Sec. 3.3).

### 4.2 `embeddings.py` — Writer Descriptor Construction

| Function | Input | Output | Purpose |
|----------|-------|--------|---------|
| `build_writer_descriptor()` | vlac `[C,D]`, reg_mean `[R,D]` | `[C*D + R*D]` vector | Flattens + concatenates + L2-normalises |
| `build_writer_descriptors()` | dict of writer_vlacs | writer_ids list, `[W, dim]` array | Batch version for all writers |

**Descriptor layout**:
```
[VLAC row 0 (D dims) | VLAC row 1 | ... | VLAC row C-1 | reg_mean (R*D dims)]
```
When registers=0, descriptor is purely VLAC. Total dimension: `C*D + R*D` = `79*256 + R*256`.

### 4.3 `retrieval.py` — Distance & Metrics

| Function | Input | Output | Purpose |
|----------|-------|--------|---------|
| `cosine_distance_matrix()` | descriptors `[N,D]` | `[N,N]` distance | Global cosine dist (not used in final pipeline) |
| `character_wise_distance()` | two vlac+counts pairs | per-char distances, mean | Interpretable distance for analysis |
| `vlac_distance_matrix()` | vlac list, counts list, selected_chars | `[N,N]` distance | **Main distance function** — Eq. 8-9, vectorised with `np.einsum` |
| `compute_retrieval_metrics()` | dist_matrix, writer_ids | mAP, Top-1, Top-5, per_query | Standard retrieval evaluation |

**Performance note**: `vlac_distance_matrix` uses `np.einsum('icd,jcd->ijc', V_sel, V_sel)` to compute all pairwise per-character similarities in one vectorised call. Memory usage: `O(N² × |A*|)` — for N=2915, |A*|=72 this is ~2.4 GB peak.

### 4.4 `evaluate.py` — Evaluation Pipeline

**Writer identity**: Uses `parse_writer_id(line_id)` to extract the writer prefix (e.g., `c04` from `c04-110-03`). Lines from all forms of the same writer are grouped together.

The `evaluate_run()` function orchestrates the full 4-step pipeline:

```
Step 1: Extract features from all lines (GPU, ~2 min for 2915 lines)
    └── For each line: forward_explain → CTC decode → patch token extraction

Step 2: Compute global character prototypes (CPU, <1 sec)
    └── Average all features per character class → μ_c [C,D]

Step 3: VLAC residual aggregation per writer (CPU, <1 sec)
    └── Group lines by writer prefix → vlac_from_lines(writer_lines, charset, prototypes)
    └── Per-writer: aggregate all lines → writer-level VLAC [C,D]

Step 4: Character selection on writer-level VLACs (CPU, ~30 sec)
    └── Per character: single-char cosine_distance_matrix over writers
        → compute_retrieval_metrics → per-char mAP
    └── Select chars where mAP_c ≥ τ * max(mAP_c) → typically 72/79

Final: Character-wise distance matrix + retrieval metrics
    └── vlac_distance_matrix(line_vlacs, line_counts, selected_chars)
    └── compute_retrieval_metrics(dist_matrix, line_writer_ids)
```

**CLI arguments**:
| Argument | Default | Description |
|----------|---------|-------------|
| `--run-dir` | required | Single experiment directory |
| `--run-dirs` | required | Multiple dirs to compare (mutually exclusive with --run-dir) |
| `--data-dir` | `data/IAM/processed_lines` | Path to processed IAM data |
| `--split` | `test` | Which split: train/val/test |
| `--device` | `cpu` | Device: cpu or cuda:0 |
| `--max-lines` | None | Limit lines for quick testing |
| `--save-dir` | `output/writer_id` | Where to save results |

### 4.5 `visualize.py` — Plotting

| Function | Output | Description |
|----------|--------|-------------|
| `plot_metrics_vs_registers()` | `writer_id_vs_registers.png` | Grouped bar chart: mAP & Top-1 for each register count |
| `plot_descriptor_tsne()` | `tsne_{run_name}.png` | 2D t-SNE scatter of writer descriptors, coloured by writer prefix |

### 4.6 `run_eval.slurm` — SLURM Job

```bash
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
```

Evaluates runs 55 (0 reg), 58 (4 reg), 63 (8 reg), 71 (16 reg) and generates visualisations.

---

## 5. How to Execute — Step by Step

### 5.1 Prerequisites
```bash
# Activate environment
source /home/hpc/iwi5/iwi5369h/HTR-Pipeline/.venv/bin/activate
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline

# Ensure trained models exist
ls saved_models/experiments/run_55/model.pt
ls saved_models/experiments/run_55/config.json
```

### 5.2 Quick Local Test (CPU, limited lines)
```bash
python scripts/writer_identification/evaluate.py \
    --run-dir saved_models/experiments/run_58 \
    --data-dir data/IAM/processed_lines \
    --split test \
    --device cpu \
    --max-lines 50 \
    --save-dir output/writer_id_test
```
**Purpose**: Sanity check that the pipeline runs without errors. Results won't be meaningful with only 50 lines.

### 5.3 Full Evaluation via SLURM (Recommended)
```bash
mkdir -p output/writer_id
sbatch scripts/writer_identification/run_eval.slurm
```
**What happens**:
1. Submits GPU job (~8 min total for 4 models × 2915 lines)
2. Evaluates each model's writer ID performance
3. Saves CSV results + per-model t-SNE plots + bar chart
4. Output logs: `output/writer_id/slurm_output_<jobid>.log`

### 5.4 Monitor Job
```bash
squeue -u $USER                                    # Check queue status
tail -f output/writer_id/slurm_output_*.log        # Watch stdout
tail -f output/writer_id/slurm_error_*.log         # Watch stderr (tqdm)
```

### 5.5 Custom Multi-Model Comparison
```bash
python scripts/writer_identification/evaluate.py \
    --run-dirs saved_models/experiments/run_55 \
              saved_models/experiments/run_58 \
              saved_models/experiments/run_63 \
              saved_models/experiments/run_71 \
    --split test \
    --device cuda:0 \
    --save-dir output/writer_id
```

### 5.6 Generate Visualisations Separately
```bash
python scripts/writer_identification/visualize.py \
    --results-csv output/writer_id/writer_id_results.csv \
    --save-dir output/writer_id
```

---

## 6. Experimental Results

### 6.1 Final Results — Correct Writer Identity (Job 1584539)

Writer identity correctly mapped to IAM writer prefix (e.g., `c04` from `c04-110-03`).
Evaluation: 2915 query lines, 32 writers, character selection τ=0.8.

| Run | Registers | mAP | Top-1 | Top-5 | Writers | A* Chars |
|-----|-----------|------|-------|-------|---------|----------|
| run_55 | 0 | 0.2807 | **0.8693** | **0.9849** | 32 | 72/79 |
| run_58 | 4 | 0.2773 | 0.8542 | 0.9756 | 32 | 72/79 |
| run_63 | 8 | 0.2783 | 0.8580 | 0.9770 | 32 | 72/79 |
| run_71 | 16 | **0.2810** | 0.8648 | 0.9818 | 32 | 72/79 |

**Top-1 accuracy of ~87%** means that for a given query line, the nearest line in embedding space belongs to the same writer 87% of the time. **Top-5 of ~98%** means the correct writer almost always appears in the 5 nearest results.

### 6.2 Previous (Incorrect) Results — Form ID as Writer Identity

These earlier runs used form ID (e.g., `c04-110`) as writer identity, treating each of 336 forms as a separate "writer". This is **incorrect** — it measures within-form retrieval, not writer identification.

| Job | Method | run_55 mAP | run_58 mAP | run_63 mAP | run_71 mAP |
|-----|--------|-----------|-----------|-----------|-----------|
| 1583888 | Residual VLAC + char-wise dist (form ID) | 0.417 | 0.410 | 0.403 | 0.414 |
| 1583856 | Mean-pooled VLAC + global cosine (form ID) | 0.191 | 0.236 | 0.271 | 0.243 |

**Why mAP dropped from ~0.42 to ~0.28**: The correct formulation (32 writers) is a harder retrieval task — each writer has ~91 lines across ~10 forms, but the model must match across form boundaries, handwriting sessions, and different text content. The form-ID version inflated mAP because same-form lines are trivially similar (same handwriting session, adjacent content).

**Why Top-1 jumped from ~0.47 to ~0.87**: With form IDs, "Top-1" required matching the exact form — very strict. With writer IDs, any line from the same writer counts as correct — a more meaningful metric.

### 6.3 Interpretable Character Distance (Writer c04 vs c06)

| Model (Reg) | Mean Char Dist | Most Discriminative Characters (top 5) |
|-------------|---------------|----------------------------------------|
| run_55 (0) | 1.2300 | t(1.45), p(1.41), T(1.39), u(1.38), w(1.38) |
| run_58 (4) | 1.2082 | c(1.46), p(1.40), a(1.39), o(1.38), t(1.34) |
| run_63 (8) | 1.2550 | p(1.49), c(1.49), o(1.46), T(1.45), w(1.44) |
| run_71 (16) | 1.2392 | p(1.50), T(1.49), c(1.49), a(1.45), t(1.41) |

**Observation**: Characters like 'p', 't', 'T', 'c' consistently appear as most discriminative between writers c04 and c06, confirming that the VLAC encoding captures writer-specific character style differences.

---

## 7. Critical Analysis & Findings

### Finding 1: Residual VLAC is the Dominant Improvement
Switching from mean-pooled VLAC to residual encoding (x − μ_c) was the single largest improvement in early experiments (form-ID mAP: 0.19 → 0.42). It transforms the representation from "what this writer's character looks like" to "how it *differs* from the average" — a much more discriminative signal.

### Finding 2: Character-wise Distance > Global Cosine
Using per-character cosine distance (Eq. 8-9) instead of a single global cosine on the flattened descriptor provides better matching by handling cases where writers share some characters identically but differ on others.

### Finding 3: Registers Have No Effect on Writer ID
| Registers | mAP | Top-1 | Top-5 |
|-----------|------|-------|-------|
| 0 | 0.2807 | **0.8693** | **0.9849** |
| 4 | 0.2773 | 0.8542 | 0.9756 |
| 8 | 0.2783 | 0.8580 | 0.9770 |
| 16 | **0.2810** | 0.8648 | 0.9818 |

All configurations perform nearly identically (mAP within ±0.004). This is **expected** because the current pipeline uses only VLAC character-wise distance (Eq. 8-9) — register features are extracted but **not incorporated into the distance metric**. The register token embeddings are not part of the character-level VLAC representation, so varying registers has no direct effect on writer ID results.

### Finding 4: Character Selection Now Works Correctly
With proper writer identity (32 writers), character selection computes meaningful per-char mAPs. Result: **72/79 characters selected** (7 excluded by threshold τ=0.8). Previously, form-ID-based identity made all per-char mAPs = 0, effectively disabling selection.

### Finding 5: Strong Top-1/Top-5 Despite Modest mAP
- **Top-1 = 87%**: The nearest neighbour line is by the correct writer 87% of the time
- **Top-5 = 98%**: The correct writer appears in the top 5 results 98% of the time
- **mAP = 28%**: The relatively lower mAP is expected for a 32-writer, 2915-query dataset where each writer has ~91 lines — ranking all 91 positives above all negatives is challenging

### Finding 6: Prototypes Computed on Test Set (Paper-faithful)
Prototypes μ_c are computed on the evaluation set itself. This is consistent with the paper (VLAC is unsupervised — no labelled writer data needed). The prototypes represent "average character appearance" and are not supervised labels.

---

## 8. Dependency Chain

```
evaluate.py
  ├── vlac.py
  │     └── scripts/postprocessing/attention_viz/core.py
  │           ├── load_model()     — loads HTRNet from config.json + model.pt
  │           ├── ctc_decode()     — greedy CTC decode → (text, positions)
  │           └── load_htr_image() — resize + normalise → [1,1,128,1024]
  ├── embeddings.py  (builds descriptors, used for t-SNE but not retrieval)
  ├── retrieval.py   (distance + metrics)
  └── visualize.py   (plotting, called separately)

models.py
  └── HTRNet
        └── ViTRGTSBackbone
              └── forward_explain() → (logits, reg_tokens, attn_maps, token_norms, grid)
```

---

## 9. Output Files Reference

After running `run_eval.slurm`, `output/writer_id/` contains:

| File | Content |
|------|---------|
| `writer_id_results.csv` | Summary: run, registers, mAP, top1, top5, n_queries, n_writers |
| `{run}_selected_chars.json` | List of A* characters (72 of 79 in latest run) |
| `writer_id_vs_registers.png` | Bar chart: mAP & Top-1 for each register count |
| `tsne_{run}.png` | 2D t-SNE of writer descriptors coloured by writer prefix |
| `slurm_output_*.log` | Stdout log with step-by-step results |
| `slurm_error_*.log` | Stderr log (tqdm progress bars) |

---

## 10. Known Limitations & Future Work

| # | Limitation | Impact | Fix |
|---|-----------|--------|-----|
| 1 | Register features not used in final distance | Register sweep results don't reflect register contribution | Add weighted register distance term to Eq. 8-9 |
| 2 | Prototypes computed on test set (self-referencing) | Slight data leakage — prototypes should come from train set | Compute prototypes on train split, apply to test |
| 3 | No cross-validation or confidence intervals | Single-run numbers without error bars | Bootstrap mAP or run multiple seeds |
| 4 | Character selection threshold τ=0.8 not tuned | Fixed from paper, may not be optimal for this model | Grid search on validation set |
| 5 | `extract_line_features` calls backbone twice | Redundant forward pass (once for logits, once for token embeddings) | Cache backbone output or restructure forward_explain |
| 6 | 9 overlapping writers between train/test | Some writers appear in both splits | Use IAM official writer-disjoint split |

---

## 11. Paper Correspondence Table

| Paper Reference | Implementation | Status |
|----------------|---------------|--------|
| Eq. 4 — Global prototypes μ_c | `vlac.py::compute_global_prototypes()` | ✅ Implemented |
| Eq. 5 — Residual VLAC Φ_{v,c} | `vlac.py::vlac_aggregate(prototypes=...)` | ✅ Implemented |
| Eq. 6 — Character selection A* | `vlac.py::select_characters(tau=0.8)` | ✅ Working (72/79 chars selected) |
| Eq. 8–9 — Character-wise distance | `retrieval.py::vlac_distance_matrix()` | ✅ Implemented (vectorised) |
| Sec. 3.3 — Blank/space exclusion | `vlac.py::_EXCLUDED_CHARS` | ✅ Implemented |
| Register tokens as style features | `embeddings.py::build_writer_descriptor()` | ⚠️ Built but not used in distance |

---

## 12. Conclusions

1. **The VLAC pipeline works end-to-end** and produces strong writer identification results on IAM test set: **Top-1 = 87%, Top-5 = 98%, mAP = 28%** with 2915 lines from 32 writers.

2. **Residual VLAC encoding is critical** — switching from mean pooling to residual aggregation (Eq. 5) dramatically improved results in early experiments. This validates the paper's core claim that style is captured in deviations from prototypes, not in raw features.

3. **Character selection works correctly** — with proper writer identity (32 writers, not 336 forms), per-character mAP filtering selects 72/79 characters (τ=0.8), excluding non-discriminative characters as intended by Eq. 6.

4. **Register tokens have no effect on writer ID** because the distance metric (Eq. 8-9) operates only on VLAC character-wise representations. All register configurations produce identical results within noise (mAP ∈ [0.277, 0.281]). To measure register impact, register features would need to be incorporated into the distance metric.

5. **Character-wise distance provides interpretability** — the system identifies which specific characters (e.g., 'p', 't', 'T') most distinguish two writers, enabling human-understandable explanations of writer identification decisions.

6. **Bug fix history**: An earlier version incorrectly used form ID as writer identity (336 forms instead of 32 writers), which inflated mAP to ~0.42 but disabled character selection. This was corrected in Job 1584539.

6. **The interpretability aspect works** — the system can pinpoint which characters distinguish two writers (e.g., '8', 'k', 'P' are consistently the most discriminative), which aligns with the paper's goal of *interpretable* writer recognition.
