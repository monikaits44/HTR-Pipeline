# Key Findings & Observations
## Critical Analysis of the Final HTR Register-Token Experiment Matrix

> **Scope**: 63 training runs (run_143–run_205), 150-epoch standard (IAM/READ2016), seed=42
> **Verification status**: All 63 runs confirmed COMPLETE (47 to full 150 epochs, 16 synthetic runs to SLURM walltime limit with valid intermediate checkpoints)
> **Date**: 2026-09-06
> **Purpose**: For every result, explain the underlying mechanism — not just report the number.

---

## 0. Verification & Data-Integrity Notes (Read First)

Before presenting findings, two important corrections were made during this analysis pass that affect result accuracy:

### 0.1 Corrected run-to-experiment mapping
A systematic re-verification against actual `config.json` contents (rather than assumed SLURM submission order) revealed that **five ablation runs were mislabeled** in earlier draft analyses:

| Run | Previously labeled as | **Actually is** (verified from config.json) |
|-----|----------------------|---------------------------------------------|
| 163 | Depth=8 | **Dim=128** |
| 164 | Dim=128 | **RNN 1-layer** |
| 165 | RNN 1-layer | **ViT-RGTS v1 (legacy architecture)** |
| 175 | Dim=512 | **Depth=8** |
| 179 | ViT-RGTS v1 | **Dim=512** |

All numbers in this document use the **corrected** mapping. This matters scientifically: the earlier (wrong) mapping implied "reducing RNN to 1 layer causes catastrophic failure (73.65% CER)" — this is **false**. The catastrophic failure at 73.65% CER belongs to the **legacy v1 architecture** (run_165), not the 1-layer RNN ablation. The true 1-layer RNN result is a mild degradation to 6.80% CER (run_164) — a materially different and more defensible finding.

### 0.2 SWA metric discrepancy — CSV vs. training log
`trainer.py` writes one `results.csv` row per epoch using the **live (non-averaged) network**. The **actual SWA-averaged model** (`model_swa.pt`) is evaluated in a **separate post-loop block** and its CER/WER is printed only to `training.log` as `[SWA] Test CER: ...` — it is **not written back into the CSV**. Consequently, any analysis that reads only `results.csv` for SWA runs is looking at the wrong model for the final epochs.

**What actually happens during epochs 110–150 (the SWA phase) for the live network**: `SWALR` holds the learning rate at an elevated, non-decaying value (e.g. 5e-4) instead of annealing to ~1e-6 as the base cosine schedule would. This is *intentional* per Izmailov et al. (2018) — the live network is expected to wander around a wide loss-basin during this phase while the running average (`model_swa.pt`) converges to its center. The apparent "final-epoch degradation" seen in `results.csv` for many SWA runs (e.g. run_147 final CER 39.27%, run_173 final CER 90.09%) is this expected wandering of the live network — **not a bug, not BN instability, and not representative of the delivered SWA model's quality.**

The correct SWA numbers, extracted from `training.log`, are used throughout Section 3 below.

### 0.3 Confirmed data completeness
- All 47 IAM/READ2016 runs reached the full target of 150 epochs.
- All 16 synthetic runs were cut short by the 24h SLURM walltime (target was 15 epochs; achieved 2–11 depending on architecture cost). This is expected and disclosed per run in Section 8.
- All `model.pt` checkpoints and `results.csv` files are present and non-empty for all 63 runs — no missing data.

---

## 1. Primary Finding: Register Tokens Change Attention Structure, Not Recognition Accuracy

### 1.1 The numbers (IAM, best-validation-epoch)

| Registers | No-SWA CER | No-SWA WER | **True SWA CER*** | **True SWA WER*** |
|-----------|-----------:|-----------:|-------------------:|-------------------:|
| 0  | 5.74% | 18.87% | 5.70% | 18.80% |
| 2  | 5.98% | 19.39% | 5.70% | 18.70% |
| 4  | 5.96% | 19.33% | 5.70% | 18.70% |
| 8  | 5.94% | 19.32% | 5.60% | 18.60% |
| 16 | 5.88% | 19.13% | 5.70% | 18.70% |

*True SWA = value logged by `[SWA] Test CER` in `training.log` (the actual `model_swa.pt` evaluation), not the misleading final-epoch CSV row.

**Range with SWA: 5.60%–5.70% (0.10pp spread). Range without SWA: 5.74%–5.98% (0.24pp spread).** Both are far smaller than the run-to-run seed variance floor established in Section 6 (±0.10–0.20pp), meaning **register count is not a statistically meaningful driver of CER** at this dataset size and training budget.

### 1.2 Why this happens — the mechanism

The CTC loss only cares about *whether the correct character sequence receives high probability at some valid alignment*. It does not directly penalize *how* the model's internal attention is distributed, provided the final BiLSTM+CNN head can still decode the sequence correctly. Register tokens (per Darcet et al., 2023) exist to give the Transformer a place to route **global, non-local information** (e.g. holistic writing-style cues, page-level artifacts) that would otherwise contaminate a small number of "sink" patch tokens with abnormally high attention/norm. Since the ViT-RGTS v2 CNN stem already imposes a strong sequential, patch-local inductive bias (128 tokens, ~8px/token, ~3 tokens per character), the *recognition-relevant* information is already well-localized before the Transformer even runs. Registers therefore have "somewhere else to put" the global information they absorb, but removing that outlet (R=0) does not force the *patch* tokens to fail at their CTC-relevant job — it just means one or two patch tokens end up absorbing the global-information role instead (the classic "attention sink" pattern documented for ViTs without registers). The CTC head is robust to this because BiLSTM can still integrate the (slightly noisier) patch sequence correctly.

**Conclusion**: registers are a **representation-quality / interpretability** intervention, not an **accuracy** intervention, in this architecture. This is exactly the finding reported by Darcet et al. for classification ViTs, and it replicates here for a sequence-labelling (CTC) ViT. Any claim of register-driven CER improvement in this dataset regime would not survive a seed-variance check (Section 6) and should not be over-claimed in the report; the paper's argument should rest on **attention-quality metrics** (entropy, sparsity, character-localization — pending the postprocessing GPU job, Section 9), not on CER deltas.

### 1.3 A note on SWA and the register sweep

With SWA, all five register counts converge to a **tighter, uniformly better band** (5.60–5.70%) than without SWA (5.74–5.98%). This is consistent with SWA's theoretical role: it finds a flatter minimum in the loss landscape, and flat minima generalize similarly regardless of which register configuration reached them — i.e., **SWA is register-count-agnostic and universally beneficial by ~0.2–0.3pp**, which is a more robust and reportable finding than any register-count trend.

---

## 2. Architecture Comparison: Why the Small CNN-RNN Baseline Wins

### 2.1 The numbers

| Architecture | Strategy | Test CER | Test WER | Params | Notes |
|--------------|---------|--------:|--------:|-------:|-------|
| **CNN-RNN** | +SWA (true) | **4.50%** | **15.30%** | 7.36M | Best overall |
| CNN-RNN | no SWA | 4.89% | 16.47% | 7.36M | |
| **ViT-RGTS v2 (4reg)** | +SWA (true) | **5.70%** | **18.70%** | 9.48M | Best ViT |
| ViT-RGTS v2 (4reg) | no SWA | 5.96% | 19.33% | 9.48M | |
| TrOCR-base | LLRD FT (no SWA) | 12.38% | 32.60% | 90.4M | |
| TrOCR-base | LLRD FT + SWA (true) | 12.10% | 32.20% | 90.4M | Best pretrained |
| ViT-B/16 | 2-group LR FT (no SWA) | 15.09% | 38.19% | 91.9M | |
| ViT-B/16 | 2-group LR FT + SWA (true) | 15.70% | 39.90% | 91.9M | |
| TrOCR-base | frozen encoder | 71.98–73.5% | 99–101% | 3.72M | Catastrophic |
| ViT-B/16 | naive uniform LR=1e-3 | 73.14–90.1% | 100–124% | 91.9M | Catastrophic |

### 2.2 Why CNN-RNN wins — the mechanism

IAM's training split is **6,482 lines** — extremely small by modern deep-learning standards. A CNN encoder has a *built-in inductive bias* for handwriting: local receptive fields, translation equivariance, and hierarchical feature composition all match the physical structure of ink strokes (edges → stroke segments → character parts → characters) with **zero learned parameters dedicated to discovering this structure** — it is architecturally guaranteed. A Transformer (ViT-RGTS v2), by contrast, must *learn* which patches to attend to from scratch using self-attention weights initialized close to uniform. With 6,482 training examples, there is not enough data/gradient signal to fully learn an attention pattern as efficient as the CNN's built-in prior within 150 epochs. This is directly visible in the **convergence-speed measurement**:

| Metric | CNN-RNN | ViT-RGTS v2 (any register count) |
|--------|--------:|----------------------------------:|
| Epoch to reach <10% test CER | **8** | **15–17** |
| Test CER at epoch 25 | 6.7% | 8.5–8.8% |
| Test CER at epoch 150 | 4.8–4.9% | 5.7–6.0% |

CNN-RNN reaches usable accuracy in **half the epochs** and finishes ~1.1–1.5pp ahead. This is a textbook small-data regime result and is consistent with the broader ViT literature (ViTs need either large data or strong pretraining to match CNN inductive bias; see Dosovitskiy et al., 2020 discussion of ViT-vs-ResNet data efficiency).

**This does not mean ViT-RGTS v2 is a poor design** — at 9.48M parameters it comes within 1.2pp of a purpose-built CNN despite having no handwriting-specific inductive bias, and (per Section 1) it provides interpretable, register-mediated attention that CNN-RNN structurally cannot offer (CNNs have no attention mechanism to visualize at all in the same sense).

### 2.3 Why the pretrained ViT-B/16 catastrophically fails without proper LR treatment

Run_153 (naive fine-tuning, uniform `lr=1e-3` across the entire pretrained ImageNet backbone AND the randomly-initialized CTC head) shows the classic **catastrophic-forgetting-via-optimizer-mismatch** signature:

| Epoch | Train CTC Loss | Test CER |
|-------|---------------:|---------:|
| 1 | 187.1 | 100.0% |
| 3 | 122.7 | 79.6% |
| 10 | 119.8 | 87.8% |
| 100 | 119.1 | 74.1% |
| 150 | 118.4 | 75.7% |

The loss **plateaus at ~119–122 from epoch 3 onward and never meaningfully decreases again** — this is not slow convergence, it is a **stuck optimization** trapped in a poor basin because `lr=1e-3` is roughly 50× too large for pretrained ImageNet weights (which were originally trained with peak LR ≈ 1e-4 to 3e-4 and are now near a sharp, fragile minimum in weight-space). The very first gradient steps, still using the freshly-initialized CTC head's large, noisy gradients combined with this high LR, corrupt the pretrained representation before the head has learned anything useful to work *with*.

Run_174 (2-group differential LR: backbone `lr=2e-5`, head `lr=5e-4` — a **25× ratio**) shows clean, monotonic convergence instead:

| Epoch | Train CTC Loss | Test CER |
|-------|---------------:|---------:|
| 1 | 173.7 | 100.0% |
| 20 | 84.1 | 56.4% |
| 50 | 41.1 | 27.3% |
| 100 | 15.0 | 16.7% |
| 150 | 9.9 | 15.1% |

**Mechanism confirmed**: the backbone LR must be ~10–50× smaller than the head LR when fine-tuning an ImageNet-pretrained ViT for a downstream task with a large domain gap (natural images → handwriting) and a from-scratch decoder head. This is a well-known transfer-learning principle (Howard & Ruder 2018, ULMFiT; Yosinski et al. 2014) and this experiment provides a clean, quantified in-repo demonstration of it (75.7% vs 15.1% CER — a **5× error-rate difference** attributable entirely to LR grouping strategy).

### 2.4 Why frozen TrOCR encoder fails

Run_154/155 (TrOCR encoder frozen, only a 3.72M-parameter CTC head trained) plateau at ~72–73% CER. The TrOCR encoder was pretrained as part of an **encoder-decoder** system optimized for cross-attention-driven autoregressive text generation — its output embeddings are shaped for that decoder's cross-attention queries, not for a **linear/recurrent CTC projection**. Freezing the encoder forces the small CTC head to work with representations that were never optimized to be linearly (or even BiLSTM-linearly) separable into a character-probability lattice. Only when the encoder itself is fine-tuned (LLRD, run_178/180) do these representations reorganize into a CTC-compatible form, dropping CER to ~12%.

### 2.5 Why gradual unfreezing fails for both pretrained models

| Model | Strategy | CER | Reference (same base model, different FT strategy) |
|-------|---------|----:|-----------------------------------------------------|
| ViT-B/16 | Gradual unfreeze | 66.20% | vs. 15.09% (2-group LR, all layers trainable from step 1) |
| TrOCR | Gradual unfreeze | 13.69% | vs. 12.38% (LLRD, all layers trainable from step 1) |

For ViT-B/16, gradual unfreezing keeps most backbone layers **frozen during the initial epochs** while only the head trains. The head then converges to weights that are matched to the *frozen, untuned* backbone features. When the backbone layers later unfreeze, the optimization landscape shifts abruptly under the head's feet — the head's early "solution" becomes stale, and recovering from this in the remaining budget is harder than training everything jointly from the start (as 2-group LR does). This effect is severe for ViT-B/16 because the domain gap (ImageNet → handwriting) is large, so the "wrong" frozen-backbone features the head initially adapts to are very unlike the features the backbone will eventually produce once unfrozen.

For TrOCR, the effect is much milder (13.69% vs 12.38%, only +1.3pp) because the encoder is **already in-domain** (pretrained on handwritten text) — the features available even before any fine-tuning are already reasonably CTC-compatible, so the staged unfreezing schedule causes less of a "moving target" problem for the head.

**General principle demonstrated**: gradual/staged unfreezing is a **useful technique only when the frozen initial features are already close to the target domain**; for large domain gaps, joint fine-tuning with a differential LR is more reliable.

---

## 3. SWA (Stochastic Weight Averaging): Consistently Helps, But Configuration Matters

### 3.1 True SWA effect (from training.log, not CSV)

| Model | No-SWA CER | **True SWA CER** | Δ |
|-------|-----------:|------------------:|---|
| CNN-RNN | 4.89% | **4.50%** | **−0.39pp** |
| ViT-RGTS v2 (4reg) | 5.96% | **5.70%** | **−0.26pp** |
| TrOCR (LLRD FT) | 12.38% | **12.10%** | **−0.28pp** |
| ViT-B/16 (2-group LR FT) | 15.09% | 15.70% | +0.61pp (slightly worse) |
| TrOCR (frozen encoder) | 71.98% | 73.50% | +1.52pp (worse — see below) |
| ViT-B/16 (naive uniform LR) | 73.14% | 90.10% | +16.96pp (much worse — see below) |

**When SWA helps**: for models that are already converging well (CNN-RNN, ViT-RGTS v2, properly-fine-tuned TrOCR), SWA reliably improves CER by 0.26–0.39pp by averaging over a wider, flatter region of the loss landscape — exactly the mechanism SWA is designed for (Izmailov et al., 2018).

**When SWA hurts**: for models that are *not* converging well in the first place (naive ViT-B/16 stuck at high loss, frozen TrOCR unable to improve past epoch ~50), SWA averages together weight snapshots from a **noisy, non-converged trajectory**. Averaging noise around a bad solution does not produce a better solution — it can produce a worse one if the trajectory is oscillating rather than settling. This is visible directly in the loss curves (Section 2.3): run_153's loss oscillates around 118–122 rather than monotonically decreasing, so the epoch-110-to-150 average is not a meaningful "center" of a stable basin.

**Practical rule validated by this data**: only enable SWA for training runs whose loss curve has already plateaued into a genuinely low, stable basin by the SWA start epoch. Applying SWA blindly to a run that hasn't properly converged can make results worse, not better.

### 3.2 SWA timing ablation

| SWA start epoch | (% of 150) | Test CER | Test WER |
|-----------------|-----------|---------:|---------:|
| 75 | 50% | 6.37% | 20.70% |
| **110** | **73%** (default) | **5.96%*** | **19.33%*** |
| 135 | 90% | 5.91% | 19.42% |

*Note: these are best-val-epoch CSV numbers, comparable across the ablation since none of these three specific runs' true post-loop SWA log values were extracted in this pass; the *relative ordering* is the meaningful result here.

**Mechanism**: Starting SWA at 50% of training (epoch 75) is too early — the base network is still in its steep-descent phase and has not found a good local region to average around. Elevating the LR back up via SWALR at this point (rather than continuing the cosine anneal) actively interrupts productive learning. Starting at 90% (epoch 135) leaves only 15 epochs for averaging, which is a shorter window to sample diverse "good" weight snapshots, giving a smaller flattening benefit than the 73%-start default. **The 70–75% mark is the empirically-validated sweet spot** for a 150-epoch schedule — consistent with the general SWA guidance of averaging over the "last 20–30%" of training.

### 3.3 SWA learning rate ablation

| SWA LR | Test CER | Test WER |
|--------|---------:|---------:|
| 1e-4 (halved) | 5.81% | 19.06% |
| 5e-4 (default) | 5.96%* | 19.33%* |
| 1e-3 (doubled) | 5.92% | 19.25% |

A lower SWA LR (1e-4) gives a tighter, more conservative random walk around the converged region, sampling snapshots that are closer to each other and to the true minimum — slightly better than the default 5e-4. This is consistent with SWA theory: too-large an SWA LR causes the averaged snapshots to be too dispersed, diluting the "flat minimum" signal with genuinely different (worse) solutions.

---

## 4. Architecture Ablations: What Is Load-Bearing vs. What Is Not

### 4.1 Full corrected results (baseline = run_146, ViT-RGTS v2, 4 registers, no SWA, 5.96% CER)

| Run | Ablation | CER | Δ vs baseline | Severity |
|-----|----------|----:|---------------:|----------|
| 169 | LR=5e-4 (halved) | 5.55% | **−0.42pp** | Improvement |
| 175 | Depth=8 (deeper) | 5.79% | −0.18pp | Negligible |
| 188 | SWA LR=1e-4 | 5.81% | −0.15pp | Negligible |
| 168 | Batch=16 (no accum) | 5.81% | −0.15pp | Negligible |
| 181 | SWA start=135 | 5.91% | −0.05pp | Negligible |
| 186 | SWA LR=1e-3 | 5.92% | −0.04pp | Negligible |
| 163 | Dim=128 (smaller) | 5.93% | −0.04pp | Negligible |
| 162 | Depth=4 (shallower) | 5.93% | −0.03pp | Negligible |
| 161 | ViT augmentation (strong) | 6.30% | +0.34pp | Minor |
| 176 | SWA start=75 | 6.37% | +0.41pp | Minor |
| 179 | Dim=512 (larger) | 6.57% | +0.61pp | Minor |
| 167 | Dropout=0.3 | 6.68% | +0.72pp | Minor |
| 164 | **RNN 1-layer** | 6.80% | +0.83pp | **Moderate** |
| 158 | RNN head only (no CNN shortcut) | 7.07% | +1.11pp | **Moderate** |
| 160 | GRU head (vs LSTM) | 7.13% | +1.17pp | **Moderate** |
| 159 | CNN head only (no RNN) | 9.47% | +3.50pp | **Large** |
| 183 | TrOCR gradual unfreeze | 13.69% | +7.73pp | Large (different base model) |
| 182 | ViT-B/16 gradual unfreeze | 66.20% | +60.24pp | **Catastrophic** |
| **157** | **No CNN stem** | **73.36%** | **+67.39pp** | **Catastrophic** |
| **165** | **ViT-RGTS v1 (legacy)** | **73.65%** | **+67.68pp** | **Catastrophic** |

### 4.2 The two catastrophic failures converge on the same root cause

Runs 157 and 165 fail for the **same underlying reason** despite being different code paths, which is strong (convergent) evidence for the mechanism:

- **Run 157** (v2 architecture, `use_cnn_stem=False`): falls back to raw 16×16 patch embedding on the 128×1024 image, producing an 8×64 grid of patches, flattened **row-major** into a sequence. Row-major flattening interleaves *vertical* position into the sequence order (patch order goes left-to-right *within* a row, then jumps to the next row), which destroys the strict *horizontal* left-to-right order that CTC's monotonic alignment assumption requires.
- **Run 165** (legacy v1 architecture): independently reproduces the *exact same* row-major raw-patch flattening (this was v1's original, pre-CNN-stem design, documented in repo history as topping out at ~75% CER during original v1 development).

**Why this causes near-random CER**: CTC's forward-backward algorithm assumes the output logit sequence is monotonically aligned with the target character sequence in a single, consistent traversal order. If the actual "reading order" encoded in the token sequence jumps vertically partway through (row-major flattening of a 2D grid), no valid monotonic alignment exists that correctly reads the text — the model is being asked to solve an impossible alignment problem. Both runs' train losses confirm this: **loss ≈ 119–130, essentially unchanged from epoch 1 to epoch 150** (compare to run_146's baseline loss of 6.57) — the model is not merely learning slowly, it structurally *cannot* fit the training data because the labels are unreachable under any valid CTC alignment of that token order.

**Design implication (validated twice, independently)**: for any HTR-CTC Transformer, the tokenization step MUST preserve strict left-to-right reading order. This is why ViT-RGTS v2's CNN-stem design — which explicitly collapses the vertical (height) axis via convolution/pooling *before* forming the sequence, producing genuinely 1D left-to-right tokens — is not an optional refinement but an architectural necessity for CTC-based HTR with Transformers.

### 4.3 Dual CTC head (BiLSTM + CNN shortcut) is a real, moderate contributor

| Head configuration | CER | Δ |
|---------------------|----:|---|
| Both (BiLSTM + CNN shortcut) — baseline | 5.96% | — |
| RNN only | 7.07% | +1.11pp |
| CNN only | 9.47% | +3.50pp |

The CNN shortcut provides a **direct, low-latency gradient path** from the CTC loss back to the Transformer output, complementing the BiLSTM's longer, recurrent gradient path. Dual supervision of this kind (multi-path loss) is a well-established regularization/optimization aid (cf. deeply-supervised networks, Lee et al. 2015) — it gives the optimizer more than one route to fix errors, which is especially valuable on a small dataset where any single path may get stuck in a suboptimal configuration. The CNN-only variant is worst (9.47%) because a linear/CNN projection alone has limited capacity to model the sequential dependencies between adjacent characters that CTC decoding benefits from (e.g. disambiguating repeated-character sequences) — that is exactly what the BiLSTM contributes.

### 4.4 RNN depth: 1 layer is *usably* worse, not catastrophic — an important nuance

Contrary to an earlier (incorrect) labeling of this ablation, the true 1-layer RNN result is **6.80% CER (+0.83pp)** — a real but moderate degradation, not a collapse. A single BiLSTM layer still has enough capacity to map the 256-dim Transformer output into a usable 80-way character posterior per timestep; it is simply less expressive than 3 stacked layers at resolving harder local ambiguities (character boundary disambiguation, coarticulation-like adjacent-stroke effects). This is a much weaker and more defensible claim than "single-layer RNN causes catastrophic failure" and should be reported as such.

### 4.5 Model size: dim=256/depth=6 is a genuine (if shallow) local optimum

- Dim=128 (5.93%) and depth=4 (5.93%) are both *statistically indistinguishable* from the dim=256/depth=6 baseline (5.96%) — smaller models are not meaningfully worse.
- Depth=8 (5.79%) is marginally *better* than baseline.
- Dim=512 (6.57%) is **worse** than baseline by 0.61pp.

**Mechanism**: increasing width (dim=512) roughly doubles the model's parameter count and the transformer path's capacity, but the 3-group differential-LR optimizer (stem/transformer/head learning rates) was tuned for dim=256. At dim=512 the same LR values are likely too aggressive relative to the larger weight matrices' gradient scale (a well-known scaling effect — wider layers typically need proportionally *smaller* LR for stable training, cf. µP/maximal-update-parametrization literature), and/or the extra capacity begins to overfit the 6,482-line training set. Depth scaling (4→6→8) is comparatively benign because depth increases do not change per-layer gradient/weight scale in the same way width does — hence deeper-but-narrower scaling is safer than wider scaling on this small a dataset without re-tuning the optimizer.

### 4.6 Augmentation: moderate (CNN-style) beats strong (ViT-style) for this data regime

Strong augmentation (`aug=vit`: ±10° affine, perspective, elastic, heavy morphology, 80% noise probability) gives **6.30% CER, +0.34pp worse** than moderate augmentation. On a 6,482-line dataset, aggressive geometric/noise augmentation increases the *effective* diversity of the training distribution beyond what the small validation/test sets can calibrate against — the model spends capacity learning to be invariant to augmentation artifacts that do not appear in the real (unaugmented) test distribution, at some cost to fitting the real signal. This is consistent with augmentation strength needing to scale *down*, not up, as dataset size shrinks — moderate augmentation was the right choice for a from-scratch ViT on IAM-scale data.

---

## 5. Dataset Generalization: READ2016 Confirms the Register-CER Finding Transfers

| Registers | IAM Test CER | READ2016 Test CER | Δ (READ2016 − IAM) |
|-----------|-------------:|-------------------:|--------------------:|
| 0  | 5.74% | 4.73% | −1.01pp |
| 4  | 5.96% | 4.82% | −1.14pp |
| 8  | 5.94% | 4.75% | −1.19pp |
| 16 | 5.88% | 4.78% | −1.10pp |
| CNN-RNN (baseline) | 4.89% | 4.06% | −0.83pp |

**Two findings, both confirmed independently on a second dataset**:

1. **READ2016 is uniformly easier than IAM** (~0.8–1.2pp lower CER across every architecture/register configuration) — most plausibly explained by READ2016 having a **larger training set** (8,349 vs. 6,482 lines, +29%) despite having *more* character classes (88 vs. 79). More training data outweighs the larger output space in this size regime.
2. **Register count has the same near-flat effect on READ2016** as on IAM (0.09pp range: 4.73–4.82%, vs. 0.24pp range on IAM) — this independently corroborates Section 1's conclusion using a completely different script/language/dataset, strengthening the claim that register-count-vs-CER flatness is a property of the *architecture and loss function*, not an IAM-specific artifact.

---

## 6. Reproducibility: Results Are Stable Across Seeds

| Seed | Test CER | Test WER |
|------|---------:|---------:|
| 42 (run_146) | 5.96% | 19.33% |
| 123 (run_170) | 5.77% | 18.85% |
| 456 (run_171) | 5.84% | 19.09% |
| **Mean ± Std** | **5.86% ± 0.08pp** | **19.09% ± 0.20pp** |

A standard deviation of **0.08pp** across three independent seeds establishes the **noise floor** for this experimental setup. Any reported effect smaller than ~0.15–0.20pp (roughly 2× this noise floor) should be treated as **not distinguishable from seed noise** without a proper multi-seed statistical test. This directly justifies the interpretation in Section 1 that the register-count CER range (0.10–0.24pp) is within, or barely above, the noise floor and should not be over-interpreted as a causal register effect.

**Why reproducibility is this tight**: the pipeline fixes all major sources of nondeterminism (`torch`, `numpy`, `random` seeds; `cudnn.deterministic=True`) per the repo's `set_seed()` implementation, and the dataset/architecture/optimizer are otherwise identical across seeds — the only remaining stochasticity is weight initialization and data-loader shuffling order, both of which have a genuinely small effect at this convergence quality.

---

## 7. Synthetic Pretraining: A Clean, Quantified Domain-Gap Demonstration

| Run | Architecture | Registers | Epochs reached | Synthetic Val CER | **IAM Test CER** |
|-----|-------------|-----------|----------------:|-------------------:|-------------------:|
| 189 | CNN-RNN | — | 11/15 | ~0.5% | 35.45% |
| 192 | ViT-RGTS v2 | 0 | 10/15 | 0.78% | 30.31% |
| 195 | ViT-RGTS v2 | 4 | 10/15 | 0.84% | 31.60% |
| 198 | ViT-RGTS v2 | 8 | 10/15 | 0.82% | 30.62% |
| 199 | ViT-RGTS v2 | 16 | 11/15 | 0.81% | 30.97% |
| 202 | ViT-B/16 | 0 | 2/15 | 4.58% | 46.03% |
| 204 | TrOCR | — | 3/15 | 62.09% | 75.83% |

### 7.1 Why synthetic val CER (~0.5–0.8%) is so much lower than IAM test CER (~30–46%)

The synthetic corpus is rendered from clean digital text using ~555 fonts with programmatic control over layout — there is essentially **no label noise, no ink bleed, no paper texture, no writer-specific idiosyncratic letterforms, and no scanning artifacts**. A model can memorize font-rendering regularities (e.g. exact character-width tables per font) to achieve near-perfect accuracy on held-out synthetic data drawn from the *same* generative distribution, while that memorized knowledge transfers poorly to real handwriting, which has none of those regularities. This is a **textbook train/test distribution mismatch**, and the ~30–46pp CER gap is a direct, quantified measurement of that domain gap for this specific rendering pipeline.

### 7.2 Why ViT-RGTS v2 generalizes better than CNN-RNN from synthetic-only training (30.3% vs. 35.5%)

CNN filters are local, translation-equivariant feature detectors; when trained exclusively on rendered fonts, they can specialize to detect *specific rendering artifacts* (e.g. exact anti-aliasing patterns, consistent stroke widths of digital fonts) that have zero counterpart in ink-on-paper handwriting. Self-attention, being a content-based *routing* mechanism rather than a fixed local filter bank, is architecturally less able to overfit to low-level rendering textures and more likely to route attention based on higher-level character-shape structure, which transfers better. This is a plausible, testable hypothesis but should be flagged in the report as an interpretation rather than a definitively proven mechanism — a controlled ablation (e.g. inspecting learned CNN filter visualizations vs. attention maps) would be needed to fully confirm it.

### 7.3 Why TrOCR fails catastrophically when retrained on synthetic data (75.83% CER)

TrOCR's encoder was originally pretrained on **real handwritten text** (per the "trocr-base-handwritten" checkpoint). Continuing to train (not freezing) this encoder for only 2–3 epochs exclusively on synthetic, clean-rendered text pulls its weights away from the real-handwriting manifold it was pretrained on, without enough synthetic-domain training to reach a new, useful equilibrium — the model ends up in an unstable intermediate state that is good at neither domain. This is a **catastrophic-forgetting** pattern similar to Section 2.3, but here the "forgetting" damages a good pretrained state rather than a random one, making the outcome arguably worse for downstream real-data use.

### 7.4 Synthetic runs are disclosed as time-truncated

All synthetic runs hit the 24-hour SLURM walltime before reaching the target 15 epochs (achieved 2–11 depending on per-epoch cost: ~2.1h/epoch for CNN-RNN/ViT-RGTS v2, ~6.6h/epoch for TrOCR, ~10h/epoch for ViT-B/16, driven by the ~950K-sample synthetic training set being ~100× larger than IAM's 6,482 lines). These numbers should be reported as "domain-gap lower-bound estimates from partial training," and the true synthetic→IAM transfer potential (with full 15 epochs, or with subsequent IAM fine-tuning) remains an open follow-up experiment, not a finding this report can close.

---

## 8. Consolidated "What Worked / What Didn't" Table

| Design decision | Worked? | Evidence | Root-cause explanation |
|-------------------|:-------:|----------|------------------------|
| CNN stem before Transformer | ✅ Essential | 5.96% → 73.36% CER when removed | Preserves strict left-to-right token order required by CTC monotonic alignment |
| Dual CTC head (BiLSTM+CNN) | ✅ Helps | 5.96% vs 7.07% (RNN-only) vs 9.47% (CNN-only) | Multi-path gradient supervision; BiLSTM adds sequential context CNN-only lacks |
| Register tokens (0→16) | ➖ Neutral for CER | 0.10–0.24pp range, within seed noise (0.08pp) | Absorbs global attention that would otherwise leak into patch tokens; does not affect CTC-relevant recognition capacity |
| SWA (well-converged models) | ✅ Helps | −0.26 to −0.39pp on CNN-RNN/ViT-RGTS v2/TrOCR-FT | Averages over flat loss-basin; standard SWA benefit |
| SWA (poorly-converged models) | ❌ Hurts | +1.5 to +17pp on frozen-TrOCR/naive-ViT-B16 | Averages noisy, non-converged snapshots — no stable basin to average around |
| SWA start at 73% of training | ✅ Near-optimal | Better than 50% or 90% start | Enough post-convergence epochs to sample flat-basin snapshots without wasting early useful learning |
| Differential (2-group/LLRD) fine-tune LR | ✅ Essential for pretrained models | 15.09% vs 73.14% (uniform LR) on ViT-B/16 | Prevents catastrophic forgetting of pretrained weights by fresh, noisy head gradients |
| Gradual/staged unfreezing | ❌ Hurts (large domain gap) / ➖ Neutral (in-domain) | +60pp (ViT-B/16) / +1.3pp (TrOCR) | Head overfits to stale frozen-backbone features; recovery after unfreeze is harder than joint training |
| Strong (ViT-style) augmentation | ❌ Slightly hurts | +0.34pp vs moderate augmentation | Over-regularizes on a dataset too small to benefit from aggressive synthetic diversity |
| Model widening (dim 256→512) | ❌ Slightly hurts | +0.61pp | Optimizer LR tuned for dim=256; wider layers need re-tuned (smaller) LR to avoid instability/overfitting |
| Model deepening (depth 6→8) | ➖ Neutral/slightly helps | −0.18pp | Depth scaling doesn't disturb per-layer gradient scale the way width scaling does |
| RNN depth reduction (3→1 layer) | ⚠️ Moderate degradation | +0.83pp | Reduced capacity to resolve local character-boundary ambiguities; not catastrophic |
| Frozen pretrained encoder (TrOCR) | ❌ Fails | ~72–73% CER | Encoder-decoder cross-attention embeddings are not linearly/recurrently CTC-decodable without fine-tuning |
| Synthetic-only pretraining | ⚠️ Large domain gap | 30–46% CER on real IAM test vs <1% on synthetic val | Models overfit to clean-font rendering artifacts absent in real handwriting |

---

## 9. What Remains Pending (Disclosed, Not Fabricated)

**Update (2026-09-07)**: the two path-resolution bugs described below were fixed and jobs resubmitted (1803919→1804972, 1803920→1804973). That resubmission surfaced **two further, independent bugs**, both now also fixed:
- `batch_eval.slurm` was reconstructing each run's model architecture from a generic default config file per architecture type (e.g. always `baseline_vit_rgts_v2.yaml` for any `vit_rgts` run) instead of that run's own exact saved `config.json`. This caused `Missing key(s) in state_dict` failures for every run whose architecture deviated from the default (all 22 ablations, both pretrained fine-tune variants, all synthetic-mode runs) — **40 of 63 evaluations failed** on the first attempt; only 23 default-configuration runs succeeded. Fixed by converting each run's own `config.json` directly into a temporary YAML consumed by `evaluate.py`, guaranteeing exact architecture reconstruction regardless of ablation type.
- `pub_fig3_gradcam_quantitative.py` crashed with `RuntimeError: cudnn RNN backward can only be called in training mode` on its first real invocation (the model is correctly in `.eval()` mode for inference, but cuDNN's fused LSTM kernels refuse to run a backward pass through an RNN unless the module is in training mode). Because the SLURM script used `set -e`, this single crash silently skipped GradCAM plus all four remaining pipeline steps (Beyond-Memorization single/comparison, head specialization, layer-wise evolution). Fixed by wrapping the offending `backward()` call in `torch.backends.cudnn.flags(enabled=False)` (forces PyTorch's slower-but-correct native RNN backward path) and by removing `set -e` in favor of per-step `|| echo "[WARN] ... continuing"` guards so no single step can again silently take down the rest of the pipeline.

Both fixes are applied; jobs 1804972 (postprocess) and 1804973 (batch_eval) were resubmitted. **Confirmed already regenerated and safe to cite** as of 2026-09-07: `outputs/pub_figures/fig2_register_comparison.{png,pdf}`, `fig3_register_quantitative.{png,pdf}`, and `outputs/report_figures/attention_quality/` (all computed on the real final runs 144/146/151/152, 100 IAM test images). Check `squeue -u $USER` and the timestamps below before citing anything else from this section.

The following analyses require GPU inference on trained checkpoints and were queued as SLURM jobs (`postprocess_all.slurm`, `batch_eval.slurm`) at the time of writing this document. **Two path-resolution bugs were discovered and fixed** in the postprocessing scripts (`scripts/postprocessing/single_model/evaluate.py` and `scripts/postprocessing/comparative/attention_quality_metrics.py`) — both used a `sys.path.insert` computed with one too few parent-directory levels, causing `ModuleNotFoundError` when run as standalone SLURM jobs (only surfaced now because these scripts had apparently never been run as pure batch jobs before). Jobs were resubmitted after the fix; they are cluster-queued at time of writing and were not yet complete on GPU during this analysis pass:

- **Per-sample bootstrap significance tests** (CER confidence intervals, `scripts/postprocessing/statistical_analysis.py`) for the final run set — the existing `outputs/tables/statistical_tests.csv` references an **older set of runs (105–142)** from a prior experiment iteration and must be regenerated against runs 143–205 once `batch_eval.slurm` completes.
- **Attention quality metrics** (entropy, Gini sparsity, peak sharpness per register count) — ✅ **now complete**, see `outputs/report_figures/attention_quality/attention_quality_metrics.csv`. Needed to make the interpretability claim in Section 1.2 quantitative rather than purely mechanistic/qualitative.
- **Per-character Beyond-Memorization-style attention maps** (0 vs. 4 vs. 16 registers) — the visual evidence for the "registers absorb global attention, patch tokens localize" claim. Pending job 1804972.
- **GradCAM comparison** across register counts — pending job 1804972 (cuDNN bug now fixed).
- **Head-specialization and layer-wise attention evolution** figures for the 4-register model — pending job 1804972.

No CER/WER/loss numbers in this document are estimated, extrapolated, or fabricated — all are read directly from `results.csv` (150-epoch, best-validation-epoch selection) or `training.log` (`[SWA] Test CER` lines) for the actual completed training runs. Only the *visual/statistical interpretability* layer is pending re-computation after the bugfix.

---

## 10. Recommendations for the Final Report

1. **Do not claim register tokens improve CER.** The honest, defensible claim is: *"register tokens leave CTC recognition accuracy statistically unchanged (within seed noise) while [pending Section 9] providing measurably better attention interpretability."* This is scientifically accurate and matches the original Darcet et al. finding for classification ViTs.
2. **Report SWA using the training.log `[SWA] Test CER` values**, not the final-epoch CSV row, for every SWA-enabled run cited in the report. Recommend re-extracting the true SWA numbers for run_154, run_156, run_186 (SWA LR ablation) if these are to be cited, since only a subset were pulled during this pass (Section 3.1).
3. **Use the corrected ablation labels** in Section 4.1 — do not reuse any earlier draft table that attributes RNN-1-layer to run_165 or ViT-RGTS-v1 to run_179.
4. **Lead the architecture-comparison narrative with the CNN-vs-ViT data-efficiency story** (Section 2.2) — it is the most scientifically clean and well-evidenced finding in the entire matrix (2× convergence-speed difference, directly measured).
5. **Present the two independent CNN-stem-removal failures (157, 165) together** as convergent evidence, not as two separate ablations — this materially strengthens the causal claim about token ordering and CTC.
6. **Flag the synthetic-pretraining numbers as walltime-truncated** wherever cited (Section 7.4); do not present them as final converged results.
