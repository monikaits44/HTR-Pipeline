# HTR‑VT + Registers + Beyond‑Memorization Attention — Feasibility & Architecture Evaluation

**Status:** Evaluation only. No code is implemented in this document.
**Author role:** Senior research scientist / system‑design review.
**Date:** 2026‑06‑21.
**Scope:** Assess combining three published ideas into the existing modular HTR‑Pipeline:
1. **HTR‑VT** (CNN/ResNet + ViT encoder + CTC) — Li et al., *Pattern Recognition* 2025.
2. **Registers** — "Vision Transformers Need Registers", Darcet et al., ICLR 2024.
3. **Beyond‑Memorization (BM) per‑character attention maps** — ICDAR 2025.

The central question the user asked: **will this combination produce *clear* per‑character attention maps, and is it worth building?**

---

## 0. TL;DR — Verdict

| Dimension | Assessment |
|---|---|
| **Engineering feasibility** | **High.** HTR‑VT is ~90% structurally identical to the repo's existing `ViTRGTSBackbone`. Registers already exist in the repo. BM‑style extraction already works (`scripts/register_attention_bm_style.py`). |
| **Data readiness (READ2016)** | **Medium.** Available from the HTR‑VT dataset drive in `lines/ + train.ln/val.ln/test.ln` format. Needs a charset rebuild (German + diacritics + historical glyphs) and a preprocessing adapter. No blocker. |
| **Will registers clean the attention maps?** | **Weak‑to‑moderate, with caveats.** Registers fix *high‑norm artifact tokens* that appear in **large, over‑trained** ViTs. HTR‑VT is **small (depth=4)**, trained on **small data**, with a **strong CNN front‑end** — exactly the regime where artifacts are *least* likely. The repo's own ViT‑RGTS sweep already showed **only 0–7.5% register absorption**, i.e. empirically small effect. |
| **Will we get BM‑style 2‑D blobs?** | **This is the real risk.** HTR‑VT **fully collapses image height to a 1‑D token row**. BM's striking 2‑D character blobs come from **2‑D cross‑attention over a preserved spatial grid**. A 1‑D CTC self‑attention map is inherently a **horizontal strip**, not a 2‑D blob. The biggest lever for "clear attention maps" is **architecture geometry (keep a 2‑D grid)**, *not* registers. |
| **Recommendation** | Build it, but **reframe the hypothesis**: study registers as a *whitespace/blank sink* and run a **controlled register sweep on HTR‑VT**, while **introducing a 2‑D‑grid variant** to actually obtain BM‑style blobs. Treat clean 2‑D maps as an *architectural* outcome, registers as a *secondary* cleanliness factor. |

---

## 1. The Three Components — What Each Actually Is

### 1.1 HTR‑VT (the base model)
From the official code (`model/HTR_VT.py`, `model/resnet18.py`):

- **Front‑end:** a **ResNet‑18 stem** (`ResNet18(embed_dim)`), 1→D channels, that **aggressively reduces height and lightly reduces width** (conv1 stride `(2,1)`, maxpool `(2,1)`, layer1 `(2,1)`, layer2/3 stride `2`, final maxpool `(2,1)`). Output `[B, D, W', H']` is flattened `view(b,c,-1).permute(0,2,1)` → token sequence `[B, L, D]`.
- **Encoder:** plain pre‑norm ViT blocks with **LayerScale** + **DropPath**, `embed_dim=768, depth=4, heads=6, mlp_ratio=4`. Standard softmax attention (`qkv → attn=softmax(QKᵀ/√d) → proj`). *(Note: it defines triangular `forward_bias/back_bias` but does not use them — attention is bidirectional.)*
- **Positional:** **fixed 2‑D sin‑cos** embeddings (not learned).
- **Regularizer (key novelty #1):** **span masking** — `generate_span_mask()` zeroes **contiguous spans** of feature tokens and substitutes a learnable `mask_token` *during training only* (`use_masking=True`). Acts like in‑run masked‑image modelling.
- **Optimizer (key novelty #2):** **SAM** (sharpness‑aware minimization) — flatter minima, better small‑data generalization. Lives in `train.py`, not the model.
- **Head:** `norm → Linear(D, nb_cls) → LayerNorm → CTC`. **No CLS token, no registers.**

**Reported results:** IAM CER 4.7 / WER 14.9; LAM CER 2.8 / WER 7.4; competitive on READ2016. The ablation that *removing the CNN front‑end pushes IAM CER 4.7→26.6* is the field's clearest proof that **a conv stem is mandatory for ViT‑on‑small‑data**.

### 1.2 Registers (attention‑cleaning mechanism)
Darcet et al. observed that large self‑supervised ViTs (DINOv2, CLIP) spawn a few **high‑norm "artifact" tokens** in low‑information background patches that hold *global* information and **corrupt local attention maps**. Adding **N learnable register tokens** gives the model a dedicated scratchpad → artifacts vanish → **smoother, more interpretable attention**.
**Crucial context:** the effect was demonstrated on **large models, large data, CLS‑token objectives**.

### 1.3 Beyond‑Memorization per‑character attention
BM produces **per‑character heatmaps** by reading **cross‑attention** from a diffusion UNet (character‑conditioned), overlaying with `inferno`, `alpha=0.5`, on **64×256 word crops**. The visual signature: a **tight 2‑D blob over the rendered glyph**. The repo has reproduced this (job 1709209) *and* adapted the *style* to the HTR self‑attention setting in `scripts/register_attention_bm_style.py` (CTC‑aligned per‑character 1‑D strips, register sweep R=0..16).

---

## 2. Current Repo Alignment (why this is mostly assembly, not invention)

| HTR‑VT element | Already in repo? | Where |
|---|---|---|
| CNN front‑end + ViT + CTC | ✅ Equivalent | `ViTRGTSBackbone` (4‑layer CNN stem) + `CTCtopR/B` heads in `models.py` |
| ResNet‑18 stem (deeper) | ⚠️ Different stem | repo uses a 4‑conv stem; HTR‑VT uses ResNet‑18 |
| Registers | ✅ Implemented | `ViTRGTSBackbone.register_tokens`, sweep 0/2/4/8/16 |
| Per‑layer attention export | ✅ Implemented | `forward_explain()` → `attn_maps: List[L][B,H,S,S]` |
| BM‑style per‑char overlay | ✅ Implemented | `scripts/register_attention_bm_style.py` |
| Fixed sin‑cos pos‑embed | ❌ | repo uses **learned** pos‑embed |
| Span masking | ❌ | not present |
| SAM optimizer | ❌ | repo uses AdamW + cosine |
| LayerScale / DropPath | ❌ | not present |

**Implication:** the work is **~70% configuration + adapter glue** on top of existing modular infrastructure, **~30% genuinely new** (ResNet‑18 stem option, span‑mask regularizer, SAM optimizer, READ2016 data path). This fits the repo's modular contract (new `configs/htrvt_*.yaml`, a new backbone branch in `HTRNet`, a new SLURM script) **without disturbing existing architectures**.

---

## 3. Proposed Unified Architecture (design, not code)

```
            ┌──────────────────────────────────────────────────────────┐
            │  Input line image  [B, 1, H, W]  (READ2016, inverted)     │
            └──────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┴─────────────────────────┐
          │  Front‑end (configurable)                          │
          │   • HTR‑VT mode: ResNet‑18 stem                    │
          │   • repo mode  : 4‑conv stem (existing)            │
          └─────────────────────────┬─────────────────────────┘
                                    │  feature map
            ┌───────────────────────┴───────────────────────┐
            │  Geometry switch (THE key design choice)       │
            │   (A) collapse height → 1‑D token row  (HTR‑VT)│
            │   (B) keep Hp×Wp 2‑D grid             (BM‑ready)│
            └───────────────────────┬───────────────────────┘
                                    │  tokens [B, L, D]
                       ┌────────────┴────────────┐
                       │  Prepend R register tokens │  ← Registers (0/2/4/8/16)
                       └────────────┬────────────┘
                                    │  [B, R+L, D]
                       ┌────────────┴────────────┐
                       │  ViT encoder (LayerScale) │  + span‑mask (train only)
                       │  export per‑layer attn    │  ← forward_explain
                       └────────────┬────────────┘
                                    │
                   ┌────────────────┴────────────────┐
                   │  CTC head (Linear or BiLSTM)     │
                   └────────────────┬────────────────┘
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │  BM‑style per‑character attention extraction            │
        │   token_idx = R + t_c (CTC peak) → attn over patch span │
        │   overlay inferno @ alpha=0.5, register‑sweep grid      │
        └────────────────────────────────────────────────────────┘
```

The **geometry switch (A/B)** is the most consequential decision and is discussed in §6.

---

## 4. Architectural Changes Required (modular, additive)

All changes are **additive** and gated by config, preserving existing behaviour:

1. **New backbone branch** in `HTRNet` (`models.py`): `arch.type == "htrvt"`.
   - Add a `ResNet18Stem` module (port of `model/resnet18.py`); reuse the existing register‑token + pos‑embed + encoder plumbing of `ViTRGTSBackbone`.
   - Add a `pos_embed_type` flag: `learned` (existing) vs `sincos2d` (HTR‑VT).
   - Add **optional `LayerScale`** to the encoder layers (small, isolated).
2. **Span‑mask regularizer** as a train‑only module (port `generate_span_mask` / `random_masking`). Gated by `arch.span_mask: {ratio, max_span_length, enabled}`. Trivial to make a no‑op at eval.
3. **SAM optimizer** as a `trainer.py` option (`optimizer: sam`). Wrap the existing param‑group logic; SAM needs a closure (two forward/backward passes). This is the only training‑loop change.
4. **`forward_explain` reuse** — already returns `attn_maps`; if geometry mode (B) is used, reshape patch‑token attention back to `Hp×Wp` for genuine 2‑D overlays.
5. **New configs**: `configs/htrvt_read2016.yaml` (+ register variants `htrvt_read2016_r{0,2,4,8,16}.yaml`).
6. **READ2016 data adapter** (see §5).
7. **New SLURM**: `experiments_execution/slurm_modular/htrvt_read2016.slurm` (mirror the existing modular jobs; rtx3080/gpu:1; absolute venv path — a known repo gotcha).
8. **BM extraction**: extend `scripts/register_attention_bm_style.py` to accept `htrvt` runs (it already does CTC‑aligned per‑char + register sweep). If geometry (B), enable 2‑D blob rendering analogous to the BM reference.

**No existing architecture, config, or script is modified destructively.**

---

## 5. READ2016 Data Readiness

- **Source:** HTR‑VT dataset drive; READ2016 ships in the same layout HTR‑VT expects: `data/read2016/{train.ln,val.ln,test.ln}` + `lines/<id>.png` + `lines/<id>.txt`.
- **Why READ2016 over IAM/LAM here:** it is the **smallest/most self‑contained** of the three, the paper reports it as a *secondary* benchmark (so reproduction targets are modest and known), and it stresses the **register hypothesis harder** (historical German script → more background clutter, diacritics, faded ink → *more* potential artifact tokens, i.e. a fairer test of whether registers help).
- **Preprocessing adapter (compatible with repo):** reuse `utils/preprocessing.py` (`1‑img/255` invert → aspect‑ratio resize → pad to fixed `H×W`). READ2016 lines are longer/variable → confirm `max_seq_len` and width (the repo uses 128×1024; HTR‑VT trains at its own size). Keep a per‑dataset `image_size` in config.
- **Charset rebuild (required):** READ2016 has a **larger alphabet** (umlauts ä/ö/ü/ß, punctuation, historical glyphs). Must regenerate `classes.npy` / index maps from READ2016 transcripts → `nb_cls` changes. This is the **only data task with real substance**; everything else is mechanical.
- **Risk:** none blocking. Effort is in charset + a `read2016_dataset.py` loader mirroring `utils/htr_dataset.py`.

---

## 6. Critical Intuition — Will the Attention Maps Be *Clear*? (the core question)

This is where senior judgement matters more than engineering. There are **two independent questions** that are easy to conflate:

### Q1. Will **registers** make HTR‑VT's attention cleaner?
**Likely only marginally. Here is the honest reasoning:**

- **Registers target a failure mode of *large, over‑trained, CLS‑objective* ViTs.** The artifact tokens Darcet et al. fix emerge with scale (ViT‑L/g, DINOv2, long pre‑training). **HTR‑VT is the opposite regime:** depth=4, small HTR datasets, CTC (no CLS), strong conv stem. The conditions that *create* artifacts are largely absent.
- **Empirical prior from *this very repo*:** the ViT‑RGTS register sweep measured **register absorption of just 0.000 (R=0) → 0.075 (R=16)** with entropy ~constant (~4.0). That is **direct local evidence** that registers absorb little in the CTC‑HTR setting. Expect a similar small effect on HTR‑VT.
- **Where registers *could* genuinely help:** as a **sink for non‑informative columns** — inter‑word whitespace and CTC‑blank positions currently scatter low‑magnitude attention. A register can soak up "global/background" mass, slightly **sharpening per‑character peaks** and reducing the classic **attention‑sink to first/last tokens**. This is a *plausible, testable, modest* benefit — not a transformation.
- **Net:** treat registers as a **secondary cleanliness knob**, and **measure** it (entropy, Gini, peak‑sharpness, register‑absorption, char‑localization accuracy — all already in `utils/attention_metrics.py`) rather than assume it.

### Q2. Will we get BM‑style **2‑D character blobs** at all?
**This is the bigger risk, and it is architectural, not register‑related.**

- BM's beautiful tight blobs come from **2‑D cross‑attention over a preserved spatial grid** (UNet, character‑query conditioned). The map answers *"which 2‑D image region does character c attend to?"*.
- **HTR‑VT, as published, fully collapses image height** to a **1‑D token row** (ResNet stem + height pooling). Self‑attention over a 1‑D row can only yield a **horizontal strip / 1‑D profile**, never a 2‑D blob. You lose vertical localization by construction.
- **Self‑attention ≠ cross‑attention.** Even along the horizontal axis, self‑attention tells you *token→token* affinity, not *character→pixels*. The repo already bridges this with **CTC peak alignment** (`token_idx = R + t_c`), which works but is **indirect** and noisier than true cross‑attention.
- **Therefore the dominant lever for "clear maps" is the geometry switch (§3, A vs B):**
  - **Mode (A) — full height collapse (faithful HTR‑VT):** best recognition, but attention maps are inherently **1‑D strips** (what `register_attention_bm_style.py` already produces). Registers give marginal sharpening.
  - **Mode (B) — keep a shallow `Hp×Wp` 2‑D grid** (e.g. pool height to 2–4 rows instead of 1): unlocks **genuine 2‑D attention** that can be reshaped to the image and overlaid like BM, at a likely small CER cost and higher token count. **This, not registers, is what will make maps look like BM.**

### Combined intuition (the one‑paragraph answer)
> Registers alone will **not** deliver the dramatic clean maps the BM figures show, because (a) HTR‑VT is in the small‑model/CTC regime where register artifacts barely form, and (b) the model destroys vertical structure, so its self‑attention is 1‑D by design. The path to **clear, BM‑like** maps is **architectural**: retain a thin 2‑D token grid (mode B) and/or add a **light character‑query cross‑attention read‑out head** so attention is *character→region* rather than *token→token*. Registers should be included as a **controlled, measured secondary variable** (whitespace/blank sink, sink‑token suppression) — valuable for a clean ablation story, but not the primary mechanism. **Expectation: modest, measurable improvement in attention‑sharpness metrics from registers; large, visible improvement from the 2‑D‑grid / cross‑attention geometry.**

---

## 7. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Registers show negligible effect (as in ViT‑RGTS) | High | Medium (weakens narrative) | Pre‑register the hypothesis as "whitespace/blank sink"; report quantitative metrics either way — a *null result is still a result*. |
| 1‑D maps look nothing like BM blobs | High (mode A) | High (user wants BM look) | Add **mode B** (2‑D grid) and/or cross‑attention read‑out head as a parallel variant. |
| SAM doubles training cost (2 forward/backward) | Certain | Low/Medium | Budget GPU‑time; SAM is optional — ablate SAM on/off. |
| READ2016 charset / longer lines break `max_seq_len` | Medium | Medium | Rebuild charset; raise `max_seq_len`/width per‑dataset in config. |
| Span‑mask + small data instability | Low | Low | It is a *regularizer* helping small data; keep `enabled` flag, tune `mask_ratio`. |
| Fidelity drift vs published HTR‑VT (learned vs sincos pos, BiLSTM head) | Medium | Low | Provide a "faithful" preset (sincos, linear head) and a "repo" preset; compare. |

---

## 8. Recommended Phased Plan (when implementation is approved)

1. **Phase 0 — Data:** READ2016 download + charset rebuild + `read2016_dataset.py` + EDA sanity (CER‑decodable transcripts).
2. **Phase 1 — Faithful HTR‑VT baseline (mode A, R=0, no registers):** reproduce a READ2016 number in the right ballpark. Establishes correctness before adding variables.
3. **Phase 2 — Register sweep on HTR‑VT (mode A, R=0/2/4/8/16):** measure attention metrics; test the *whitespace‑sink* hypothesis. **This directly answers Q1.**
4. **Phase 3 — Geometry variant (mode B, 2‑D grid):** the experiment that actually yields BM‑style 2‑D blobs; optional light cross‑attention read‑out head. **This answers Q2.**
5. **Phase 4 — Ablations:** SAM on/off, span‑mask on/off, learned vs sincos pos‑embed.
6. **Phase 5 — BM‑style visual deliverables:** extend `scripts/register_attention_bm_style.py` for `htrvt` runs + 2‑D overlays + register‑sweep comparison grids; quantitative attention tables.

Each phase is a self‑contained SLURM job in the existing modular style; nothing is monolithic.

---

## 9. Decision Matrix (go / no‑go per goal)

| Your goal | Recommended config | Confidence |
|---|---|---|
| **Best READ2016 recognition** | Faithful HTR‑VT, mode A, SAM+span‑mask, R=0 | High |
| **Clean BM‑style 2‑D attention maps** | Mode **B** (2‑D grid) ± cross‑attn head; registers secondary | Medium‑High |
| **Scientific story: "do registers clean HTR attention?"** | HTR‑VT mode A, controlled R sweep + metrics | High (even if effect is small) |
| **Lowest engineering effort, reuse everything** | Keep repo's 4‑conv stem + existing register sweep, just add READ2016 + SAM/span‑mask | High |

---

## 10. Bottom Line

- **Feasible and well‑aligned** with the modular repo — mostly assembly plus a focused data task.
- **Registers are worth including but should be framed and measured as a secondary cleanliness factor**, not the hero. The repo's own evidence predicts a small effect in this regime.
- **The decisive factor for "clear attention maps" is geometry**: keep a 2‑D token grid (or add character‑query cross‑attention) so the maps are *character→region* and *2‑D*. That, not registers, reproduces the BM look.
- **Recommended:** build it in phases, with **mode A (faithful) for accuracy** and **mode B (2‑D) for attention clarity**, and let the **quantitative attention metrics adjudicate the register question** instead of assuming the outcome.
```
