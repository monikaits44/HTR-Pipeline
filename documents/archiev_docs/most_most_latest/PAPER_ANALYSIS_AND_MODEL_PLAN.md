# Paper Analysis: Models, Architectures & Recommendations for HTR with ViT + Register Tokens

> **Purpose**: Detailed analysis of all papers in the `/papers` folder, focusing on ViT-based architectures for HTR, register tokens, attention-guided explainability, and a finalized plan for model experimentation.

---

## Table of Contents

1. [Paper-by-Paper Analysis](#1-paper-by-paper-analysis)
2. [Deep Dive: Four Priority Papers](#2-deep-dive-four-priority-papers)
3. [ViT-Based Models for HTR: What the Literature Shows](#3-vit-based-models-for-htr)
4. [Register Tokens: Complete Understanding](#4-register-tokens-complete-understanding)
5. [Recommendations & Finalized Model Plan](#5-recommendations--finalized-model-plan)

---

## 1. Paper-by-Paper Analysis

### Paper 0: "Advancements and Challenges in Handwritten Text Recognition" (AlKendi et al., 2024)

**Type**: Survey (30 pages)

**Models Covered**:
| Architecture | Examples | Level |
|-------------|----------|-------|
| CNN | AHCR-DLS (2-CNN) | Character |
| CRNN + CTC | CRNN-MDLSTM, CNN-BGRU | Line |
| Transformer | Transformer-T with Cross-Attention | Character/Subword |
| Attention-Gated CNN-BGRU | For Kazakh script | Character |
| End-to-end | OrigamiNet (page-level) | Page |
| CNN backbone + Transformer decoder | (Retsinas et al.) | Line |

**Key Findings for Our Work**:
- Survey confirms that **CRNN + CTC** remains the dominant paradigm for line-level HTR
- Transformer-based approaches are gaining ground but typically require **more data** than available in IAM alone
- The paper highlights that **data augmentation** and **transfer learning** are critical for transformer success on limited data
- No ViT-specific models are surveyed in depth — this is a gap our work addresses

---

### Paper 1: "Handwritten Text Recognition: A Survey" (Garrido-Munoz et al., 2024)

**Type**: Comprehensive survey (20 pages)

**Key Architecture Timeline** (from the paper's Figure 2):
```
2006: CTC (Graves et al.)
2017: Transformer (Vaswani) → CRNN+CTC for HTR (Puigcerver)
2020: OrigamiNet, Recurrence-Free HTR (Coquenet)
2021: VAN, TrOCR, Transformer for HTR (Kang et al.)
2022: DAN (Document Attention Network)
2023: DTrOCR (Fujitake)
```

**Models Specifically Mentioned**:
| Model | Architecture | Approach |
|-------|-------------|----------|
| Puigcerver (2017) | CNN + BiLSTM + CTC | The baseline CRNN approach |
| TrOCR (Li et al., 2023) | ViT encoder + Transformer decoder | Pretrained, autoregressive |
| DTrOCR (Fujitake, 2023) | Decoder-only Transformer | GPT-style, image tokens → text |
| VAN (Coquenet, 2021) | Vertical Attention Network | Attention-based, paragraph-level |
| DAN (Coquenet, 2022) | Document Attention Network | Full-page, no segmentation |

**Critical Insight**:
> "A CNN backbone with a Transformer encoder, a CTC-based decoder, plus an explicit language model, is the most effective strategy to date for line-level transcriptions." — Diaz et al. (cited in paper)

This directly validates our **ViT-RGTS v2** approach (CNN stem + Transformer + CTC).

**On ViT for HTR**:
- The paper notes growing interest in "scalable and parallelizable architectures such as the Transformer by adapting the Vision Transformer to the HTR field"
- Transformer-based HTR benefits from both **CTC** and **encoder-decoder** configurations
- **Large labeled corpora or pretraining** is increasingly seen as necessary for transformer-based models

---

### Paper 5: "Best Practices for a Handwritten Text Recognition System" (Retsinas et al., 2024)

**Type**: Methodology paper — **the basis of our codebase**

**Architecture**: CNN (ResNet-style) + BiLSTM + CTC

**Key Contributions** (all implemented in our pipeline):
1. **Aspect-ratio preserving preprocessing**: Resize to 128×1024, pad with median
2. **Column-wise MaxPool**: Collapses height to 1, producing left-to-right sequence
3. **CTC shortcut**: Auxiliary 1D conv branch trained with CTC loss alongside the BiLSTM

**Architecture Diagram**:
```
Input [128×1024]
  → Conv 7×7 (32ch)
  → ResBlock×2 (64ch) → MaxPool 2×2
  → ResBlock×4 (128ch) → MaxPool 2×2
  → ResBlock×4 (256ch) → MaxPool 2×2
  → ColumnMaxPool → [128×256] feature sequence
  → BiLSTM (256 hidden) × 3 layers
  → Linear → CTC predictions [128 × nclasses]
  + CTC Shortcut: Conv1D → parallel CTC predictions
```

**Reported Results**:
- **IAM line-level**: 4.6% CER (paper), we achieved **5.02%** (run_73)
- **RIMES**: 2.2% CER

**Relevance**: This is our CNN-RNN baseline. The ViT-RGTS v2 replaces the ResNet backbone with a CNN stem + Transformer while keeping the same CTC head architecture.

---

### Paper 6: "TrOCR: Transformer-based Optical Character Recognition" (Li et al., 2023)

**Type**: Architecture paper

**Architecture**: **Full Transformer** (encoder-decoder), pre-trained

```
Image → ViT Encoder (DeiT/BEiT) → Transformer Decoder → Autoregressive text output
```

**Key Design**:
- **Encoder**: BEiT-base or DeiT-base (ViT-B/16, 86M params)
- **Decoder**: Standard Transformer decoder with cross-attention
- **Pre-training**: Stage 1 on large synthetic data (684M printed + handwritten), Stage 2 on task-specific data
- **No CTC**: Uses autoregressive decoding with cross-entropy loss
- **Input**: 384×384 images (fixed square)

**Results**:
- IAM: **4.22% CER** (with pre-training on massive data)
- Without pre-training: performance degrades severely

**Relevance to Our Work**:
- We use only the **TrOCR encoder** with a CTC head (not the full encoder-decoder)
- Our best TrOCR result: 15.47% CER (run_98) — far from the paper's 4.22%
- Gap is due to: (a) we don't use the decoder, (b) we don't pre-train on 684M samples, (c) our CTC adaptation (height-pool + repeat-interleave) loses spatial information
- **Recommendation**: TrOCR is designed for encoder-decoder; forcing CTC on it is suboptimal

---

## 2. Deep Dive: Four Priority Papers

---

### Paper 4: "Vision Transformers Need Registers" (Darcet et al., ICLR 2024)

**The foundational paper for our register token approach.**

#### The Problem: Attention Map Artifacts

The paper identifies that **all modern ViTs** (except original DINO) exhibit artifacts in attention maps:
- **High-norm outlier tokens**: ~2% of patch tokens have ~10× higher L2 norm
- They appear in **low-informative background regions** (patches similar to neighbors)
- They emerge during training of **large models** (≥ViT-L) after sufficient epochs
- They appear around the **middle layers** of the transformer

#### What Outlier Tokens Actually Do

| Property | Normal Tokens | Outlier Tokens |
|----------|--------------|----------------|
| Position prediction accuracy | 41.7% | 22.8% |
| Pixel reconstruction error | 18.38 | 25.23 |
| ImageNet classification (linear probe) | 65.8% | **69.0%** |

**Key insight**: Outlier tokens **discard local spatial information** and instead **store global image-level information**. The model repurposes redundant patches as internal computation buffers.

#### The Solution: Register Tokens

```
Input: [CLS] [REG₁] [REG₂] ... [REGₙ] [patch₁] [patch₂] ... [patchₘ]
                 ↑ learnable tokens, output discarded
```

- Add N learnable tokens to the input sequence
- These tokens are used by the model as **dedicated memory/computation slots**
- At output, register tokens are **discarded** — only CLS and patch tokens used
- This **isolates** the global-information-storage behavior away from patch tokens

#### Key Experimental Findings

**Artifacts disappear with registers**:
- Adding ≥1 register removes all norm outliers
- Feature maps become significantly smoother
- Attention maps become interpretable (like original DINO)

**Performance impact** (registers vs no registers):

| Model | ImageNet Top-1 | ADE20k mIoU | NYUd rmse ↓ |
|-------|---------------|-------------|-------------|
| DeiT-III | 84.7 → 84.7 | 38.9 → 39.1 | 0.511 → 0.512 |
| DINOv2 | 84.3 → **84.8** | 46.6 → **47.9** | 0.378 → **0.366** |

**Register count ablation** (DINOv2 ViT-L):
- 1 register: removes artifacts
- 4 registers: optimal for dense tasks
- 4-16 registers: ImageNet accuracy continues improving slightly
- **The paper uses 4 registers in all final experiments**

**Emergent behavior**: Different registers attend to different parts of the scene (similar to slot attention), without any explicit training signal for this.

#### Relevance to Our HTR Work

Our ViT-RGTS v2 implements registers but in a **fundamentally different setting**:

| Aspect | Paper (Darcet et al.) | Our ViT-RGTS v2 |
|--------|----------------------|-----------------|
| Model size | ViT-B to ViT-g (86M-1B params) | ~6M params |
| Training data | ImageNet-22k (14M images) | IAM (6.5K lines) |
| Training method | DINOv2 self-supervised | Supervised CTC |
| Pretraining | Yes (extensive) | No (from scratch) |
| Register output | Discarded | **Kept for analysis** |
| Main goal | Fix artifacts, improve dense tasks | **Explainability + writer style capture** |

**Key question for our work**: Do artifacts even appear in our small model trained on small data? The paper shows artifacts emerge in "sufficiently large models after sufficient training." Our 6M-param model on 6.5K samples may be too small for this phenomenon. This is worth investigating explicitly.

---

### Paper 2: "Interpretable Writer Recognition via VLAC" (Raven et al., 2024)

**The most directly relevant paper for using ViT features for writer identification and character-level analysis.**

#### Architecture

The paper uses **two parallel models**:

**Model 1 — HTR (ΘHTᵣ): Decoder-free Transformer**
```
Input: text line [512 × 64 px]
  → Modified ResNet-18 (patch embedding, stride-adjusted)
  → 128 tokens, each corresponding to a 4×64 pixel vertical slice
  → 4-layer Transformer encoder (dim=384)
  → Linear → 80 classes (alphabet + blank)
  → CTC loss
```
- **No decoder**: Single forward pass produces entire character sequence
- Each output token maps to a specific image region (4×64 px)
- CER: 4.9% on IAM

**Model 2 — Feature Extractor (ΘF): Self-supervised ViT**
```
Input: text line [512 × 64 px]
  → ViT-small with patch size 4×64
  → 128 local feature tokens (matching ΘHTᵣ)
  → Trained with AttMask (self-supervised, like DINO)
  → Final patch tokens used as local features (CLS discarded)
```

#### VLAC: Vectors of Locally Aggregated Characters

The key innovation — **aggregate features by character**:

1. ΘHTᵣ predicts which character each token represents (via CTC)
2. ΘF extracts a feature vector for each token
3. One-to-one correspondence: token j → character prediction c_{i,j} AND feature x_{i,j}
4. Group features by character: all tokens predicted as 'a' → group_a
5. Compute per-character global prototype μ_c (mean across all documents)
6. VLAC encoding: Φ_{v,c} = ℓ₂(Σ (x - μ_c)) — residuals from prototype, normalized
7. Document distance = average of character-wise cosine distances

**Results**:
| Method | IAM Top-1 | IAM mAP | CVL Top-1 | CVL mAP |
|--------|-----------|---------|-----------|---------|
| Sum-pooling | 96.2 | 96.1 | 97.7 | 91.4 |
| VLAD | 97.4 | 97.4 | 98.3 | 93.5 |
| **VLAC (theirs)** | **98.1** | **97.8** | **99.3** | **96.3** |

#### Direct Relevance to Our Register Token Work

This paper provides the **methodological framework** for using our register tokens:

1. **Character-level attention** → Our attention maps from `forward_explain()` can similarly map tokens to characters
2. **Writer-specific features via ViT** → Our register tokens potentially capture similar writer-level information
3. **Self-supervised ViT training (AttMask/DINO)** → Could replace or complement our supervised CTC training
4. **Decoder-free transformer for HTR** → Their ΘHTᵣ is architecturally similar to our ViT-RGTS v2 (CNN stem → transformer → CTC), validating our approach
5. **Patch size 4×64** → They use tall, narrow patches matching a full text line height, similar in spirit to our CNN stem's column tokens

**Key takeaway**: The paper shows that **ViT patch tokens carry character-level local information AND writer-level global information simultaneously**. Register tokens could serve as dedicated global information stores, making the patch tokens carry purer local (character) information — which is exactly the hypothesis from Paper 4.

---

### Paper 3: "Beyond Memorization: Training-Free Style Mixing" (2025)

**Extends WordStylist (Paper 7) with attention-guided character-level writer style injection.**

#### Architecture: Modified WordStylist (Latent Diffusion)

```
U-Net (noise predictor) with:
  ├── Residual Blocks (spatial feature processing)
  ├── Spatial Transformer Blocks (cross-attention: image features ↔ text embedding)
  ├── Writer embedding injection (element-wise into ResBlocks)
  └── Attention map extraction (from middle block)
```

This is NOT an HTR model — it's a **handwriting generation** model. But its techniques for attention-based character localization are directly relevant.

#### Key Technique: Character-Level Attention Maps

The paper extracts attention maps A from the U-Net's cross-attention:
- Shape: `[B, L, H, W]` where L=10 (max characters)
- Each A_c gives spatial localization for character c
- Used to determine where each character is in the generated image
- These maps guide **per-character writer style injection**

**Style Mixing Process**:
1. Extract attention maps A_c from middle U-Net block
2. Compute max X-coordinate for each character → transition boundary
3. Apply binary mask: characters before boundary get writer W_r, after get W_o
4. Generate image with mixed styles

**Results**: 
- 176K generated variations from 44K IAM words (4 character positions × 44K)
- HWD (Handwriting Distance) metric confirms smooth per-character style transitions
- Variability increases monotonically with number of characters modified

#### Relevance to Our Work

1. **Attention maps for character localization**: Their cross-attention maps A_c demonstrate that transformers learn spatial character positions. Our self-attention maps from ViT-RGTS v2 carry similar information — the register-to-patch attention shows where each register "looks."

2. **Writer embedding as global information**: Their writer embedding W is explicitly injected. In our model, **register tokens implicitly learn to capture the same information** — the parallel is striking.

3. **Per-character analysis methodology**: Their HWD per-character computation framework can be adapted to analyze our register token effects: how does varying register count affect character-level vs global representations?

4. **Attention-guided visualization**: The attention map visualization technique (Fig. 5, 6) is exactly what we should produce for our register analysis — showing how register attention shifts across layers and how it correlates with character positions.

---

### Paper 7: "WordStylist: Styled Verbatim Handwritten Text Generation with LDM" (Nikolaidou et al., 2023)

**The base model that Paper 3 extends.**

#### Architecture: Conditional Latent Diffusion Model

```
Training:
  Input word image → VAE Encoder (VE) → latent z
  z + noise → noisy z_t
  z_t → U-Net(z_t, timestep_t, style_Y, text_cτ) → predicted noise
  Loss: ||ε - ε_θ||²

Sampling:
  Random noise z_T → U-Net denoising (T=600 steps) → z_0
  z_0 → VAE Decoder (VD) → synthetic word image
```

**Key Components**:
- **VAE**: Pretrained stable-diffusion VAE (Hugging Face)
- **U-Net**: 1 ResNet block per module (small for scarce data), dim=320, 4 attention heads
- **Style conditioning**: Embedding layer for writer class index → added to timestep embedding
- **Text conditioning**: Character tokenization → embedding → positional encoding → self-attention → cross-attention with U-Net features

**HTR Evaluation** (using our Best Practices architecture):

| Training Data | CER (%) | WER (%) |
|--------------|---------|---------|
| Real IAM | 4.86 | 14.11 |
| WordStylist synthetic only | 8.80 | 21.93 |
| **Real IAM + WordStylist** | **4.67** | **13.28** |
| Real IAM + GANwriting | 4.87 | 13.88 |
| Real IAM + SmartPatch | 4.83 | 13.90 |

**Key findings**:
- Synthetic data from WordStylist **improves** HTR when combined with real data (4.86 → 4.67% CER)
- Writer retrieval: mAP 97.84% (WordStylist) vs 97.61% (real IAM) — nearly indistinguishable styles
- The HTR model used for evaluation is **exactly our CNN+BiLSTM+CTC architecture** (Paper 5)

#### Relevance to Our Work

1. **Data augmentation potential**: WordStylist can generate synthetic training data that improves HTR by ~0.2% CER. This could help our ViT-RGTS v2 model, which suffers from limited training data.

2. **Style embedding ≈ register tokens**: The writer class embedding in WordStylist captures global style. Our register tokens learn similar information — the key question is whether register tokens can be used for writer identification or style control.

3. **IAM word-level, Aachen split**: WordStylist uses the same data split as our pipeline, making direct comparison possible.

---

## 3. ViT-Based Models for HTR: What the Literature Shows

### Taxonomy of ViT Approaches for HTR

Based on all papers, ViT-based HTR models fall into these categories:

| Category | Example | Encoder | Decoder | Training Loss | Data Needs |
|----------|---------|---------|---------|---------------|------------|
| **CNN-RNN + CTC** | Best Practices (Paper 5) | CNN (ResNet) | BiLSTM | CTC | Low (6.5K) |
| **CNN-Transformer + CTC** | Our ViT-RGTS v2 | CNN stem + Transformer | BiLSTM | CTC | Low-Medium |
| **ViT Encoder + CTC** | Paper 2 (ΘHTᵣ) | ResNet-18 → Transformer | Linear (CTC) | CTC | Low (6.5K) |
| **Pretrained ViT + CTC** | Our TorchVision/TrOCR | ViT-B/16 or TrOCR enc | BiLSTM | CTC | High (needs PT) |
| **Full Transformer** | TrOCR (Paper 6) | BEiT/DeiT | Transformer dec | CE (autoregressive) | Very High (684M) |
| **Self-supervised ViT** | Paper 2 (ΘF, AttMask) | ViT-small | None (features) | Self-supervised | Medium |

### What Works for HTR at IAM Scale (~6.5K samples)

1. **CNN + RNN + CTC** remains king at small scale (4.6-5.0% CER)
2. **CNN-Transformer hybrids + CTC** are competitive (5.99-6.05% CER) and provide explainability
3. **Pretrained models without massive pre-training data** underperform significantly
4. **Self-supervised ViT** (AttMask/DINO) provides excellent features for downstream tasks (writer identification) without task-specific labels
5. **Decoder-free Transformers + CTC** (Paper 2) achieve 4.9% CER on IAM — very close to CNN-RNN

---

## 4. Register Tokens: Complete Understanding

### What Register Tokens Do (from Paper 4)

```
Without registers:
  Patch tokens = local info + FORCED to also store global info
  → Some patches become "outliers" (high-norm, lose local info)
  → Attention maps become noisy/artifactual

With registers:
  Register tokens = dedicated global info storage
  Patch tokens = pure local spatial info
  → No outliers
  → Clean attention maps
  → Better dense prediction
```

### Register Tokens in the HTR Context

In handwriting recognition, the "global information" stored by register tokens likely includes:
- **Writer style**: Stroke thickness, slant, spacing patterns
- **Line-level statistics**: Average character width, baseline position
- **Script/language features**: Character frequency distributions
- **Ink/background characteristics**: Contrast, noise level

The "local information" in patch tokens should be:
- **Character identity**: What character this patch represents
- **Character position**: Where in the sequence this character appears
- **Local stroke details**: Specific curves, connections, serifs

### What We've Observed So Far

From our experiments (runs 60-92):
- **Register count (0-16) has minimal effect on CER** (all within 4.08-4.45%)
- This suggests registers **don't hurt** recognition but also don't significantly help
- **Attention maps** from models with vs without registers have not been systematically compared yet
- **Register token embeddings** are saved but not yet analyzed for writer information content

### What Needs Investigation

1. Do attention artifacts (high-norm outlier tokens) even appear in our small 6M-param model?
2. Do attention maps become cleaner with registers (as in Paper 4)?
3. Do register token embeddings cluster by writer (like Paper 2's VLAC)?
4. Can register tokens be used for writer identification (like Paper 7's style embeddings)?

---

## 5. Recommendations & Finalized Model Plan

### Priority Assessment

Given the project goals:
1. **Primary**: Observe effect of register token variation on attention map visualization in ViT
2. **Secondary**: Achieve competitive HTR performance
3. **Tertiary**: Enable writer identification / style analysis

### Models NOT Worth Implementing Further

| Model | Reason to Skip |
|-------|---------------|
| Full TrOCR (enc-dec) | Needs 684M pre-training samples; autoregressive decoding incompatible with CTC pipeline |
| TorchVision ViT-B/16 with CTC | 86M params on 6.5K samples = massive overfitting; runs 97-101 confirm poor results (28-86% CER) |
| ViT-RGTS v1 (raw patches) | Broken by design (row-major flatten destroys CTC order); already proven failure |

### Finalized Model Plan

#### Tier 1: Core Experiments (Must Do)

**1A. Systematic Attention Map Analysis (existing models)**

Use the trained models from runs 60-71 (register sweep, 80 epochs) to:
- Compare attention map quality across register counts (0, 1, 3, 5, 6, 7, 8, 14, 16)
- Measure token norm distributions: do outlier tokens appear?
- Compute attention entropy per layer: does it decrease (more focused) with registers?
- Visualize register-to-patch attention: what spatial regions do registers attend to?
- Compare patch-to-patch attention: is it cleaner with registers?

**Required**: Only analysis scripts, no new training. Use saved attention weights from `attention_weights/epoch_080/`.

**1B. Register Token Embedding Analysis**

Using saved `register_tokens.npy` from runs with different register counts:
- Cluster register embeddings across different input samples
- Check if register embeddings separate by writer
- Compute cosine similarity of register embeddings for same vs different writers
- Compare to Paper 2's VLAC approach

**Required**: Analysis notebook, no training.

---

#### Tier 2: Recommended New Experiments

**2A. Decoder-Free Transformer (inspired by Paper 2)**

Paper 2's ΘHTᵣ achieves 4.9% CER on IAM with a **simpler architecture** than ours:
```
Modified ResNet-18 → 128 tokens (4×64 patches) → 4-layer Transformer (dim=384) → Linear + CTC
```

**Adaptation for our pipeline**:
- Replace our CNN stem with a **modified ResNet-18** as patch embedder
- Use patch size **4×64** (full height, 4px wide) — each token is a vertical strip
- 4-layer Transformer, dim=384
- **Add register tokens** (our contribution on top of their architecture)
- Train with CTC loss (same as current pipeline)

**Why**: This is architecturally validated at 4.9% CER on IAM (Paper 2), close to our CNN-RNN baseline. Adding registers to THIS architecture and comparing attention maps would be the cleanest experiment for our thesis.

**Implementation effort**: Moderate — new backbone class in `models.py`, reuse existing CTC head and training loop.

**Config**:
```yaml
arch:
  type: vit_resnet_patch  # New type
  patch_height: 64        # Full line height
  patch_width: 4          # Narrow vertical strips
  dim: 384
  depth: 4
  heads: 6
  num_registers: 4        # Sweepable
  head_type: both
  rnn_type: lstm
  rnn_layers: 3
  rnn_hidden_size: 256
```

**2B. Self-Supervised ViT Feature Extraction (inspired by Paper 2)**

Train a ViT with **AttMask** (DINO-style self-supervised) on IAM text lines, then:
- Use frozen features for writer identification
- Compare feature quality with vs without registers
- Apply VLAC encoding for interpretable writer recognition

**Why**: This directly tests whether register tokens capture writer-specific information (the "global information" hypothesis from Paper 4) in a self-supervised setting.

**Implementation effort**: High — requires implementing AttMask training.

---

#### Tier 3: Optional/Future Experiments

**3A. Synthetic Data Pre-training (Paper 7 approach)**

Use our existing synthetic data pipeline to:
1. Pre-train ViT-RGTS v2 on synthetic data (data.mode=synthetic)
2. Fine-tune on IAM
3. Compare CER and attention maps before/after fine-tuning
4. Test if registers capture different information after pre-training vs from-scratch

**Why**: Paper 7 shows synthetic data improves CER by 0.2%. More importantly, pre-training on more data could cause attention artifacts (Paper 4 shows artifacts appear with "sufficient training"), making register effects more visible.

**3B. Character-Level Attention Analysis (Paper 3 approach)**

Adapt Paper 3's attention map analysis to our model:
1. Extract per-token character predictions (from CTC output)
2. Map register attention to character positions
3. Measure how register attention correlates with character identity
4. Compare across register counts: do more registers lead to more character-specific register specialization?

**3C. DINOv2 or MAE Pre-trained ViT with Registers**

Instead of training ViT from scratch, use a **self-supervised pre-trained** ViT:
- DINOv2 ViT-S/14 or ViT-B/14 (smaller than ViT-B/16, fits our memory)
- Add register tokens (or use DINOv2 with registers, which is available)
- Fine-tune CTC head only or with LLRD
- Compare attention maps: pre-trained registers should show cleaner separation than from-scratch

**Why**: Paper 4's register effects are most visible on pre-trained models. Our from-scratch 6M-param model may be too small to exhibit artifacts.

---

### Summary: Prioritized Action Plan

| Priority | Experiment | New Training? | Implementation Effort | Expected Impact |
|----------|-----------|---------------|----------------------|-----------------|
| **P1** | Attention map analysis (existing runs 60-71) | No | Low (scripts only) | HIGH — direct thesis contribution |
| **P1** | Register embedding clustering/writer ID | No | Low (analysis) | HIGH — tests core hypothesis |
| **P2** | Decoder-free ViT (Paper 2 style) + registers | Yes | Medium | HIGH — validated architecture |
| **P2** | Synthetic pre-training → IAM fine-tuning | Yes | Low (pipeline exists) | MEDIUM — may reveal artifact behavior |
| **P3** | Self-supervised ViT (AttMask) + registers | Yes | High | MEDIUM — novel contribution |
| **P3** | Character-level attention analysis | No | Medium (analysis) | MEDIUM — visualization contribution |
| **P3** | DINOv2 pre-trained + registers + CTC | Yes | Medium | HIGH — strongest register effects |

---

### Architecture Comparison Table (Final)

| Model | Params | Training Data | Expected CER | Register Support | Attention Quality | Effort |
|-------|--------|--------------|-------------|-----------------|-------------------|--------|
| CNN-RNN (baseline) | 7.4M | IAM 6.5K | 5.0% | N/A | No attention | Done |
| **ViT-RGTS v2 (current)** | 6M | IAM 6.5K | 6.0% | **Yes (0-16)** | Good | Done |
| Decoder-free ViT (Paper 2) | ~5M | IAM 6.5K | ~5.0% | **Yes (add)** | Excellent | Medium |
| ViT-RGTS v2 + synthetic PT | 6M | Synth + IAM | ~5.5% | **Yes** | Better (more data) | Low |
| DINOv2-S + registers + CTC | 22M | ImageNet PT + IAM FT | ~4-5% | **Yes** | Best | Medium |
| TrOCR enc + CTC | 86M | HF PT + IAM FT | ~15% | No | Poor with CTC | Done |

### Final Recommendation

**For the thesis goal of "observing effect of register token variation on attention maps":**

1. **Start with P1**: Analyze existing trained models — this is the lowest-hanging fruit with highest thesis impact
2. **Then P2**: Implement the Paper 2 decoder-free architecture with registers — this provides a validated baseline to compare against ViT-RGTS v2
3. **If time permits**: DINOv2 pre-trained + registers would give the cleanest demonstration of Paper 4's findings applied to HTR

The key insight from the papers is: **register effects on attention are most visible when the model is large enough and trained long enough for artifacts to emerge**. Our small from-scratch model may not show dramatic differences. To get the strongest results, either (a) pre-train on more data, or (b) use a pre-trained backbone like DINOv2.
