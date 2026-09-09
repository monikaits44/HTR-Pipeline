# Why Our Attention Maps Differ From the Paper — Root Cause Analysis

## Executive Summary

Our attention maps cannot look like "Beyond Memorization" Fig. 5 because **our architecture is fundamentally different**. The paper uses **cross-attention** in a **diffusion U-Net generator**, while we use **self-attention** in a **ViT CTC recognizer**. These are different mechanisms that produce different attention patterns by design. This is NOT a code bug — it's an architectural reality.

---

## 1. The Architecture Gap (Root Cause)

### "Beyond Memorization" Fig. 5 — What the paper does

| Aspect | Paper's Architecture |
|--------|---------------------|
| **Model** | U-Net diffusion model (WordStylist) |
| **Task** | Handwriting **generation** (image → image) |
| **Attention type** | **Cross-attention** in Spatial Transformer blocks |
| **Query** | Image feature map I ∈ ℝ^(B × H × W × D) — **2D spatial** |
| **Key/Value** | Text embedding T ∈ ℝ^(B × L × D) — L=10 character slots |
| **Output shape** | A_c ∈ ℝ^(B × H × W) — a **full 2D spatial map** per character |
| **What it shows** | Where each character **is being generated** in 2D space |
| **Why it's clear** | The model is **explicitly trained** to spatially localize characters for generation; each of the L=10 text slots has a **dedicated** attention distribution over H×W spatial positions |

**Key quote from paper (Section 3.2):**
> "Cross-attention mechanisms are employed in Spatial Transformer (ST) block to enable interactions between the predefined textual embedding T and the image feature map I. Specifically, I is used as queries, while T acts as keys and values."

The attention maps have shape `(B, L, H, W)` where:
- L = 10 maximum character positions (fixed, padded)
- H × W = 2D spatial resolution of the U-Net middle block

Each `A_c` is a **2D heatmap** showing which spatial pixels correspond to character c.

### Our ViT-RGTS v2 — What we have

| Aspect | Our Architecture |
|--------|-----------------|
| **Model** | ViT-RGTS v2 (CNN stem → TransformerEncoder → CTC head) |
| **Task** | Handwriting **recognition** (image → text) |
| **Attention type** | **Self-attention** among patch tokens |
| **Query/Key/Value** | All are patch tokens ∈ ℝ^(B × S × D) — **same tokens** |
| **Output shape** | A[t, :] ∈ ℝ^(Wp) — a **1D vector** (Hp=1, so one row of tokens) |
| **What it shows** | Which other patch tokens does token t attend to |
| **Why it's diffuse** | Self-attention distributes information broadly; the model needs global context (spacing, word structure) to recognize characters, not just local character pixels |

**Critical architectural facts:**
- CNN stem collapses height: 128×1024 → 1×128 tokens. **Hp = 1**.
- Each token is a **column** spanning the full 128px height
- Attention is inherently **1D** (128 positions along width)
- No text embedding exists — there is no dedicated "character c" query
- We approximate character attention by reading self-attention row at CTC timestep t_c

---

## 2. Why Self-Attention ≠ Cross-Attention for Character Maps

### Cross-Attention (paper's U-Net)
```
Score = softmax(Q_spatial · K_text^T / √d)
       ↓
A_c[h, w] = how much pixel (h,w) attends to character c
```
- **Designed for localization**: The text embedding explicitly encodes "character c"
- **2D spatial**: Each character gets a height × width attention map
- **One-to-one**: Character slot c → spatial region for character c
- **Training signal**: The diffusion loss forces the model to generate character c at the right location

### Self-Attention (our ViT)
```
Score = softmax(Q_patch · K_patch^T / √d)
       ↓
A[t, j] = how much token t attends to token j
```
- **Not designed for localization**: No text embedding, no character slots
- **1D**: Hp=1, so attention is over 128 column positions
- **Many-to-many**: Every token attends to every other token
- **Training signal**: CTC loss requires correct output sequence, NOT spatial alignment
- **Our approximation**: Use row A[R+t_c, R:R+Wp] as "character c's attention" — this shows where timestep t_c looks, but it's a **proxy**, not a purpose-built localization

### Why our maps look "diffuse"

1. **Self-attention must be broad**: Token t_c needs context from neighboring tokens (spacing, adjacent characters) to decide what character to output. Clean single-peak attention would mean the model ignores context — which would hurt recognition.

2. **CTC doesn't enforce spatial alignment**: CTC loss only cares that the output character sequence is correct. Multiple timesteps can output the same character (repeated + collapse). The model is free to "look anywhere" as long as it gets the sequence right.

3. **1D tiling creates visual uniformity**: Since Hp=1, our "2D" attention maps are just a 1D signal tiled vertically 128 times. The paper's maps are genuinely 2D with natural spatial structure.

4. **Head averaging dilutes patterns**: Individual heads may specialize (some attend locally, some globally). Averaging 8 heads produces a blurry composite.

---

## 3. The Register Paper (Darcet et al., 2024) — What They Actually Show

The "Vision Transformers Need Registers" paper does **not** show character-level attention maps. Their key visualizations are:

| Figure | What It Shows | Visualization Type |
|--------|--------------|-------------------|
| **Fig. 1** | Artifact tokens disappear with registers | [CLS] → patch attention map (2D, 14×14 grid) |
| **Fig. 7** | Token norm distribution shifts | Histogram of output token L2 norms |
| **Fig. 8** | Artifacts vs. number of registers | Qualitative attention maps + downstream metrics |
| **Fig. 9** | Register tokens attend to different objects | Individual register → patch attention maps |

**Why their attention maps are clear:**
- They use **ViT on natural images** with 14×14 or 16×16 patch grids → **2D spatial structure**
- They show **CLS → patch** attention (or register → patch), which is a global summary
- Their models (DINOv2, DeiT-III) are trained on **ImageNet with objects** — attention naturally highlights objects
- They do NOT show character-level or text-level decomposition

**Our situation:**
- We have **1×128 patch grid** (1D, not 2D) → no spatial richness
- We have **no CLS token** (CTC uses all patch tokens as output)
- Our task is **text line recognition** — no distinct objects to segment
- Register effect in our case is primarily artifact absorption, not object discovery

---

## 4. What Our Visualizations Can Legitimately Show

### What works (and is valid)

| Visualization | Status | Validity |
|--------------|--------|----------|
| **Token norm artifacts** (Fig 3) | ✅ Working | Directly comparable to Darcet et al. Fig. 7. Shows register tokens absorb outlier norms |
| **Global attention heatmap** (Fig 1) | ✅ Working | Shows overall attention distribution shift with registers. Comparable to Darcet Fig. 1 |
| **Attention entropy/localization metrics** | ✅ Working | Quantitative evidence of register effect (1.1% entropy reduction, 20% Xmax improvement) |
| **Character attention approximation** (Fig 2) | ⚠️ Working but inherently limited | CTC self-attention proxy ≠ paper's cross-attention. Shows approximate spatial focus, not precise character localization |
| **GradCAM** (Fig 5) | ✅ Working | Gradient attribution — independent of attention mechanism, always valid |

### What we CANNOT replicate

| Paper Figure | Why Not |
|-------------|---------|
| "Beyond Memorization" Fig. 5 (crisp 2D char blobs) | We don't have cross-attention. Our model has no text embedding. Self-attention is inherently different |
| Darcet Fig. 9 (register attending to objects) | Our images are text lines, not multi-object scenes. Registers absorb global/padding info, not "objects" |
| Any 2D spatial attention map | Our CNN stem collapses Hp=1, so all maps are fundamentally 1D |

---

## 5. Honest Assessment — Is This a Problem?

**No.** The inability to replicate Fig. 5 from "Beyond Memorization" is expected and scientifically correct. Here's why:

### For the supervisor: Key talking points

1. **Different architecture = different attention**. The paper uses a diffusion U-Net with cross-attention between text and image. We use a ViT encoder with self-attention + CTC. These produce fundamentally different attention patterns. Attempting to force our maps to look like theirs would be scientifically dishonest.

2. **Our visualizations are appropriate for our architecture**:
   - Token norm analysis → directly from Darcet et al. methodology
   - CTC self-attention row as character proxy → valid approximation, acknowledged as proxy
   - Entropy/localization metrics → quantitative, reproducible
   - GradCAM → architecture-independent attribution

3. **The register effect IS visible**, just not as a "character blob" change:
   - Entropy: 5.992 → 5.925 (1.1% reduction with 8 registers)
   - Xmax ordering: ρ = 0.169 → 0.204 (20% improvement with registers)
   - Token norms: outlier artifacts absorbed by registers
   - CER: 6.06% → 5.93% (marginal improvement with 16 registers)

4. **The modest effect size is itself a finding**: Register tokens in CTC-based ViTs have a smaller effect than in ImageNet classification ViTs because:
   - Text lines have less spatial redundancy than natural images (fewer "uninformative background" patches)
   - CTC's sequence-to-sequence loss doesn't create the same attention artifact pattern as classification
   - The CNN stem already provides good local feature extraction, reducing the transformer's need for long-range attention

---

## 6. Questions to Ask Your Supervisor

### Architecture & Methodology Questions

1. **"Given that our ViT-RGTS uses CTC self-attention (not cross-attention like the diffusion U-Net in 'Beyond Memorization'), should we focus on a different type of visualization that better suits our encoder-only architecture?"**
   - *Context*: The paper's Fig. 5 uses cross-attention from a generative model where text embedding is explicitly aligned to spatial positions. Our model has no such mechanism.

2. **"The CNN stem collapses spatial height to Hp=1, making our attention maps inherently 1D (128 column tokens). Should we consider an alternative architecture (e.g., 2D patch grid without CNN stem, or adding a cross-attention decoder) to get richer 2D attention maps?"**
   - *Trade-off*: 2D patches would give spatially richer attention but the CNN stem is crucial for CTC performance (it provides local feature extraction + height invariance).

3. **"For the register token analysis, should we follow Darcet et al.'s visualization protocol (token norm histograms, CLS→patch attention, object discovery metrics) rather than trying to replicate character-level attention from a different architecture?"**
   - *Context*: Darcet et al. never show character-level maps — they show artifact removal and downstream task improvement.

### Experimental Direction Questions

4. **"The register effect on attention metrics is modest (~1% entropy, ~20% Xmax improvement). Is this sufficient for publication, or should we run on the full IAM test set for statistical significance (p-values, confidence intervals)?"**
   - *Current*: 5 images. Recommended: full test set (>1000 images) for robust statistics.

5. **"Would an encoder-decoder architecture (e.g., Transformer decoder with cross-attention to ViT encoder) be worth investigating? This would give us true cross-attention character maps comparable to the paper's Fig. 5."**
   - *Implication*: Major architecture change. Would produce clean character attention but changes the recognition model fundamentally.

6. **"Should we compare per-head attention maps instead of head-averaged? Individual heads may show clearer spatial patterns (one head for local, another for long-range context)."**
   - *Current limitation*: We average 8 heads → diffuse pattern. Individual heads might show specialist behavior.

7. **"For the register paper comparison, would it be more impactful to show object/character discovery metrics (like Darcet's LOST experiment) rather than attention map quality?"**
   - *Idea*: Use register/non-register features for text line segmentation or character segmentation as a downstream task, similar to how Darcet uses object discovery.

### Scope & Reporting Questions

8. **"What level of 'attention map clarity' is expected for a CTC-based recognition model? Should we explicitly state in the paper/report that CTC self-attention maps are inherently diffuse compared to cross-attention maps from generative models?"**

9. **"Should I focus the explainability contribution on (a) replicating specific paper figures, or (b) providing appropriate analysis for our architecture (token norms, entropy, GradCAM)?"**

10. **"Given the small effect of registers on recognition accuracy (6.06% → 5.93% CER), should the narrative focus on interpretability benefits (cleaner attention) rather than performance gains?"**

---

## 7. Summary Comparison Table

| Dimension | "Beyond Memorization" Fig. 5 | Our ViT-RGTS v2 | Darcet et al. (Registers) |
|-----------|------------------------------|-----------------|--------------------------|
| **Architecture** | U-Net diffusion | CNN stem + ViT encoder | ViT (DINOv2/DeiT-III) |
| **Task** | Handwriting generation | Handwriting recognition | Image classification |
| **Attention type** | Cross-attention (text × image) | Self-attention (patch × patch) | Self-attention (patch × patch) |
| **Spatial dims** | 2D (H × W) | 1D (1 × 128) | 2D (14 × 14 or 16 × 16) |
| **Character decomposition** | Native (L=10 text slots) | Proxy (CTC timestep row) | N/A (no characters) |
| **Map clarity** | Very clear 2D blobs | Diffuse 1D profiles | Clear 2D patterns |
| **Register effect** | N/A | Modest (1% entropy, 20% Xmax) | Strong (artifacts → clean) |
| **Why clear/diffuse** | Model trained to localize chars | Model trained to sequence chars | Model trained to classify objects |

---

## 8. Recommended Reporting Strategy

### Do Report:
1. **Token norm analysis** — artifact absorption (directly from Darcet methodology)
2. **Quantitative attention metrics** — entropy, localization, Xmax correlation
3. **GradCAM** — architecture-independent, always valid
4. **CER comparison** — the bottom-line recognition metric
5. **Honest characterization** — "CTC self-attention provides approximate spatial attribution; we acknowledge this is not equivalent to cross-attention character localization"

### Do NOT:
1. Claim our attention maps are equivalent to cross-attention maps
2. Force cosmetic processing (excessive gamma, artificial sharpening) to make maps "look like" the paper
3. Compare our 1D attention profiles directly to the paper's 2D attention maps without acknowledging the difference

### The narrative:
> "Register tokens in our CTC-based ViT serve as attention sinks that absorb global patterns, producing cleaner patch-token representations. This architectural hygiene effect manifests as reduced attention entropy (1.1%), improved spatial ordering (20%), and reduced token norm artifacts — consistent with Darcet et al.'s findings on classification ViTs. The register effect is moderate because CTC text lines have less spatial redundancy than natural images, and the CNN stem already provides effective local feature extraction."
