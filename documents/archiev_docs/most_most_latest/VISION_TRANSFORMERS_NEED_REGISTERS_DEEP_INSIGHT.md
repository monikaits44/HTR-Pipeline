# Vision Transformers Need Registers — Complete Research Insight

> **Paper**: Darcet, Oquab, Mairal, Bojanowski — FAIR, Meta & Inria (ICLR 2024)  
> **Purpose**: A senior-researcher-level breakdown of every claim, model, experiment, token mechanism, and design decision in this paper, structured from fundamental concepts to advanced implementation details.

---

## Table of Contents

1. [The Core Problem in One Sentence](#1-the-core-problem)
2. [Background: How a Standard ViT Works](#2-background-standard-vit)
3. [The Artifact Discovery — What Goes Wrong](#3-the-artifact-discovery)
4. [The Three Token Types: CLS, Patch, Register](#4-the-three-token-types)
5. [Exactly Where and How Registers Are Placed](#5-exactly-where-registers-are-placed)
6. [Which Models the Authors Trained](#6-which-models-the-authors-trained)
7. [How Many Registers — The Ablation Study](#7-how-many-registers)
8. [What Registers Learn — Emergent Behavior](#8-what-registers-learn)
9. [Full Experimental Results](#9-full-experimental-results)
10. [Appendix Insights Most People Miss](#10-appendix-insights)
11. [Mapping to Our HTR Pipeline Implementation](#11-mapping-to-our-implementation)
12. [Summary of Key Equations and Mechanisms](#12-summary-equations)
13. [Differentiating CLS and Register Tokens — Position Proof & Practical Approaches](#13-differentiating-cls-and-register-tokens)

---

## 1. The Core Problem

**One sentence**: Modern Vision Transformers silently **hijack** a small number of patch tokens in low-information regions (background), turning them into high-norm outliers that store global image-level information — destroying their local spatial content and creating artifacts in attention maps. Adding dedicated "register" tokens gives the model an explicit place to do this, keeping patch tokens clean.

---

## 2. Background: How a Standard ViT Works

Before understanding registers, you must understand the three processing stages of a standard ViT:

### Stage 1: Patch Embedding

An image is divided into non-overlapping patches (e.g., 16×16 pixels). Each patch is linearly projected into a D-dimensional embedding:

```
Image [H × W × 3]
  → Split into P patches of size (p × p)   where P = (H/p) × (W/p)
  → Linear projection: each patch → D-dimensional vector
  → Result: P patch tokens, each [D]
```

For a 224×224 image with 16×16 patches: P = 14 × 14 = **196 patch tokens**.

### Stage 2: Add Special Tokens + Position Embeddings

```
[CLS]    ← 1 learnable D-dim vector (for classification)
patch_1  ← from image
patch_2  ← from image
...
patch_P  ← from image

Total sequence: S = 1 + P tokens  (e.g., 197 for ViT-B/16 on 224×224)
```

Each token gets a **learned positional embedding** added: `token_i = token_i + pos_embed_i`

### Stage 3: Transformer Encoder

The full sequence passes through L transformer layers. Each layer:

```
for each layer l:
    1. LayerNorm → Multi-Head Self-Attention → Residual connection
    2. LayerNorm → FFN (MLP) → Residual connection
```

**Self-attention** is where every token attends to every other token:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

This means **every token can read from and write to every other token** — including CLS ↔ patches, patch ↔ patch, etc.

### What comes out:

- **CLS token output**: Used for image-level tasks (classification)
- **Patch token outputs**: Used for dense/spatial tasks (segmentation, detection, depth)
- **Attention maps**: The softmax weights showing which tokens attended to which — used for visualization and interpretability

---

## 3. The Artifact Discovery

### 3.1 What the authors found

When they visualized the attention maps (CLS → patches) of modern ViTs, they saw **peaky outlier spots** — small bright dots in the attention map that didn't correspond to semantically meaningful image regions. These appeared in:

| Model | Training | Artifacts? |
|-------|----------|-----------|
| DINO ViT-B/16 | Self-supervised (distillation) | **No** ← the exception |
| DINOv2 ViT-g/14 | Self-supervised (improved) | **Yes** |
| DeiT-III ViT-B | Supervised (ImageNet-22k labels) | **Yes** |
| DeiT-III ViT-L | Supervised (ImageNet-22k labels) | **Yes** |
| OpenCLIP ViT-B | Text-supervised (CLIP-style) | **Yes** |
| OpenCLIP ViT-L | Text-supervised (CLIP-style) | **Yes** |

**DINO is the only exception.** All other modern ViTs exhibit this behavior. The paper's goal is to understand why and fix it.

### 3.2 How artifacts are detected quantitatively

The authors discovered a simple, reliable metric: **the L2 norm of output token embeddings**.

```
For DINOv2 ViT-g/14:
  - Normal patch tokens: norm ≈ 0–100
  - Artifact patch tokens: norm ≈ 200–600  (roughly 10× higher)
  - Proportion of outliers: ~2.37% of all patch tokens
```

The norm distribution is **bimodal** (two clear peaks), making it trivial to separate:
- **Normal tokens**: norm < 150
- **Outlier tokens**: norm ≥ 150

(The cutoff of 150 is hand-picked for DINOv2-g; it varies by model.)

### 3.3 When and where outliers emerge

The paper provides three crucial characterizations (Figure 4):

**(a) Across layers** (DINOv2 ViT-g, 40 layers):
- Layers 1–15: All tokens have similar norms
- **Layer ~15 (middle)**: Outlier tokens start differentiating
- Layers 15–40: Outliers grow progressively higher in norm

**(b) Across training iterations**:
- First 1/3 of training: No outliers
- After ~1/3 of training: Outliers begin appearing
- They persist for the rest of training

**(c) Across model sizes** (DINOv2):

| Model Size | Parameters | Outliers? |
|-----------|-----------|-----------|
| ViT-Tiny | ~5M | No |
| ViT-Small | ~22M | No |
| ViT-Base | ~86M | No |
| **ViT-Large** | **~300M** | **Yes** |
| ViT-Huge | ~630M | Yes |
| ViT-giant | ~1.1B | Yes |

**Key finding**: Artifacts only appear in models ≥ ViT-Large. Smaller models don't have enough capacity to develop this behavior.

### 3.4 Where outliers appear spatially

Outlier patches appear in image regions that are **redundant** — where a patch is very similar to its neighbors (e.g., uniform background, sky, wall).

**Evidence**: The cosine similarity between an outlier patch and its 4 spatial neighbors (measured at the input embedding layer, before any transformer processing) is significantly higher than for normal patches. The model identifies patches carrying little unique information and repurposes them.

Spatially, outliers tend to appear **near borders** rather than the center (Figure 10 in appendix). This makes sense because ImageNet images are mostly object-centric — borders are more likely to be background.

### 3.5 What information outlier tokens carry

The authors train linear probes on top of individual token embeddings:

**Local information probing** (do tokens know where they are / what pixels they contain?):

| Task | Normal Tokens | Outlier Tokens |
|------|--------------|---------------|
| Position prediction (top-1 acc) | 41.7% | **22.8%** ← lost positional info |
| Pixel reconstruction (L2 error ↓) | 18.38 | **25.23** ← lost pixel info |

Outliers have **discarded** their local spatial content.

**Global information probing** (can a single token classify the whole image?):

| Dataset | Normal Token | Outlier Token | CLS Token |
|---------|-------------|---------------|-----------|
| ImageNet-1k | 65.8% | **69.0%** | 86.0% |
| Aircraft | 17.1% | **79.1%** | 87.3% |
| Caltech-101 | 73.2% | **97.6%** | 96.9% |
| Cars | 10.8% | **85.2%** | 91.5% |
| DTD | 63.1% | **84.9%** | 85.2% |
| Flowers | 59.5% | **99.6%** | 99.7% |
| Pets | 47.8% | **94.1%** | 96.9% |

**Outlier tokens perform almost as well as CLS for classification.** They have become "secondary CLS tokens" — carrying global information about the entire image, at the cost of destroying their local patch information.

### 3.6 The paper's interpretation

> "The model learns to recognize redundant tokens, and recycle the corresponding tokens to aggregate global image information while discarding spatial information."

This is a **natural computational mechanism**: the self-attention mechanism needs intermediate storage for global aggregation, and it uses low-information patches as scratch space. This is not a bug in the training — it's an emergent optimization. But it has the **side effect** of corrupting those patch token representations, which hurts dense prediction tasks and makes attention maps noisy.

---

## 4. The Three Token Types: CLS, Patch, Register

After the proposed fix, a ViT with registers has **three distinct token types** in its sequence:

### 4.1 [CLS] Token

| Property | Detail |
|----------|--------|
| **Count** | 1 |
| **Initialization** | Learnable parameter (random init) |
| **Position** | First in the sequence (index 0) |
| **Input information** | None (content-free, same for every image) |
| **Positional embedding** | Gets its own positional embedding |
| **Participates in attention** | Yes — attends to and is attended by all tokens |
| **Output used?** | **Yes** — used for classification and global representation |
| **Role** | Aggregates global image information through attention |

### 4.2 Patch Tokens

| Property | Detail |
|----------|--------|
| **Count** | P = (H/p) × (W/p). E.g., 196 for 224×224 with 16×16 patches |
| **Initialization** | Linear projection of actual image pixels |
| **Position** | After CLS (and after registers if present) |
| **Input information** | Real image content — pixels from a specific spatial region |
| **Positional embedding** | Gets position-specific embedding (encoding spatial location) |
| **Participates in attention** | Yes — attends to and is attended by all tokens |
| **Output used?** | **Yes** — used for dense tasks (segmentation, depth, detection) |
| **Role** | Encode local spatial and visual information about their image region |

### 4.3 [REG] Register Tokens ← The paper's contribution

| Property | Detail |
|----------|--------|
| **Count** | N (hyperparameter; paper uses N=4 in most experiments) |
| **Initialization** | Learnable parameters (random init, like CLS) |
| **Position** | After CLS, before patch tokens |
| **Input information** | **None** — content-free, identical across all images |
| **Positional embedding** | They receive positional embeddings (same as other tokens) |
| **Participates in attention** | **Yes** — they attend to all tokens and are attended by all tokens |
| **Output used?** | **No** — explicitly discarded after the transformer. Never used for any loss or downstream task |
| **Role** | Provide dedicated "scratch space" for the model to store/process global information without corrupting patch tokens |

### 4.4 Critical distinction: Register vs CLS

Both are learnable, content-free tokens. The key difference:

| Aspect | CLS | Register |
|--------|-----|----------|
| Output used for training | **Yes** (classification loss) | **No** (discarded) |
| Output used for inference | **Yes** (image representation) | **No** (discarded) |
| Training signal | Receives direct gradient from task loss | Only indirect gradient (through attention interaction) |
| Number | Always 1 | Hyperparameter (typically 4) |
| Purpose | Explicit output interface | Internal computation buffer |

**Registers have NO output purpose.** They exist purely as computation buffers. The model learns to use them during training because they reduce the need to corrupt patch tokens, which **does** receive gradient signal through the training objective.

---

## 5. Exactly Where and How Registers Are Placed

### 5.1 Sequence construction (Figure 6 in the paper)

The input sequence to the transformer is constructed as:

```
Position:  [0]     [1]     [2]     ...  [N]      [N+1]    [N+2]   ...  [N+P]
Token:     [CLS]   [REG₁]  [REG₂]  ...  [REGₙ]   patch₁   patch₂  ...  patchₚ
                    ↑ learnable, output discarded ↑         ↑ from image ↑
```

**Total sequence length**: S = 1 + N + P

For example, with ViT-B/16, 224×224 image, 4 registers:
- S = 1 + 4 + 196 = **201** tokens

### 5.2 Step-by-step forward pass

```python
# STEP 1: Patch embedding
patches = linear_projection(image)           # [B, P, D]  e.g., [B, 196, 768]

# STEP 2: Prepare special tokens
cls_token = learnable_parameter               # [1, 1, D]  → expand to [B, 1, D]
reg_tokens = learnable_parameters             # [1, N, D]  → expand to [B, N, D]

# STEP 3: Concatenate into full sequence
tokens = concat([cls_token, reg_tokens, patches], dim=1)
#   Shape: [B, 1+N+P, D]  e.g., [B, 201, 768]

# STEP 4: Add positional embeddings
tokens = tokens + positional_embeddings[:, :S, :]

# STEP 5: Pass through transformer encoder
for layer in transformer_layers:
    tokens = layer(tokens)                    # Self-attention among ALL tokens
# Output shape: [B, 1+N+P, D]

# STEP 6: Split output tokens
cls_output = tokens[:, 0, :]                  # [B, D]     — USED for classification
reg_output = tokens[:, 1:1+N, :]              # [B, N, D]  — DISCARDED
patch_output = tokens[:, 1+N:, :]             # [B, P, D]  — USED for dense tasks
```

### 5.3 What "discarded" means precisely

At the **output** of the transformer:
- `cls_output` → fed to classification head / used as image representation
- `patch_output` → used for segmentation, depth estimation, object discovery
- `reg_output` → **thrown away**. Not used in any loss function. Not used for any downstream task.

But during the **forward pass**, registers participate fully in self-attention:
- Other tokens can attend **to** registers (reading information from them)
- Registers can attend **to** other tokens (reading information from patches/CLS)
- This two-way interaction is how registers serve as computation buffers

### 5.4 Where the gradient comes from

Since registers are discarded at the output, how do they learn?

**Through the attention mechanism.** In each transformer layer:

1. When a patch token or CLS token attends to a register, the register's value contributes to the patch/CLS output via the attention-weighted sum.
2. The training loss (on CLS or patch tokens) backpropagates through the attention weights, which include contributions from registers.
3. This gradient flows back to the register tokens' learnable values and to the transformer weights that process them.

The registers learn to hold information that **helps the patch and CLS tokens** compute better outputs — that's their incentive.

---

## 6. Which Models the Authors Trained

### 6.1 Three training paradigms

The authors deliberately tested across **three fundamentally different** training methods to prove generality:

| Training Method | Type | Data | Model Size | Key Reference |
|----------------|------|------|-----------|---------------|
| **DeiT-III** | Label-supervised (classification) | ImageNet-22k | ViT-B/16 | Touvron et al., 2022 |
| **OpenCLIP** | Text-supervised (contrastive) | Shutterstock (text-image pairs) | ViT-B/16 | Ilharco et al., 2021 |
| **DINOv2** | Self-supervised (distillation) | ImageNet-22k | ViT-L/14 | Oquab et al., 2023 |

### 6.2 What "with registers" means in each case

For each method, they trained **two versions**:
- **Without registers**: Standard architecture (CLS + patches)
- **With registers**: Modified architecture (CLS + 4 registers + patches)

**Everything else was kept identical**: same hyperparameters, same data, same training schedule. The only difference is the additional 4 register tokens.

### 6.3 Detailed model configurations

**DeiT-III (supervised)**:
- Architecture: ViT-B/16 (12 layers, 768-dim, 12 heads)
- Data: ImageNet-22k (14M images, 21,841 classes)
- Training: Standard DeiT-III recipe (label supervision, CE loss)
- Patch count: 196 (224×224 input, 16×16 patches)
- With registers: 4 learnable register tokens added to sequence

**OpenCLIP (text-supervised)**:
- Architecture: ViT-B/16 image encoder (12 layers, 768-dim, 12 heads)
- Data: Shutterstock licensed image-text corpus
- Training: CLIP contrastive loss (align image and text embeddings)
- Patch count: 196
- With registers: 4 learnable register tokens added to sequence

**DINOv2 (self-supervised)**:
- Architecture: ViT-L/14 (24 layers, 1024-dim, 16 heads)
- Data: ImageNet-22k (14M images)
- Training: Self-distillation (student-teacher, no labels)
- Patch count: 256 (224×224 input, 14×14 patches)
- With registers: 4 learnable register tokens added to sequence
- **Also trained with 0, 1, 2, 4, 8, 16 registers** for ablation

### 6.4 Pre-existing models analyzed (not trained by the authors)

Additionally, the paper **analyzed but did not train** these models to demonstrate the artifact phenomenon:
- DINO ViT-B/16 (the artifact-free exception)
- DINOv2 ViT-g/14 (the original pre-trained model, 40 layers, 1.1B params)
- DeiT-III ViT-L
- OpenCLIP ViT-L
- MAE ViT-L (Masked Autoencoder — also artifact-free, discussed in Appendix E)

---

## 7. How Many Registers — The Ablation Study

The authors trained **6 DINOv2 ViT-L/14 models** varying only the register count:

### 7.1 Artifact removal

| Registers | Artifacts in attention maps? |
|-----------|---------------------------|
| 0 | Yes — visible outlier spots |
| 1 | **No** — artifacts completely gone |
| 2 | No |
| 4 | No |
| 8 | No |
| 16 | No |

**Just 1 register is enough to remove all artifacts.** The model redirects all global-info-storage behavior to that single register instead of corrupting patch tokens.

### 7.2 Downstream performance

| Registers | ImageNet Top-1 | Avg Segmentation mIoU | Avg Depth rmse ↓ |
|-----------|---------------|----------------------|-----------------|
| 0 | 84.35 | 66.05 | 2.83 |
| 1 | 84.40 | 66.65 | 2.77 |
| 2 | 84.50 | **66.80** | **2.73** |
| 4 | 84.60 | 66.70 | 2.75 |
| 8 | 84.65 | 66.50 | 2.76 |
| 16 | **84.80** | 66.30 | 2.80 |

**Observations**:
- **Classification**: Monotonically improves with more registers (more computation budget)
- **Dense tasks (segmentation, depth)**: Peak around 2–4 registers, then slightly decline
- **Sweet spot**: The paper uses **4 registers** in all final experiments as a balanced choice

### 7.3 Computational cost

| Registers | FLOP increase | Parameter increase |
|-----------|--------------|-------------------|
| 1 | ~0.5% | negligible |
| 4 | **<2%** | negligible |
| 8 | ~3.5% | negligible |
| 16 | ~6% | negligible |

The cost is negligible because:
- Registers add only N×D learnable parameters (e.g., 4×768 = 3,072 params for ViT-B)
- FLOPs increase scales as O(N×S) in attention (adding N tokens to a sequence of S)
- For S=196 patches, adding 4 tokens is a ~2% increase in attention computation

---

## 8. What Registers Learn — Emergent Behavior

### 8.1 Registers absorb the outlier behavior

From Table 4 (Appendix D), comparing a 0-register and 1-register DINOv2 model on Aircraft classification:

| Token type | Without registers | With 1 register |
|-----------|------------------|-----------------|
| CLS token | 84.6% accuracy | 85.2% accuracy |
| Normal patch | 15.5% | 14.5% |
| Outlier patch | 73.3% | N/A (no outliers exist!) |
| Register | N/A | **71.1%** |

The register token achieves **71.1%** accuracy — nearly identical to the outlier tokens' **73.3%**. This proves the register has absorbed exactly the same global-information-storage role that outlier patches used to fill.

### 8.2 Patch tokens remain unchanged

From Table 5, comparing local information in patch tokens:

| Metric | Without registers (non-outlier patches) | With 4 registers (all patches) |
|--------|--------------------------------------|------------------------------|
| Position prediction (top-1 acc) | 66.3% | 65.8% |
| Pixel reconstruction (L2 error ↓) | 15.9 | 16.0 |

The non-outlier patches carry the **same local information** regardless of whether registers are present. Registers don't change the behavior of normal patches — they only remove the outlier corruption.

### 8.3 Norm distribution shift

From Figure 15 (Appendix D):

**Without registers**: Token norms show a bimodal distribution:
- Peak 1: ~50 (normal tokens, majority)
- Peak 2: ~300–600 (outlier tokens, ~2.37%)

**With 4 registers**: 
- Patch tokens: **unimodal**, all in the 0–100 range (no outliers)
- Register tokens: High-norm (~300–600), similar to where outliers used to be
- Register norms appear **quantized** (discrete values) — the paper leaves investigation of this for future work

### 8.4 Emergent spatial specialization

**Figure 9**: The attention maps of the [CLS] token and selected register tokens (reg0, reg6, reg8, reg12) show:

- [CLS] attends broadly to the main object
- Different registers attend to **different parts** of the scene
- Some registers attend to foreground objects, others to background regions
- This behavior is **never explicitly trained** — it emerges naturally

This is analogous to **slot attention** (Locatello et al., 2020), where learned slots specialize to attend to different objects. The registers develop a similar specialization spontaneously.

**Figure 16** (Appendix D): Average attention maps across many ImageNet images:

| Token | Attention pattern |
|-------|------------------|
| CLS | Centered blob (focuses on central object) |
| reg0 | Centered, slightly broad |
| reg1 | Centered, similar to CLS |
| reg2 | Slightly upper-biased |
| **reg3** | **Border-focused** (attends to image edges) |
| Typical patch | Very localized (small spot around its spatial location) |

Registers produce attention maps with **large spatial support** (global), similar to CLS, and very different from patches (which are local). This confirms registers carry global, not local, information.

---

## 9. Full Experimental Results

### 9.1 Feature quality — Linear probing (Table 2a)

| Model | ImageNet Top-1 | ADE20k mIoU | NYUd rmse ↓ |
|-------|---------------|-------------|-------------|
| DeiT-III | 84.7 | 38.9 | 0.511 |
| DeiT-III + reg | **84.7** | **39.1** | 0.512 |
| OpenCLIP | 78.2 | 26.6 | 0.702 |
| OpenCLIP + reg | 78.1 | **26.7** | **0.661** |
| DINOv2 | 84.3 | 46.6 | 0.378 |
| DINOv2 + reg | **84.8** | **47.9** | **0.366** |

**Conclusion**: Registers never hurt performance, and often improve it (especially for DINOv2 dense tasks).

### 9.2 Object discovery — LOST (Table 3)

| Model | VOC 2007 | VOC 2012 | COCO 20k |
|-------|----------|----------|----------|
| DeiT-III | 11.7 | 13.1 | 10.7 |
| DeiT-III + reg | **27.1** | **32.7** | **25.1** |
| OpenCLIP | 38.8 | 44.3 | 31.0 |
| OpenCLIP + reg | 37.1 | 42.0 | 27.9 |
| DINOv2 | 35.3 | 40.2 | 26.9 |
| DINOv2 + reg | **55.4** | **60.0** | **42.0** |

**Massive improvement** for DeiT-III (+15.4 on VOC07) and DINOv2 (+20.1 on VOC07). Object discovery algorithms rely on clean feature maps, which registers provide.

### 9.3 Zero-shot classification — OpenCLIP (Table 2b)

| Model | ImageNet Top-1 |
|-------|---------------|
| OpenCLIP | 59.9 |
| OpenCLIP + reg | **60.1** |

No degradation. Slight improvement.

---

## 10. Appendix Insights Most People Miss

### 10.1 Why DINO doesn't have artifacts (Appendix E)

MAE (Masked Autoencoder) also doesn't have artifacts. The paper hypothesizes:
- Both DINO (original) and MAE use **local losses** on patch tokens
- DINO uses a local self-distillation objective
- MAE reconstructs individual masked patches
- Neither requires global aggregation in the patch tokens during training

In contrast, DINOv2 uses a **global objective** (iBOT + DINO combined), forcing the model to aggregate global information — which leads to the hijacking of patch tokens.

### 10.2 Interpolation artifacts (Appendix A)

The original DINOv2 codebase had a subtle bug: positional embeddings were interpolated from 16×16 to 7×7 **without antialiasing**, creating a vertical striping pattern in where outliers appeared. The authors fixed this with antialiased interpolation. This is a practical detail that matters for reimplementations.

### 10.3 Artifacts appear in ALL attention heads (Appendix F)

Figure 18 shows per-head attention maps. Artifacts appear in **every head** of the last transformer block, not just specific heads. Some heads show more artifacts than others, but none are clean.

### 10.4 OpenCLIP's value projection filters outliers (Appendix C)

A surprising finding: In OpenCLIP models **without** registers, the outlier patches are visible in the key and query projections, but **not in the value projection**. The outliers live in the null space of the value projection matrix. This means:
- The attention weights (from Q×K^T) show artifacts
- But the actual attended values (Attention × V) are clean
- This explains why LOST with OpenCLIP values works OK even without registers

---

## 11. Mapping to Our HTR Pipeline Implementation

Our `ViTRGTSBackbone` in `models.py` implements register tokens with some key differences from the paper:

### 11.1 Token sequence comparison

**Paper (standard ViT with registers)**:
```
[CLS] [REG₁] [REG₂] [REG₃] [REG₄] [patch₁] [patch₂] ... [patch₁₉₆]
  ↑         ↑                              ↑
  Used    Discarded                       Used
```

**Our ViT-RGTS v2 (HTR pipeline)**:
```
[REG₁] [REG₂] [REG₃] [REG₄] [col_token₁] [col_token₂] ... [col_token₁₂₈]
  ↑                              ↑
  KEPT for analysis             Used for CTC
```

Key differences:

| Aspect | Paper | Our Implementation |
|--------|-------|--------------------|
| CLS token | Present, output used | **Absent** (not needed for CTC) |
| Register output | **Discarded** | **Kept** for analysis (explainability goal) |
| Patch tokens | 2D grid (14×14 or 16×16) | **1D column sequence** (128 tokens from CNN stem) |
| Patch source | Linear projection of raw image patches | **CNN stem** features (height collapsed) |
| Primary task | Classification, dense prediction | **CTC-based text recognition** |

### 11.2 Where registers are positioned in our code

From `models.py`, `ViTRGTSBackbone.forward()`:

```python
# Register tokens expanded to batch size
reg_tokens = self.register_tokens.expand(B, -1, -1)  # [B, R, D]

# Concatenate: registers FIRST, then patches
tokens = torch.cat([reg_tokens, patch_tokens], dim=1)  # [B, R+128, D]

# Positional embeddings for ALL tokens (registers + patches)
pos = self.pos_embed[:, :S, :]
tokens = tokens + pos

# Transformer encoder processes ALL tokens together
encoded = self.encoder(tokens)  # [B, S, D]

# Split at output
reg_out = encoded[:, :self.num_registers, :]    # [B, R, D] ← KEPT
patch_out = encoded[:, self.num_registers:, :]  # [B, 128, D] ← used for CTC
```

### 11.3 Our forward_explain() for attention extraction

Unlike the paper's models (which use standard PyTorch MultiheadAttention without exposing weights), our implementation **manually computes attention** in `forward_explain()`:

```python
# For each transformer layer:
q, k, v = linear(normed, in_proj_weight, in_proj_bias).chunk(3)
q = q.view(B, S, num_heads, head_dim).transpose(1, 2)  # [B, H, S, D_h]
k = k.view(B, S, num_heads, head_dim).transpose(1, 2)
v = v.view(B, S, num_heads, head_dim).transpose(1, 2)

attn_weights = softmax(q @ k.T / sqrt(d_k))  # [B, H, S, S]
attn_maps.append(attn_weights.cpu())          # SAVED for analysis
```

This gives us the full attention matrix where:
- Rows 0..R-1, Columns 0..R-1: **register ↔ register** attention
- Rows 0..R-1, Columns R..S-1: **register → patch** attention (what registers look at)
- Rows R..S-1, Columns 0..R-1: **patch → register** attention (how patches use registers)
- Rows R..S-1, Columns R..S-1: **patch ↔ patch** attention

### 11.4 How the paper's findings apply to our HTR case

| Paper finding | Implication for our HTR pipeline |
|--------------|--------------------------------|
| Artifacts only appear in ≥ViT-L (~300M params) | Our model is ~6M params — artifacts **unlikely** to appear |
| Artifacts need "sufficient training length" | Our 80-epoch training on 6.5K samples may be insufficient |
| Artifacts appear with global training objectives | Our CTC loss is **local** (per-character) — less likely to cause artifacts |
| 1 register removes all artifacts | If artifacts exist, even 1 register should fix them |
| 4 registers is the sweet spot | Our experiments sweep 0–16, consistent with paper |
| Registers develop spatial specialization | In HTR, registers might specialize to different text regions or writer features |
| Registers carry global information | In HTR, this could be writer style, line statistics, or document-level features |

---

## 12. Summary of Key Equations and Mechanisms

### 12.1 Standard self-attention (no registers)

Input sequence: $\mathbf{X} = [\mathbf{x}_{cls}, \mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_P] \in \mathbb{R}^{(1+P) \times D}$

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Every token's output is a weighted sum of **all** tokens' values.

### 12.2 Self-attention with registers

Input sequence: $\mathbf{X} = [\mathbf{x}_{cls}, \mathbf{r}_1, \ldots, \mathbf{r}_N, \mathbf{x}_1, \ldots, \mathbf{x}_P] \in \mathbb{R}^{(1+N+P) \times D}$

The attention computation is **identical** — no architectural change to the transformer layer. The only difference is the input sequence is longer by N tokens.

### 12.3 Output extraction

$$\mathbf{y}_{cls} = \text{output}[0] \quad \leftarrow \text{used}$$
$$\mathbf{y}_{reg_i} = \text{output}[i], \quad i \in \{1, \ldots, N\} \quad \leftarrow \text{discarded}$$
$$\mathbf{y}_{patch_j} = \text{output}[N+j], \quad j \in \{1, \ldots, P\} \quad \leftarrow \text{used}$$

### 12.4 What registers do across layers (conceptual)

```
Layer 1-5 (early):
  Registers: absorb broad context from all patches
  Patches: process local features normally

Layer 6-15 (middle):
  Registers: begin aggregating global image-level statistics
  Patches: read from registers for contextual information
  (Without registers: SOME patches get hijacked here)

Layer 16+ (late):
  Registers: hold refined global information (high norm)
  Patches: remain focused on local spatial details (normal norm)
  CLS: reads from both patches and registers for classification
```

### 12.5 The norm signature

Without registers:
$$||\mathbf{y}_{patch}||_2 = \begin{cases} \sim 50 & \text{normal patch (97.6\%)} \\ \sim 300\text{-}600 & \text{outlier patch (2.4\%)} \end{cases}$$

With registers:
$$||\mathbf{y}_{patch}||_2 \sim 50 \quad \text{(all patches, unimodal)}$$
$$||\mathbf{y}_{reg}||_2 \sim 300\text{-}600 \quad \text{(high norm, quantized)}$$

---

## 13. Differentiating CLS and Register Tokens — Position Proof & Practical Approaches

Both CLS and register tokens are learnable, content-free parameters with no pixel-level image information. This section provides a detailed, step-by-step breakdown of **how their positions are confirmed** in the official codebase, **how they differ structurally**, and **practical methods to distinguish them** at inference and analysis time.

---

### 13.1 How Position "After CLS, Before Patches" Is Confirmed — Step-by-Step from DINOv2 Source

The **official DINOv2 implementation** (`facebookresearch/dinov2`) provides the most authoritative proof. The critical function is `prepare_tokens_with_masks`:

```python
def prepare_tokens_with_masks(self, x, masks=None):
    B, nc, w, h = x.shape
    
    # STEP 1: Patch embedding
    x = self.patch_embed(x)           # [B, P, D] — raw patch tokens
    
    # STEP 2: Prepend CLS token to patches
    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    # NOW: x = [CLS, patch_1, patch_2, ..., patch_P]  shape: [B, 1+P, D]
    
    # STEP 3: Add positional embeddings (to CLS + patches ONLY)
    x = x + self.interpolate_pos_encoding(x, w, h)
    # pos_embed covers indices [0..P] where index 0 = CLS, indices 1..P = patches
    # REGISTERS ARE NOT PRESENT YET → they get NO positional embedding
    
    # STEP 4: Insert register tokens BETWEEN CLS and patches
    if self.register_tokens is not None:
        x = torch.cat(
            (
                x[:, :1],    # CLS token (index 0) — ALREADY has pos_embed
                self.register_tokens.expand(x.shape[0], -1, -1),  # [B, N, D] — NO pos_embed
                x[:, 1:],   # patch tokens (indices 1..P) — ALREADY have pos_embed
            ),
            dim=1,
        )
    # FINAL: x = [CLS, REG_1, REG_2, ..., REG_N, patch_1, ..., patch_P]
    #  index:      0     1      2          N      N+1           N+P
    return x
```

**This reveals three critical facts:**

| Fact | Evidence |
|------|----------|
| CLS is always at **index 0** | `x[:, :1]` — first element, never moves |
| Registers occupy **indices 1 to N** | Inserted after CLS, before patches via `torch.cat` |
| Patches start at **index N+1** | `x[:, 1:]` (the old patches from pos 1 onward) end up after registers |

**And one subtle but crucial fact:**

> **Register tokens receive NO positional embedding.** Position embeddings are applied in Step 3, *before* registers are inserted in Step 4. This is a deliberate design choice — registers are position-agnostic computation buffers.

### 13.2 Output Splitting — How DINOv2 Confirms Positions at the Output Side

After the transformer processes all tokens, DINOv2 splits them back using **hard-coded index arithmetic**:

```python
def forward_features(self, x, masks=None):
    x = self.prepare_tokens_with_masks(x, masks)
    
    for blk in self.blocks:
        x = blk(x)
    
    x_norm = self.norm(x)
    
    return {
        "x_norm_clstoken":     x_norm[:, 0],                                    # Index 0 = CLS
        "x_norm_regtokens":    x_norm[:, 1 : self.num_register_tokens + 1],      # Indices 1..N
        "x_norm_patchtokens":  x_norm[:, self.num_register_tokens + 1 :],        # Indices N+1..end
    }
```

And in `get_intermediate_layers`:
```python
class_tokens = [out[:, 0] for out in outputs]                              # Always index 0
outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]      # Skip CLS + registers
```

**The position is deterministic by construction** — there is no dynamic lookup. You know where each token type is because you put it there during `prepare_tokens_with_masks`.

---

### 13.3 How Our HTR Pipeline Confirms Positions (ViTRGTSBackbone)

Our `ViTRGTSBackbone` uses a slightly different convention (no CLS token), but the same principle:

```python
# From models.py — ViTRGTSBackbone.forward()

# Registers FIRST, then patches
reg_tokens = self.register_tokens.expand(B, -1, -1)    # [B, R, D]
tokens = torch.cat([reg_tokens, patch_tokens], dim=1)  # [B, R+Np, D]

# Positional embeddings cover ALL tokens (registers + patches)
pos = self.pos_embed[:, :S, :]
tokens = tokens + pos

# After transformer:
reg_out   = encoded[:, :self.num_registers, :]      # Indices 0..R-1 = registers
patch_out = encoded[:, self.num_registers:, :]      # Indices R..end = patches
```

And in `TorchVisionViTBackbone` (with both CLS and registers):

```python
# CLS first, then registers, then patches
tokens = torch.cat([cls_tokens, reg_tokens, patch_tokens], dim=1)
# [CLS(0), REG_1(1), ..., REG_N(N), patch_1(N+1), ..., patch_P(N+P)]

# After transformer:
cls_out   = encoded[:, 0, :]                              # Index 0 = CLS
patch_out = encoded[:, 1 + self.num_registers:, :]        # Skip CLS + registers
```

---

### 13.4 Why CLS and Registers Look "The Same" at First Glance

Both CLS and registers share these properties:

| Property | CLS Token | Register Token |
|----------|-----------|---------------|
| Source | Learnable parameter (`nn.Parameter`) | Learnable parameter (`nn.Parameter`) |
| Content at input | Zero/random (no image pixels) | Zero/random (no image pixels) |
| Shape | `[1, 1, D]` | `[1, N, D]` |
| Participates in self-attention | Yes (all layers) | Yes (all layers) |
| Gradient source | Indirect (through attention to/from other tokens) | Indirect (through attention to/from other tokens) |
| Initialized | `normal_(std=1e-6)` in DINOv2 | `normal_(std=1e-6)` in DINOv2 |

**They are architecturally identical as input embeddings.** The only differences are:
1. **Position in the sequence** (construction-time)
2. **Whether their output is used** (inference-time)
3. **Whether they receive direct loss supervision** (training-time)

---

### 13.5 Seven Practical Methods to Differentiate CLS from Registers

Since the architecture treats them identically inside the transformer, you must differentiate them **externally**. Here are concrete approaches, ordered from simplest to most powerful:

#### Method 1: Index-Based Identification (Primary, Deterministic)

The **only guaranteed** way — use the known sequence positions:

```python
# After transformer forward pass, given output tensor `encoded` [B, S, D]:

def split_token_types(encoded, num_registers, has_cls=True):
    """Split the output sequence into CLS, register, and patch tokens."""
    idx = 0
    cls_out = None
    
    if has_cls:
        cls_out = encoded[:, 0, :]           # CLS always at index 0
        idx = 1
    
    reg_out = encoded[:, idx:idx + num_registers, :]   # Registers at indices 1..N (or 0..N-1)
    patch_out = encoded[:, idx + num_registers:, :]    # Patches fill the rest
    
    return cls_out, reg_out, patch_out
```

#### Method 2: L2 Norm Signature (Post-Training, Quantitative)

After training, CLS and register tokens develop **distinct norm profiles**:

```python
def identify_by_norm(encoded, num_registers, has_cls=True):
    """
    After training, registers develop HIGH norms (300-600),
    CLS has MEDIUM-HIGH norm, patches have LOW norm (~50).
    """
    norms = encoded.norm(dim=-1)  # [B, S]
    
    # Typical norm ranges (model-dependent):
    # Patches:    norm ≈ 30-100
    # CLS:        norm ≈ 100-200
    # Registers:  norm ≈ 200-600 (highest, quantized values)
    
    return norms
```

**Key insight from the paper**: Register norms are **quantized** (discrete high values), while CLS norm is continuous and intermediate. This is an emergent post-training property.

#### Method 3: Positional Embedding Analysis (Architecture-Level)

In DINOv2, CLS and patches share the positional embedding space, but **registers do NOT receive positional embeddings**:

```python
def check_positional_embedding_assignment(model):
    """
    In DINOv2: pos_embed covers [CLS, patch_1, ..., patch_P] — NO registers.
    In our ViTRGTSBackbone: pos_embed covers [reg_1, ..., reg_R, patch_1, ..., patch_P] — ALL tokens.
    
    This is a design choice that DIFFERENTIATES them.
    """
    # DINOv2 approach:
    # - CLS pos = pos_embed[:, 0]  ← has positional info
    # - Registers: NONE ← no positional identity
    # - Patches pos = pos_embed[:, 1:]  ← has spatial position
    
    # Our approach:
    # - Registers pos = pos_embed[:, 0:R]  ← have positional identity
    # - Patches pos = pos_embed[:, R:]  ← have spatial position
    
    pos_embed = model.pos_embed  # [1, max_seq_len, D]
    
    # Compare learned positional embeddings for register vs patch positions
    reg_positions = pos_embed[0, :model.num_registers, :]   # What the model learned for register positions
    patch_positions = pos_embed[0, model.num_registers:, :] # What the model learned for patch positions
    
    # After training, register positional embeddings will have DIFFERENT structure
    # from patch positional embeddings (patches encode spatial grid, registers don't)
    return reg_positions, patch_positions
```

#### Method 4: Attention Pattern Analysis (Behavioral, Post-Training)

CLS and registers develop **distinct attention patterns**:

```python
def differentiate_by_attention_pattern(attn_maps, num_registers, has_cls=True):
    """
    CLS: Broad, diffuse attention over all patches (summarizes entire image)
    Registers: Specialized, each attends to DIFFERENT subsets of patches
    Patches: Local, concentrated attention near their spatial neighbors
    """
    # attn_maps[layer] shape: [B, H, S, S]
    # Take last layer, head-averaged
    attn = attn_maps[-1].mean(dim=1)  # [B, S, S]
    
    idx = 0
    if has_cls:
        cls_attention = attn[:, 0, :]  # What CLS attends to [B, S]
        idx = 1
    
    reg_attention = attn[:, idx:idx+num_registers, :]  # [B, R, S] what each register attends to
    patch_attention = attn[:, idx+num_registers:, :]   # [B, P, S] what each patch attends to
    
    # DIFFERENTIATING METRIC: Entropy of attention distribution
    # CLS: HIGH entropy (attends broadly)
    # Each Register: MEDIUM entropy (specialized but still global)
    # Patches: LOW entropy (local attention)
    
    def attention_entropy(attn_row):
        """Shannon entropy of attention distribution."""
        # attn_row: [B, S] — already softmaxed
        return -(attn_row * torch.log(attn_row + 1e-10)).sum(dim=-1)  # [B]
    
    if has_cls:
        cls_entropy = attention_entropy(cls_attention)    # High (~5-6 bits)
    reg_entropy = attention_entropy(reg_attention.mean(dim=1))  # Medium (~4-5 bits)
    patch_entropy = attention_entropy(patch_attention.mean(dim=1))  # Low (~2-3 bits)
    
    return cls_entropy, reg_entropy, patch_entropy
```

#### Method 5: Linear Probing for Global vs Local Information

Train lightweight probes to test what each token "knows":

```python
def probe_global_vs_local(encoded, num_registers, has_cls=True, labels=None, positions=None):
    """
    From the paper (Table 4-5):
    - CLS: Excellent global classification, poor position/pixel reconstruction
    - Registers: Good global classification (slightly below CLS), poor local info
    - Patches: Poor global classification, excellent local position/pixel info
    
    Train two probes:
      1. Classification probe: token → image class (global)
      2. Position probe: token → spatial grid position (local)
    """
    idx = 0
    if has_cls:
        cls_feat = encoded[:, 0, :]  # [B, D]
        idx = 1
    reg_feat = encoded[:, idx:idx+num_registers, :].mean(dim=1)  # [B, D] average register
    patch_feat = encoded[:, idx+num_registers:, :]  # [B, P, D]
    
    # After training linear probes:
    # Global task (classification):
    #   CLS accuracy >> Register accuracy >> Patch accuracy
    #   Example: 86% > 69% > 14%
    
    # Local task (position prediction):
    #   Patch accuracy >> CLS accuracy >> Register accuracy
    #   Example: 42% > 25% > 23%
    
    # This separates CLS from registers: CLS is the BEST global token,
    # registers are secondary but still much better than patches at global tasks.
    pass
```

#### Method 6: Gradient Flow Analysis (Training-Time)

During training, CLS receives **direct gradient** from the classification loss, while registers receive **only indirect gradient** through attention:

```python
def analyze_gradient_flow(model, loss):
    """
    Backward pass reveals structural differences:
    - CLS: grad comes directly from loss (e.g., cross-entropy on CLS output)
    - Registers: grad comes ONLY through attention interactions
    
    This means:
    - CLS gradients are LARGER in magnitude
    - Register gradients are SMALLER and more diffuse
    """
    loss.backward()
    
    # CLS gradient (direct from loss):
    cls_grad = model.cls_token.grad  # [1, 1, D]
    cls_grad_norm = cls_grad.norm().item()
    
    # Register gradient (indirect through attention):
    reg_grad = model.register_tokens.grad  # [1, N, D]
    reg_grad_norm = reg_grad.norm(dim=-1).mean().item()
    
    # Typically: cls_grad_norm >> reg_grad_norm
    print(f"CLS grad norm: {cls_grad_norm:.6f}")
    print(f"Reg grad norm (mean): {reg_grad_norm:.6f}")
    # Ratio is often 10x-100x
```

#### Method 7: Cosine Similarity Clustering (Empirical)

After training, CLS and register embeddings occupy **different regions** of the embedding space:

```python
def cluster_analysis(encoded_batch, num_registers, has_cls=True):
    """
    Across many images:
    - CLS embeddings cluster by IMAGE CLASS (what's in the image)
    - Register embeddings cluster by FUNCTION (what role they serve)
    - Patch embeddings cluster by SPATIAL POSITION + local content
    
    Compute pairwise cosine similarity within and between token types.
    """
    # Collect over many images:
    all_cls = []    # [N_images, D]
    all_regs = []   # [N_images, R, D]
    all_patches = []  # [N_images, P, D]
    
    # After collecting:
    # 1. CLS-to-CLS similarity: varies (different images → different representations)
    # 2. REG_i-to-REG_i similarity across images: HIGH (same register, same role)
    # 3. REG_i-to-REG_j similarity: LOW (different registers specialize differently)
    # 4. CLS-to-REG similarity: MEDIUM (both global but different roles)
    
    # This means: if you don't know which is CLS vs register, compute
    # cross-image consistency. The one with HIGHEST cross-image variance
    # (changes most per image) is CLS. The ones with highest cross-image
    # CONSISTENCY (similar role regardless of image) are registers.
    pass
```

---

### 13.6 Complete Differentiation Procedure for Our Pipeline

Since our `ViTRGTSBackbone` does **not** use a CLS token (CTC doesn't need one), and our `TorchVisionViTBackbone` uses both CLS and registers, here's a concrete procedure to confirm their identities:

```python
def full_differentiation_analysis(model, image_tensor, device='cuda'):
    """
    Complete analysis to confirm and differentiate CLS vs register tokens.
    Works for TorchVisionViTBackbone (has CLS + registers).
    """
    model.eval()
    x = image_tensor.to(device)
    
    # Get explainability output
    seq_tokens, cls_out, reg_out, attn_maps, token_norms, (Hp, Wp) = model.forward_explain(x)
    
    B = x.size(0)
    S = token_norms.size(1)  # Total sequence length
    num_reg = model.num_registers
    
    print("="*60)
    print("TOKEN IDENTITY VERIFICATION")
    print("="*60)
    
    # 1. POSITION VERIFICATION
    print(f"\n[1] Sequence layout:")
    print(f"    Total tokens S = {S}")
    print(f"    Index 0: CLS token")
    print(f"    Indices 1..{num_reg}: Register tokens ({num_reg} registers)")
    print(f"    Indices {num_reg+1}..{S-1}: Patch tokens ({S - 1 - num_reg} patches)")
    
    # 2. NORM VERIFICATION
    print(f"\n[2] L2 Norms (averaged over batch):")
    norms = token_norms.mean(dim=0)  # [S]
    cls_norm = norms[0].item()
    reg_norms = norms[1:1+num_reg].tolist()
    patch_norm_mean = norms[1+num_reg:].mean().item()
    patch_norm_std = norms[1+num_reg:].std().item()
    
    print(f"    CLS norm:          {cls_norm:.2f}")
    for i, rn in enumerate(reg_norms):
        print(f"    Register-{i} norm:   {rn:.2f}")
    print(f"    Patch norm (mean): {patch_norm_mean:.2f} ± {patch_norm_std:.2f}")
    
    # Expected: CLS ~ 100-200, Registers ~ 200-600, Patches ~ 30-100
    # If CLS norm < register norms → confirms CLS is NOT a register
    
    # 3. ATTENTION ENTROPY VERIFICATION
    print(f"\n[3] Attention entropy (last layer, head-averaged):")
    last_attn = attn_maps[-1].mean(dim=1)  # [B, S, S]
    
    def entropy(row):
        row = row.clamp(min=1e-10)
        return -(row * row.log()).sum(dim=-1).mean().item()
    
    cls_entropy = entropy(last_attn[:, 0, :])
    print(f"    CLS entropy:       {cls_entropy:.3f}")
    for i in range(num_reg):
        reg_ent = entropy(last_attn[:, 1+i, :])
        print(f"    Register-{i} entropy: {reg_ent:.3f}")
    patch_entropies = [entropy(last_attn[:, 1+num_reg+j, :]) for j in range(min(5, S-1-num_reg))]
    print(f"    Patch entropy (first 5, mean): {sum(patch_entropies)/len(patch_entropies):.3f}")
    
    # Expected: CLS entropy > Register entropy > Patch entropy
    
    # 4. CROSS-IMAGE CONSISTENCY (run on multiple images)
    print(f"\n[4] Summary:")
    print(f"    CLS: Global aggregator, highest entropy, medium norm, output USED for classification")
    print(f"    Registers: Computation buffers, specialized attention, highest norm, output DISCARDED")
    print(f"    Patches: Local spatial tokens, lowest entropy, lowest norm, output USED for CTC/dense tasks")
    
    return {
        'cls_norm': cls_norm,
        'reg_norms': reg_norms,
        'patch_norm_mean': patch_norm_mean,
        'cls_entropy': cls_entropy,
    }
```

---

### 13.7 The Definitive Answer: Why Position Is Deterministic (Not Learned)

A common confusion: "Does the model *decide* where CLS and registers go?"

**No.** The position is **hard-coded at construction time** by the programmer:

```python
# The ORDER in torch.cat IS the position. Period.
tokens = torch.cat([cls_tokens, reg_tokens, patch_tokens], dim=1)
#                   ^^^^^^^^^^  ^^^^^^^^^^  ^^^^^^^^^^^^
#                   index 0     index 1-N   index N+1 onward
```

The transformer's self-attention is **permutation equivariant** — it doesn't inherently know "which position is which." The **only** way the model knows a token's role is through:

1. **Its content** (CLS/registers start as learned constants; patches start with image data)
2. **Its positional embedding** (encodes the sequence position)
3. **What you do with its output** (CLS → loss; registers → discarded; patches → dense output)

This means you can place registers in **any position** relative to CLS and patches, and the model will learn to use them. The paper's choice of [CLS, REG, ..., PATCH, ...] is a convention, confirmed by DINOv2's `prepare_tokens_with_masks`. Our pipeline's choice of [REG, ..., PATCH, ...] (no CLS) is equally valid.

---

### 13.8 Summary Table: CLS vs Register Differentiation

| Differentiating Factor | CLS Token | Register Tokens | How to Measure |
|----------------------|-----------|-----------------|----------------|
| **Sequence index** | Always 0 | 1 to N (after CLS) | `encoded[:, 0]` vs `encoded[:, 1:N+1]` |
| **Output usage** | Fed to classification head | **Discarded** | Check loss function inputs |
| **Positional embedding** | Receives dedicated pos_embed[0] | None (DINOv2) or shared pos_embed (our pipeline) | Inspect `prepare_tokens` code |
| **Gradient magnitude** | High (direct loss signal) | Low (indirect via attention) | `.grad.norm()` during training |
| **L2 norm after training** | Medium-high (~100-200) | Highest (~200-600, quantized) | `token.norm(dim=-1)` |
| **Attention entropy** | Highest (broad, summarizes all) | Medium (specialized per register) | Shannon entropy of attention row |
| **Cross-image consistency** | Low (changes per image class) | High (same functional role) | Cosine similarity across images |
| **Global classification probe** | Best accuracy (~86%) | Good accuracy (~69-97%) | Linear probe trained on token → class |
| **Local position probe** | Poor (~23%) | Poor (~23%) | Linear probe trained on token → position |
| **Number** | Always 1 | Hyperparameter (typically 4) | `model.num_registers` |
| **Historical origin** | BERT (2018), standard ViT | Darcet et al. (2024) | — |

---

## Key Takeaways for Your Research

1. **Register tokens are NOT a new type of attention or transformer architecture.** They are simply additional learnable input tokens whose output is discarded. The transformer itself is unchanged.

2. **The paper's contribution is the INSIGHT, not the technique.** Adding learnable tokens was already done (Memory Transformer, 2020). The new contribution is understanding WHY ViTs develop artifacts and showing that registers fix them by providing dedicated buffer space.

3. **For your HTR pipeline**: Your model (6M params, CTC loss, 6.5K training samples) is in a very different regime than the paper's models (86M–1.1B params, diverse losses, 14M+ images). The core artifact phenomenon may not manifest in your setting, but registers can still serve as an **explicit global information channel** that separates writer style from character identity — which is the explainability angle unique to your work.

4. **What to measure in your experiments**:
   - Token norm distributions (should you see a bimodal pattern without registers?)
   - Register attention maps (do different registers specialize to different text regions?)
   - Register embedding similarity across writers (do they cluster by writer?)
   - Attention map smoothness with vs without registers (even if artifacts don't appear, is there a qualitative difference?)
