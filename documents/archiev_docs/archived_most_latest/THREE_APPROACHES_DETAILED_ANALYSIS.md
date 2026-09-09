# Three Novel Approaches for ViT-RGTS HTR Pipeline: Detailed Analysis

**Prepared by:** Research Analysis for HTR-Pipeline Integration  
**Date:** May 2026  
**Reference Pipeline:** ViT-RGTS (CNN-stem → ViT → BiLSTM-CTC) for Handwritten Text Recognition  

---

## Table of Contents

1. [Current Architecture Baseline](#1-current-architecture-baseline)
2. [Approach 1: Test-Time Registers (Jiang et al., NeurIPS 2025 Spotlight)](#2-approach-1-test-time-registers)
3. [Approach 2: Gated Attention (Qiu et al., NeurIPS 2025 Best Paper)](#3-approach-2-gated-attention)
4. [Approach 3: U-Net + ViT Combined Architecture](#4-approach-3-u-net--vit-combined-architecture)
5. [Comparative Analysis](#5-comparative-analysis)
6. [Integration Roadmap](#6-integration-roadmap)
7. [Recommended Priority Order](#7-recommended-priority-order)

---

## 1. Current Architecture Baseline

### Pipeline

```
Input [B, 1, 128, 1024]
  → CNN Stem: Conv(1→32,k7,s4×2) → Conv(32→64,k3,s2×2) → Conv(64→128,k3,s2×2) → Conv(128→256,k3,s2×1)
  → [B, 256, 4, 128]  (4 height rows × 128 width columns)
  → AdaptiveMaxPool2d(1, None) → [B, 256, 1, 128]  (height collapsed)
  → Flatten → [B, 128, 256]  (128 patch tokens, dim=256)
  → Prepend R register tokens → [B, R+128, 256]
  → Add positional embeddings
  → nn.TransformerEncoder (6 layers, 8 heads, pre-norm, GELU, FFN=1024)
  → Split: registers [B, R, 256] + patches [B, 128, 256]
  → Reshape to [128, B, 256] (time-major)
  → BiLSTM (3 layers, hidden=256, bidirectional) → [128, B, 512]
  → LayerNorm → Linear(512, 80) → CTC decode
```

### Current Register Results

| Run | Registers | CER (%) | Notes |
|-----|-----------|---------|-------|
| run_76 | 0 | 6.43 | Baseline |
| run_84 | 4 | 6.47 | Slight regression |
| run_63 | 8 | 6.26 | Improvement |
| run_71 | 16 | 6.10 | Best |

### Key Characteristics
- **Pre-norm** transformer (`norm_first=True`) with standard `nn.TransformerEncoderLayer`
- **Dual CTC supervision**: RNN path + CNN shortcut (weight 0.1) during training
- **1D attention**: Height is collapsed before ViT, so self-attention is purely horizontal
- **128 patch tokens**: Each covers ~8px horizontally (~3 tokens per character)
- **Register tokens**: Prepended, learned, excluded from CTC path

---

## 2. Approach 1: Test-Time Registers

### Paper: "Vision Transformers Don't Need Trained Registers"
**Authors:** Nick Jiang*, Amil Dravid*, Alexei A. Efros, Yossi Gandelsman (UC Berkeley)  
**Venue:** NeurIPS 2025 **Spotlight**  
**Code:** https://github.com/nickjiang2378/test-time-registers

### 2.1 Core Idea

Instead of retraining ViTs with additional register tokens (expensive), identify the specific **register neurons** — a sparse set of MLP neurons that cause high-norm outlier tokens — and redirect their activations to an appended zero-token at inference time, mimicking register behaviour without any training.

### 2.2 Mechanism (Step-by-Step)

1. **Outlier Discovery:** In trained ViTs, certain patch tokens develop abnormally high norms at low-information image positions (backgrounds). These "outlier tokens" act as global memory sinks.

2. **Register Neuron Detection (Algorithm 1):**
   - Pass M images through the model, log neuron activations
   - For each image, find outlier positions (tokens with norm > threshold)
   - For each neuron in layers 0..top_layer: compute mean activation at outlier positions
   - Return top-K neurons with highest average activation at outlier positions
   - Typically ~100 neurons out of ~100K total

3. **Test-Time Intervention (Algorithm 2):**
   - Append R zero-tokens to the input sequence (test-time registers)
   - At each register neuron during forward pass:
     - Copy the maximum activation value across all patch tokens to the register token
     - Zero out (or set to mean) the activations on all image patch tokens
   - Result: Outlier information migrates to the appended token, patch tokens become "clean"

4. **Hook-based Implementation:**
   ```python
   hook_manager = HookManager(model)
   hook_manager.reinit(mode=HookMode.INTERVENE)
   hook_manager.intervene_register_neurons(
       num_registers=1,
       neurons_to_ablate=register_neurons,  # {layer: [neuron_indices]}
       scale=1.0,
       normal_values="zero"  # or "mean", "only_outliers", "same"
   )
   hook_manager.finalize()
   output = model(input)
   ```

### 2.3 Paper Results

| Task | Original | w/ Trained Regs | w/ Test-Time Regs |
|------|----------|-----------------|-------------------|
| IN1k Linear Probe (DINOv2-L) | 86.4 | 86.7 | 86.4 |
| ADE20k mIoU | 48.3 | 49.1 | 49.1 |
| Zero-Shot Seg mIoU | 38.3 | 33.9 | **38.9** |
| LOST VOC07 (Object Discovery) | 32.2 | 56.2 | 53.8 |
| VLM Avg. (LLaVA) | 46.2 | — | 46.2 |

**Key takeaway:** Test-time registers match or exceed trained registers on dense prediction and segmentation, with dramatic improvement on object discovery (+21 points on DINOv2).

### 2.4 Integration with ViT-RGTS HTR Pipeline

#### What Changes

```python
# In models.py, ViTRGTSBackbone.forward():
# 1. Append R zero-tokens AFTER the patch tokens (not prepend like trained registers)
# 2. Apply hook interventions on register neurons in MLP layers
# 3. Exclude the appended tokens from CTC path (same as trained registers)
```

#### Integration Plan (Modular)

**Step 1: Create `utils/test_time_registers.py`**

```python
class TTRegisterManager:
    """Test-time register neuron detection and intervention for ViT-RGTS."""
    
    def __init__(self, model, num_registers=1):
        self.model = model
        self.num_registers = num_registers
        self.register_neurons = {}  # {layer_idx: [neuron_indices]}
        self.hooks = []
    
    def find_register_neurons(self, dataloader, top_k=50, 
                               norm_threshold=30, top_layer=None):
        """
        Algorithm 1: Detect register neurons from training/val images.
        For each MLP layer, find neurons with consistently high activation
        at outlier token positions.
        """
        # 1. Forward pass through images, collect neuron activations
        # 2. Identify outlier positions (high-norm tokens)
        # 3. Score neurons by mean activation at outlier positions
        # 4. Return top-K across all layers
        pass
    
    def install_hooks(self, scale=1.0, normal_values="zero"):
        """
        Install forward hooks on MLP layers to redirect register neuron
        activations to appended zero-tokens.
        """
        pass
    
    def remove_hooks(self):
        """Remove all hooks."""
        pass
```

**Step 2: Modify forward pass to support appended tokens**

```python
# In ViTRGTSBackbone.forward():
def forward(self, x, num_tt_registers=0):
    # ... existing code until token concatenation ...
    
    if num_tt_registers > 0:
        tt_regs = torch.zeros(B, num_tt_registers, D, device=x.device)
        tokens = torch.cat([reg_tokens, patch_tokens, tt_regs], dim=1)
    else:
        tokens = torch.cat([reg_tokens, patch_tokens], dim=1)
    
    # ... encoder ...
    
    # Split: trained_regs | patches | test-time_regs
    reg_out = encoded[:, :R, :]
    patch_out = encoded[:, R:R+Np, :]
    tt_reg_out = encoded[:, R+Np:, :] if num_tt_registers > 0 else None
    
    seq_tokens = patch_out.transpose(0, 1)
    return seq_tokens, reg_out, (Hp, Wp), tt_reg_out
```

**Step 3: Config addition**

```yaml
# In baseline_vit_rgts_v2.yaml:
arch:
  test_time_registers: 0       # 0 = disabled, 1+ = append this many at inference
  tt_reg_neuron_threshold: 30  # norm threshold for outlier detection
  tt_reg_top_k: 50             # number of register neurons to identify
```

#### Difficulty of Integration
**Low-Medium.** The hook-based approach is non-invasive. The model architecture stays unchanged; hooks modify activations on-the-fly. Main work is adapting `find_register_neurons` to our architecture (MLP is inside `nn.TransformerEncoderLayer`, need to hook `linear2` of each layer).

#### Will It Improve Results?

**Prediction: YES for attention map quality, UNCERTAIN for CER.**

Rationale:
- **Attention maps:** Very likely to improve. Our run_76 (0 registers) almost certainly has outlier tokens. Test-time registers would redirect them to the appended token, producing cleaner attention patterns without retraining.
- **CER:** The CTC alignment was learned without test-time registers, so the model's decoder expects the representation as-is. Intervention might slightly disrupt the learned mapping. Expected: **+/- 0.1% CER, significant attention map improvement.**
- **Key experiment:** Apply test-time registers to run_76 (0 trained registers). If attention maps become comparable to run_71 (16 trained registers) while maintaining ~6.4% CER, this validates the approach for interpretability without retraining.

**Why this is worth trying:**
1. **Zero training cost** — only inference-time modification
2. **Professor's request** is specifically about attention map quality — this directly addresses it
3. **Novel contribution:** No one has applied test-time registers to HTR/CTC models
4. **Publishable comparison:** test-time registers vs. trained registers on the same architecture

---

## 3. Approach 2: Gated Attention

### Paper: "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free"
**Authors:** Zihan Qiu et al. (Alibaba/Qwen team)  
**Venue:** NeurIPS 2025 **Best Paper Award** (4 out of 5,290 accepted)  
**Code:** https://github.com/qiuzh20/gated_attention

### 3.1 Core Idea

Add a **sigmoid gate** after the Scaled Dot-Product Attention (SDPA) output, computed from the query. This gate modulates each attention head independently, introducing:

1. **Non-linearity** into the low-rank V×O projection
2. **Input-dependent sparsity** — the gate can suppress entire heads per token
3. **Elimination of "attention sink"** — no more disproportionate attention to first/padding tokens

### 3.2 Mechanism

Standard attention:
```
attn_output = softmax(QK^T / √d) · V
output = attn_output · W_O
```

Gated attention (headwise):
```
Q_extended = W_Q(x)  →  split into [Q, gate_logits]
gate = sigmoid(gate_logits)           # [B, T, H, 1] per head
attn_output = softmax(QK^T / √d) · V
gated_output = attn_output * gate     # element-wise per head
output = gated_output · W_O
```

#### Two Variants

1. **Headwise gating:** One scalar gate per head per token position  
   - Extra parameters: `num_heads` additional outputs from Q projection  
   - Gate shape: `[B, seq_len, num_heads, 1]`  

2. **Elementwise gating:** One gate per element of the attention output  
   - Extra parameters: `head_dim × num_heads` additional outputs from Q projection  
   - Gate shape: `[B, seq_len, num_heads, head_dim]`  

### 3.3 Paper Results (15B MoE, 3.5T tokens)

| Metric | Baseline | Headwise Gate | Elementwise Gate |
|--------|----------|---------------|------------------|
| Avg. benchmark | — | +0.5-1.0% | +0.8-1.2% |
| Training stability | Sensitive to LR | Tolerates 2× LR | Tolerates 2× LR |
| RULER (long context) | Degraded >32K | Stable to 128K | Stable to 128K |
| Attention sink | Present | Reduced | Eliminated |

**Key finding:** Gating eliminates the attention sink phenomenon — the tendency to allocate disproportionate attention to the first token — which is directly related to the outlier token problem addressed by registers.

### 3.4 Integration with ViT-RGTS HTR Pipeline

#### What Changes

Our ViT uses `nn.TransformerEncoderLayer` which internally uses `nn.MultiheadAttention`. To add gating, we need a **custom attention layer**.

**Step 1: Create `models_gated.py` or extend `models.py`**

```python
class GatedMultiheadAttention(nn.Module):
    """
    Standard MHSA with a query-derived sigmoid gate on the output.
    
    Supports headwise (1 scalar per head) and elementwise (full dim per head) gating.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1, 
                 gate_type="headwise"):  # "headwise" or "elementwise"
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.gate_type = gate_type
        
        # Standard Q, K, V projections
        # Q gets extra dimensions for the gate
        if gate_type == "headwise":
            gate_dim = num_heads  # 1 scalar per head
        else:  # elementwise
            gate_dim = embed_dim  # full dim per head
        
        self.q_proj = nn.Linear(embed_dim, embed_dim + gate_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, need_weights=False):
        B, S, D = x.shape
        
        # Project Q with gate, K, V
        qg = self.q_proj(x)  # [B, S, D + gate_dim]
        
        if self.gate_type == "headwise":
            q = qg[:, :, :self.embed_dim]
            gate_logits = qg[:, :, self.embed_dim:]  # [B, S, H]
            gate = torch.sigmoid(gate_logits).unsqueeze(-1)  # [B, S, H, 1]
        else:
            q = qg[:, :, :self.embed_dim]
            gate_logits = qg[:, :, self.embed_dim:]  # [B, S, D]
            gate = torch.sigmoid(gate_logits).view(B, S, self.num_heads, self.head_dim)
        
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to [B, H, S, d]
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # SDPA
        attn_weights = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = attn_weights @ v  # [B, H, S, d]
        
        # Apply gate
        attn_output = attn_output.transpose(1, 2)  # [B, S, H, d]
        attn_output = attn_output * gate
        attn_output = attn_output.reshape(B, S, D)
        
        output = self.o_proj(attn_output)
        
        if need_weights:
            return output, attn_weights
        return output
```

**Step 2: Create custom TransformerEncoderLayer**

```python
class GatedTransformerEncoderLayer(nn.Module):
    """Pre-norm transformer layer with gated attention."""
    
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, gate_type="headwise"):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = GatedMultiheadAttention(d_model, nhead, dropout, gate_type)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

**Step 3: Config extension**

```yaml
arch:
  attention_gate: "none"  # "none", "headwise", "elementwise"
```

#### Difficulty of Integration
**Medium.** Requires replacing `nn.TransformerEncoderLayer` with a custom implementation. Need to ensure `forward_explain()` still works (manually extract attention weights). The custom layer is ~40 lines of code. Must retrain from scratch.

#### Will It Improve Results?

**Prediction: LIKELY YES for both CER and attention quality.**

Rationale:
- **Attention sink elimination:** Our ViT-RGTS *does* exhibit outlier tokens (that's why registers help). Gated attention attacks the same root cause — the softmax constraint forces attention distribution to sum to 1, creating sinks. The gate can suppress this per-head.
- **Training stability:** The paper shows gated models tolerate larger learning rates and train more stably. Our ViT-RGTS uses 3 separate LR groups and careful scheduling — gating may simplify this.
- **Non-linearity:** The low-rank V×O mapping in standard attention limits expressivity. The gate introduces a data-dependent non-linearity, which may help the model better separate similar characters (e.g., 'n'/'u', 'l'/'t').
- **Parameter cost:** Headwise gating adds only `num_heads = 8` extra parameters per layer to Q. For 6 layers: +48 parameters total. Negligible.
- **Expected improvement:** +0.1–0.3% CER reduction, significantly sharper attention maps (no sink pattern).

**Why this is worth trying:**
1. **NeurIPS 2025 Best Paper** — highest possible validation of the approach
2. **Directly addresses the attention sink = outlier token problem** — same phenomenon registers address
3. **Minimal parameter overhead** — headwise adds just 48 params to a 7M+ param model
4. **Orthogonal to registers** — can combine gating WITH register tokens for compound benefit
5. **Novel in HTR** — no one has applied gated attention to CTC-based recognition
6. **Already adopted by Qwen3-Next** — real-world production validation

**Critical insight:** Gated attention and register tokens attack the same problem (attention sinks/outliers) from opposite ends:
- Registers: Provide a "garbage collector" token to absorb global info
- Gating: Prevent the need for garbage collection by allowing heads to suppress themselves

**Combined hypothesis:** Gated attention + register tokens may provide the best of both — the gate handles per-head sparsity, registers provide explicit global memory.

---

## 4. Approach 3: U-Net + ViT Combined Architecture

### 4.1 Concept

A U-Net encoder-decoder with a ViT bottleneck combines the best of both worlds:
- **U-Net:** Multi-scale spatial features with skip connections preserving fine-grained stroke details
- **ViT:** Global context understanding in the bottleneck for sequence-level reasoning

### 4.2 Architecture Design for HTR

```
INPUT [B, 1, 128, 1024]
  │
  ├── Encoder Block 1: Conv(1→32,k3,s2) + BN + GELU     → [B, 32, 64, 512]   ──┐
  ├── Encoder Block 2: Conv(32→64,k3,s2) + BN + GELU    → [B, 64, 32, 256]   ──┤
  ├── Encoder Block 3: Conv(64→128,k3,s2) + BN + GELU   → [B, 128, 16, 128]  ──┤
  ├── Encoder Block 4: Conv(128→256,k3,s(2,1)) + BN + GELU → [B, 256, 8, 128] ──┤
  │                                                                              │
  ├── ViT Bottleneck:                                                            │
  │     Pool to [B, 256, 1, 128] → 128 tokens                                   │
  │     + R register tokens                                                      │
  │     TransformerEncoder (6L, 8H)                                              │
  │     → [B, 128, 256] patch tokens                                             │
  │     Reshape to [B, 256, 1, 128]                                              │
  │     Upsample height to [B, 256, 8, 128]                                      │
  │                                                                              │
  ├── Decoder Block 4: ConvT + Skip[Enc4] + Conv       → [B, 128, 16, 128]    ──┘
  ├── Decoder Block 3: ConvT + Skip[Enc3] + Conv       → [B, 64, 32, 256]
  ├── Decoder Block 2: Pool height → [B, 64, 1, 256]   (collapse to 1D for CTC)
  │
  └── Final: flatten → BiLSTM → CTC
```

### 4.3 Key Design Decisions

**Why skip connections matter for HTR:**
- Character strokes have fine spatial details lost in deep encoders
- Ascenders/descenders (b, d, p, q) require multi-scale vertical info
- Skip connections preserve edge information while ViT bottleneck captures sequence context

**Why NOT a full U-Net decoder for CTC:**
- CTC needs a 1D temporal sequence, not a 2D spatial map
- Decoder only partially reconstructs — height collapses at a later stage than current pipeline
- Preserves 2D info longer, giving ViT's global reasoning access to richer spatial features

### 4.4 Integration with Current Pipeline

#### What Changes

**Major architectural change** — essentially replacing `ViTRGTSBackbone` with `UNetViTBackbone`.

```python
class UNetViTBackbone(nn.Module):
    """
    U-Net encoder with ViT bottleneck for HTR.
    Skip connections preserve spatial details lost in the current
    CNN-stem-only approach.
    """
    
    def __init__(self, embed_dim=256, depth=6, num_heads=8, 
                 num_registers=4, dropout=0.1):
        super().__init__()
        
        # Encoder (same convolutions as current CNN stem, but save intermediates)
        self.enc1 = self._make_enc_block(1,   32,  stride=(2, 2))
        self.enc2 = self._make_enc_block(32,  64,  stride=(2, 2))
        self.enc3 = self._make_enc_block(64,  128, stride=(2, 2))
        self.enc4 = self._make_enc_block(128, embed_dim, stride=(2, 1))
        
        # Bottleneck: height pool + ViT
        self.bottleneck_pool = nn.AdaptiveMaxPool2d((1, None))
        self.vit = ...  # Same TransformerEncoder as current
        
        # Decoder (lightweight — only reconstruct to enc4 level)
        self.dec4_upsample = nn.Upsample(scale_factor=(8, 1))
        self.dec4_conv = self._make_dec_block(embed_dim * 2, 128)
        
        # Final: collapse height, ready for CTC
        self.final_pool = nn.AdaptiveMaxPool2d((1, None))
    
    def forward(self, x):
        # Encode with skip connections
        e1 = self.enc1(x)       # [B, 32, 64, 512]
        e2 = self.enc2(e1)      # [B, 64, 32, 256]
        e3 = self.enc3(e2)      # [B, 128, 16, 128]
        e4 = self.enc4(e3)      # [B, 256, 8, 128]  ← 2D features!
        
        # ViT bottleneck (1D)
        b = self.bottleneck_pool(e4)  # [B, 256, 1, 128]
        tokens = ...  # Same as current: flatten, add registers, transformer
        
        # Decode: restore 2D + skip connection
        b_2d = self.dec4_upsample(b_out)  # [B, 256, 8, 128]
        d4 = self.dec4_conv(cat([b_2d, e4], dim=1))  # [B, 128, 8, 128]
        
        # Collapse to 1D for CTC
        out = self.final_pool(d4)  # [B, 128, 1, 128]
        seq = out.squeeze(2).transpose(1, 2).transpose(0, 1)  # [128, B, 128]
        
        return seq, reg_out, (1, 128)
```

**Config extension:**

```yaml
arch:
  type: "unet_vit"  # new architecture type
  skip_connections: true
  decoder_depth: 1  # lightweight decoder
```

#### Difficulty of Integration
**High.** This is a full architectural redesign. Requires:
1. New backbone class (~150-200 lines)
2. Modifications to `HTRNet` to handle different backbone types
3. New training configs and hyperparameter tuning
4. CTC head dimension changes (decoder output may differ from current 256)
5. Retraining from scratch
6. May need to adjust BiLSTM input dim

#### Will It Improve Results?

**Prediction: POSSIBLE but uncertain. Most likely marginal CER improvement with MUCH better GradCAM visualizations.**

Rationale:

**Arguments FOR:**
- The current pipeline loses ALL vertical information at the AdaptiveMaxPool2d. Skip connections would preserve ascender/descender structure, potentially helping disambiguate characters like 'b' vs 'h', 'p' vs 'q'
- U-Net skip connections are proven to preserve fine-grained spatial details in segmentation — analogous benefit for stroke-level features
- The ViT bottleneck maintains the global sequence reasoning capability
- GradCAM on decoder layers would produce **much richer 2D heatmaps** because the decoder maintains 8×128 spatial dimensions (32× more spatial resolution than the current 1D attention)

**Arguments AGAINST:**
- The current CNN stem + pool approach already works well (6.10% CER with 16 registers)
- HTR line images are dominated by horizontal structure — vertical info may be less critical than in natural images
- Added complexity (more parameters, longer training, harder to debug)
- U-Net decoders are designed for pixel-level prediction (segmentation); CTC operates on a 1D temporal sequence
- Risk of over-engineering for marginal gains

**Expected improvement:** 0.0–0.3% CER (uncertain), significantly improved 2D attention visualizations.

**Why this might NOT be worth it as a first priority:**
1. Highest development cost of the three approaches
2. Most uncertain benefit for CER
3. Requires full retraining
4. The current pipeline already achieves competitive results

**When it IS worth trying:**
- If the thesis needs a novel architectural contribution (not just application of existing methods)
- If 2D spatial attention maps are a hard requirement and GradCAM on the CNN stem is insufficient
- As a longer-term research direction after validating approaches 1 and 2

---

## 5. Comparative Analysis

### 5.1 Side-by-Side Comparison

| Criterion | Test-Time Registers | Gated Attention | U-Net + ViT |
|-----------|:-------------------:|:---------------:|:-----------:|
| **Implementation effort** | Low (~100 lines) | Medium (~200 lines) | High (~400+ lines) |
| **Requires retraining** | **No** | Yes (from scratch) | Yes (from scratch) |
| **CER improvement (est.)** | ±0.1% | +0.1–0.3% | 0.0–0.3% |
| **Attention map quality** | Significant ↑ | Significant ↑ | Major ↑ (2D) |
| **Parameter overhead** | 0 | +48 (headwise) | +30–50K (decoder) |
| **Novelty for HTR** | High | High | Medium |
| **Publication potential** | High (NeurIPS Spotlight method) | Very High (Best Paper method) | Medium |
| **Risk** | Low | Low-Medium | Medium-High |
| **Combinable** | ✅ with Gated | ✅ with TTR | ✅ with both |

### 5.2 How They Address the Same Root Problem

All three approaches address the **attention artifact / outlier token** problem but from different angles:

```
                    Root Cause: softmax attention forces sum-to-1
                              |
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     Test-Time Registers   Gated Attn     U-Net + ViT
     "Redirect outliers    "Let heads     "Give spatial info
      to garbage tokens"   suppress        via skip connections
                           themselves"      so ViT doesn't need
                                           to encode everything"
              |               |               |
              ▼               ▼               ▼
        Inference-only    Training-time    Architecture-level
        modification      modification     redesign
```

### 5.3 Combinability Matrix

| Combination | Feasibility | Expected Synergy |
|-------------|:-----------:|:-----------------|
| TTR + Gated Attention | **Easy** | Gating reduces sinks during training; TTR cleans residual artifacts at inference |
| TTR + U-Net + ViT | **Easy** | TTR on the ViT bottleneck of U-Net |
| Gated Attn + U-Net + ViT | **Easy** | Replace standard attention in ViT bottleneck with gated attention |
| All three | **Feasible** | Maximum coverage but diminishing returns likely |

### 5.4 Experiment Design Matrix

| Experiment ID | Architecture | Registers (trained) | Test-Time Regs | Gated Attn | Expected CER |
|---------------|:------------:|:-------------------:|:--------------:|:----------:|:------------:|
| E0 (baseline) | ViT-RGTS | 0 | No | No | 6.43% |
| E1 | ViT-RGTS | 16 | No | No | 6.10% |
| **E2** | ViT-RGTS | **0** | **Yes (1)** | No | ~6.4% (attn ↑) |
| **E3** | ViT-RGTS | **16** | **Yes (1)** | No | ~6.1% (attn ↑↑) |
| **E4** | ViT-RGTS-Gated | 0 | No | **headwise** | ~6.2-6.3% |
| **E5** | ViT-RGTS-Gated | 8 | No | **headwise** | ~5.9-6.1% |
| **E6** | ViT-RGTS-Gated | 8 | Yes (1) | **headwise** | ~5.9-6.0% |
| E7 | U-Net-ViT | 8 | No | No | ~6.0-6.3% |
| E8 | U-Net-ViT-Gated | 8 | Yes (1) | headwise | ~5.8-6.1% |

**Bold = highest priority experiments**

---

## 6. Integration Roadmap

### Phase 1: Test-Time Registers (1–2 days)

```
Week 1, Day 1-2:
├── Create utils/test_time_registers.py
│   ├── TTRegisterManager class
│   ├── find_register_neurons() — adapt Algorithm 1 for nn.TransformerEncoderLayer
│   └── install_hooks() / remove_hooks()
├── Modify ViTRGTSBackbone.forward() to optionally append zero-tokens
├── Run register neuron detection on run_76 (0 regs) using validation set
├── Evaluate: attention maps before/after, CER before/after
└── Visualize with existing scripts/visualization/char_gradcam.py
```

**Key adaptation needed:** The paper targets DINOv2/CLIP where the MLP is a separate module. In our `nn.TransformerEncoderLayer`, the MLP is `self.linear1` and `self.linear2`. We need to hook into the activation after `self.linear1` (the GELU activation) in each layer.

### Phase 2: Gated Attention (3–5 days)

```
Week 1-2:
├── Create models/gated_attention.py
│   ├── GatedMultiheadAttention (headwise + elementwise variants)
│   └── GatedTransformerEncoderLayer
├── Modify ViTRGTSBackbone to accept gate_type parameter
├── Modify forward_explain() for gated attention weight extraction
├── Create config: baseline_vit_rgts_gated.yaml
├── Train: 80 epochs, R={0, 8}, gate_type={headwise, elementwise}
├── Evaluate CER + attention maps
└── Compare to baseline registers
```

### Phase 3: U-Net + ViT (if needed, 1–2 weeks)

```
Week 3+:
├── Create models/unet_vit_backbone.py
├── Implement encoder blocks with skip connections
├── ViT bottleneck integration
├── Lightweight decoder for 2D → 1D transition
├── New config: unet_vit.yaml
├── Train: 80 epochs, R={0, 8}
└── Compare to ViT-RGTS baseline
```

---

## 7. Recommended Priority Order

### Priority 1: Test-Time Registers ⭐⭐⭐⭐⭐

**Why first:**
- Zero retraining cost
- Directly addresses professor's attention map quality request
- Novel for HTR
- Can be tested on ALL existing trained models immediately
- NeurIPS 2025 Spotlight = strong academic backing

### Priority 2: Gated Attention ⭐⭐⭐⭐

**Why second:**
- NeurIPS 2025 Best Paper = strongest possible validation
- Directly attacks the attention sink problem
- Minimal parameter overhead
- Orthogonal to registers — combine for compound benefit
- Requires retraining but uses same pipeline infrastructure

### Priority 3: U-Net + ViT ⭐⭐⭐

**Why third:**
- Highest development cost
- Most uncertain CER benefit
- Better justified after demonstrating that spatial info helps (via GradCAM analysis)
- Consider only if CER is not competitive or if 2D attention maps are absolutely required

### Decision Framework

```
IF professor wants better attention maps ASAP:
    → Start with Test-Time Registers (no retraining, immediate results)

IF thesis needs a training-time improvement:
    → Implement Gated Attention (strongest paper backing, minimal code change)

IF both attention quality AND CER improvement needed:
    → Test-Time Registers first (validate hypothesis), then Gated Attention (improve training)
    → Combine: Gated Attention + Register Tokens + Test-Time Registers

IF architectural novelty is needed for thesis:
    → U-Net + ViT (but validate simpler approaches first)
```

---

## Appendix A: Mathematical Connection Between Approaches

The outlier token phenomenon can be understood through the softmax attention constraint:

$$\sum_{j=1}^{S} \text{softmax}(q_i^T k_j / \sqrt{d})_j = 1$$

When token $i$ has "nothing useful to attend to" (e.g., a padding or background token), it still must distribute its attention somewhere. The model learns to dump attention onto specific "sink" tokens.

**Register tokens** solve this by providing a dedicated sink target: $\sum_{j} a_{ij} = a_{i,\text{reg}} + \sum_{j \neq \text{reg}} a_{ij} = 1$

**Gated attention** solves this by allowing heads to self-suppress: $\text{output}_h = \sigma(g_h) \cdot (\sum_j a_{ij} v_j)$. When $g_h \approx 0$, the head contributes nothing regardless of attention distribution.

**U-Net skip connections** reduce the *need* for sinks by providing spatial information via a separate path, so the ViT bottleneck doesn't need to encode fine-grained details it struggles to represent in 1D.

---

## Appendix B: Files to Create/Modify

| Action | File | Approach |
|--------|------|----------|
| **Create** | `utils/test_time_registers.py` | TTR |
| Modify | `models.py` (ViTRGTSBackbone.forward) | TTR |
| Modify | `configs/baseline_vit_rgts_v2.yaml` | TTR |
| **Create** | `models/gated_attention.py` | Gated |
| Modify | `models.py` (ViTRGTSBackbone.__init__) | Gated |
| Modify | `models.py` (forward_explain) | Gated |
| **Create** | `configs/baseline_vit_rgts_gated.yaml` | Gated |
| **Create** | `models/unet_vit_backbone.py` | U-Net+ViT |
| **Create** | `configs/unet_vit.yaml` | U-Net+ViT |
| Modify | `models.py` (HTRNet.__init__ dispatch) | U-Net+ViT |
| Modify | `scripts/trainer.py` (param groups) | Gated, U-Net |
