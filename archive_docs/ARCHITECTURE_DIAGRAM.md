# Architecture Comparison: CNN-RNN vs ViT-B/16

## Side-by-Side Comparison

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          INPUT IMAGE [1, 128, 1024]                      │
│                     (Grayscale handwritten text line)                    │
└───────────────────────────┬──────────────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
                v                       v
    ┌──────────────────────┐  ┌──────────────────────────┐
    │     CNN-RNN MODEL    │  │      ViT-B/16 MODEL      │
    │      (~7M params)    │  │      (~80M params)       │
    └──────────────────────┘  └──────────────────────────┘


════════════════════════════════════════════════════════════════════════════
                            CNN-RNN ARCHITECTURE
════════════════════════════════════════════════════════════════════════════

Input: [B, 1, 128, 1024]
         │
         ▼
┌─────────────────────┐
│   CNN Encoder       │
│   (ResNet-style)    │
├─────────────────────┤
│ • BasicBlock 1      │  [B, 64, 64, 512]
│ • BasicBlock 2      │  [B, 128, 32, 256]
│ • BasicBlock 3      │  [B, 256, 16, 128]
│ • BasicBlock 4      │  [B, 512, 8, 64]
└─────────────────────┘
         │
         ▼
   [B, 512, 8, 64]
         │
         ▼ (reshape)
   [B, 512, 512]  (8 × 64 = 512 positions)
         │
         ▼ (transpose to [512, B, 512])
┌─────────────────────┐
│   RNN Decoder       │
│   (LSTM/GRU)        │
├─────────────────────┤
│ • 2-layer LSTM      │
│ • Hidden: 512       │
│ • Bidirectional     │
└─────────────────────┘
         │
         ▼
   [128, B, 1024]  (forward + backward)
         │
         ▼
┌─────────────────────┐
│   Linear Head       │
│   (1024 → classes)  │
└─────────────────────┘
         │
         ▼
   [128, B, 80]  ← OUTPUT (for CTC loss)


════════════════════════════════════════════════════════════════════════════
                             ViT-B/16 ARCHITECTURE
════════════════════════════════════════════════════════════════════════════

Input: [B, 1, 128, 1024]
         │
         ▼
┌─────────────────────────┐
│   Patch Embedding       │
│   (16×16 patches)       │
├─────────────────────────┤
│ • Conv2d(1→768, k=16,   │
│          stride=16)     │
│ • Flatten spatial dims  │
│ • Add positional embed  │
└─────────────────────────┘
         │
         ▼
   [B, 512, 768]  (512 patches, 768-dim)
         │
         ▼
┌─────────────────────────────────────────┐
│         Transformer Encoder             │
│            (12 blocks)                  │
├─────────────────────────────────────────┤
│  Block 1:                               │
│    ┌───────────────────────────┐       │
│    │ Layer Norm                │       │
│    │         ↓                 │       │
│    │ Multi-Head Self-Attention │       │
│    │   (12 heads × 64 dim)     │       │
│    │         ↓                 │       │
│    │ Residual Connection       │       │
│    │         ↓                 │       │
│    │ Layer Norm                │       │
│    │         ↓                 │       │
│    │ MLP (768 → 3072 → 768)    │       │
│    │         ↓                 │       │
│    │ Residual Connection       │       │
│    └───────────────────────────┘       │
│                                         │
│  Blocks 2-12: (same structure)          │
└─────────────────────────────────────────┘
         │
         ▼
   [B, 512, 768]
         │
         ▼ (transpose to [512, B, 768])
┌─────────────────────────┐
│   Sequence Decoder      │
│   (RNN or Linear)       │
├─────────────────────────┤
│ Option A: RNN           │
│   • 2-layer LSTM        │
│   • Hidden: 512         │
│                         │
│ Option B: Linear        │
│   • Direct projection   │
└─────────────────────────┘
         │
         ▼
   [512, B, 80]  ← OUTPUT (for CTC loss)


════════════════════════════════════════════════════════════════════════════
                         ATTENTION MECHANISMS
════════════════════════════════════════════════════════════════════════════

┌──────────────────────────┐          ┌──────────────────────────┐
│   CNN-RNN ATTENTION      │          │    ViT ATTENTION         │
├──────────────────────────┤          ├──────────────────────────┤
│                          │          │                          │
│ 1. CNN Spatial:          │          │ 1. Self-Attention:       │
│    • Feature maps        │          │    • Patch-to-patch      │
│    • [B, C, H, W]        │          │    • [B, H, N, N]        │
│    • Multiple layers     │          │    • Multi-head (12)     │
│    • Channel averaging   │          │    • Stored per block    │
│                          │          │                          │
│ 2. RNN Temporal:         │          │ 2. Attention Rollout:    │
│    • Hidden states       │          │    • Accumulated across  │
│    • [T, B, C]           │          │      all 12 layers       │
│    • Magnitude as weight │          │    • Shows info flow     │
│    • Heatmap over time   │          │    • End-to-end pattern  │
│                          │          │                          │
│ 3. Sequence Attention:   │          │ 3. Layer-wise View:      │
│    • Character-level     │          │    • Early layers (low)  │
│    • Bar chart           │          │    • Mid layers (medium) │
│    • Per-timestep        │          │    • Late layers (high)  │
└──────────────────────────┘          └──────────────────────────┘


════════════════════════════════════════════════════════════════════════════
                       COMPUTATIONAL COMPARISON
════════════════════════════════════════════════════════════════════════════

┌─────────────────┬───────────────────┬─────────────────────────┐
│   Metric        │     CNN-RNN       │       ViT-B/16          │
├─────────────────┼───────────────────┼─────────────────────────┤
│ Parameters      │   ~7M             │   ~80M                  │
├─────────────────┼───────────────────┼─────────────────────────┤
│ FLOPs/image     │   ~5G             │   ~25G                  │
├─────────────────┼───────────────────┼─────────────────────────┤
│ GPU Memory      │   ~4GB (bs=16)    │   ~10GB (bs=8)          │
├─────────────────┼───────────────────┼─────────────────────────┤
│ Training Speed  │   1x (baseline)   │   2-3x slower           │
├─────────────────┼───────────────────┼─────────────────────────┤
│ Inference Time  │   ~100ms          │   ~250ms                │
├─────────────────┼───────────────────┼─────────────────────────┤
│ Convergence     │   20-30 epochs    │   40-60 epochs          │
├─────────────────┼───────────────────┼─────────────────────────┤
│ CER (IAM test)  │   5-7%            │   4-6% (w/ training)    │
└─────────────────┴───────────────────┴─────────────────────────┘


════════════════════════════════════════════════════════════════════════════
                         ATTENTION VISUALIZATION
════════════════════════════════════════════════════════════════════════════

CNN-RNN Output:
    sample_0000_combined.png            ← All CNN layers overlaid
    sample_0000_cnn_layer_0.png         ← Layer 0 spatial attention
    sample_0000_cnn_layer_1.png         ← Layer 1 spatial attention
    sample_0000_rnn_heatmap.png         ← RNN temporal attention
    sample_0000_sequence_attention.png  ← Character-level bar chart

ViT Output:
    sample_0000_transformer_attention.png  ← Multi-head from 3 layers
    sample_0000_attention_rollout.png      ← Accumulated attention


════════════════════════════════════════════════════════════════════════════
                              RECEPTIVE FIELD
════════════════════════════════════════════════════════════════════════════

CNN-RNN:
    ┌───────────────────────────────────┐
    │  Local → Hierarchical → Global    │
    ├───────────────────────────────────┤
    │  Conv1: 3×3 (local)               │
    │  Conv2: 5×5 (small region)        │
    │  Conv3: 11×11 (medium region)     │
    │  Conv4: 23×23 (larger region)     │
    │  RNN: Full sequence (global)      │
    └───────────────────────────────────┘
    • Strong local inductive bias
    • Hierarchical feature extraction
    • Global context via RNN

ViT:
    ┌───────────────────────────────────┐
    │  Global from Layer 1              │
    ├───────────────────────────────────┤
    │  Each patch attends to ALL other  │
    │  patches from the first layer.    │
    │                                   │
    │  Self-attention = O(N²) where N   │
    │  is number of patches (512).      │
    │                                   │
    │  No built-in locality bias -      │
    │  learns patterns from data.       │
    └───────────────────────────────────┘
    • Global receptive field from start
    • Learns inductive bias from data
    • More flexible but data-hungry


════════════════════════════════════════════════════════════════════════════
                           TRADE-OFFS SUMMARY
════════════════════════════════════════════════════════════════════════════

Use CNN-RNN when:                    Use ViT when:
  ✓ Limited compute (4-6GB GPU)        ✓ Large datasets (100k+ samples)
  ✓ Small datasets (IAM ~13k)          ✓ Sufficient GPU memory (12GB+)
  ✓ Fast training needed               ✓ Research/experimentation
  ✓ Production deployment              ✓ Interpretable attention needed
  ✓ Strong inductive bias helps        ✓ Exploring transformers for HTR
  ✓ Baseline model                     ✓ Potential for transfer learning

```

## Key Architectural Differences

### 1. Feature Extraction
- **CNN-RNN**: Hierarchical feature extraction (local → global)
- **ViT**: Direct patch embedding with global attention

### 2. Sequence Modeling
- **CNN-RNN**: RNN (LSTM/GRU) models temporal dependencies
- **ViT**: Self-attention captures dependencies between all patches

### 3. Positional Information
- **CNN-RNN**: Implicit through convolutional structure
- **ViT**: Explicit through learned positional embeddings

### 4. Attention Mechanism
- **CNN-RNN**: Post-hoc visualization of activations
- **ViT**: Native self-attention weights stored during forward pass

### 5. Parameter Distribution
- **CNN-RNN**: Most parameters in CNN encoder + RNN decoder
- **ViT**: Most parameters in transformer blocks (12 × ~6.5M)

---

## Visualization Comparison

### CNN-RNN: Multi-Stage Attention
```
Input Image
    ↓
CNN Layer 1 → Spatial Attention (low-level edges)
CNN Layer 2 → Spatial Attention (textures)
CNN Layer 3 → Spatial Attention (strokes)
CNN Layer 4 → Spatial Attention (characters)
    ↓
RNN States → Temporal Attention (sequence flow)
    ↓
Decoded Text → Character-wise Attention (bar chart)
```

### ViT: Self-Attention Patterns
```
Input Patches (512 patches)
    ↓
Layer 1  → Self-Attention (local patterns)
Layer 6  → Self-Attention (medium-range dependencies)
Layer 12 → Self-Attention (global context)
    ↓
Attention Rollout → End-to-end information flow
    ↓
Decoded Text
```

---

## Code Integration Points

```python
# trainer.py
def create_model(config, num_classes):
    model_type = getattr(config, 'model_type', 'cnn-rnn')
    if model_type == 'vit':
        return create_vit_htr_model(config, num_classes)  # ViT
    else:
        return HTRNet(config.arch, num_classes)           # CNN-RNN

# visualize_attention.py
def detect_model_type(model):
    if isinstance(model, HTRViT):
        return 'vit'
    else:
        return 'cnn-rnn'

# Conditional visualization
if model_type == 'vit':
    attention_weights = model.get_attention_weights()
    visualizer.visualize_transformer_attention(...)
    visualizer.visualize_attention_rollout(...)
else:
    with AttentionHook(model, visualizer):
        visualizer.visualize_and_save(...)
```

---

This diagram provides a complete visual reference for understanding the architectural differences and design decisions! 📊
