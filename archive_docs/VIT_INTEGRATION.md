# Vision Transformer (ViT) for Handwritten Text Recognition

## Overview

This codebase now supports **two model architectures** for HTR:

1. **CNN-RNN** (Original): ResNet-based CNN encoder + LSTM/GRU decoder
2. **ViT-B/16** (New): Vision Transformer with 16x16 patches + RNN/Linear decoder

Both architectures use **CTC loss** for sequence-to-sequence alignment and support **modular attention visualization**.

---

## Architecture: ViT-B/16

### Components

#### 1. **Patch Embedding**
- Splits input image (128×1024) into **16×16 patches**
- Creates **8×64 = 512 patches** per image
- Projects each patch to **768-dimensional embedding**
- Adds learnable **positional embeddings**

#### 2. **Transformer Encoder**
- **12 Transformer blocks** (ViT-Base depth)
- Each block contains:
  - **Multi-Head Self-Attention** (12 heads, 64 dim per head)
  - **MLP** (feed-forward network, expansion ratio 4.0)
  - **Layer Normalization** (pre-norm architecture)
  - **Residual connections**
- Stores attention weights for visualization

#### 3. **Sequence Decoder**
- **RNN decoder** (default): 2-layer LSTM with 512 hidden units
- **Linear decoder** (alternative): Direct projection from transformer output
- Output: **[T=512, B, num_classes]** for CTC loss

### Model Statistics
- **Parameters**: ~80M (ViT-B/16 with RNN decoder)
- **Input**: [B, 1, 128, 1024] grayscale images
- **Output**: [512, B, num_classes] logits for CTC

---

## Configuration: `config_vit.yaml`

```yaml
# Preprocessing
preproc:
  image_height: 128
  image_width: 1024

# ViT Architecture
arch_vit:
  patch_size: 16          # 16x16 patches
  embed_dim: 768          # ViT-Base embedding dimension
  depth: 12               # Number of transformer blocks
  num_heads: 12           # Multi-head attention heads
  mlp_ratio: 4.0          # MLP expansion ratio
  dropout: 0.1            # Dropout rate
  decoder_type: 'rnn'     # 'rnn' or 'linear'
  decoder_hidden: 512     # RNN hidden size
  decoder_layers: 2       # RNN layers

# Dataset
data:
  path: ./data/IAM/processed_lines
  batch_size: 8           # Lower for ViT (higher memory)
  num_workers: 4

# Training
train:
  epochs: 50
  learning_rate: 0.0001   # Lower LR for ViT
  weight_decay: 0.01      # Weight decay for regularization
  scheduler: 'cosine'     # Learning rate scheduler
  warmup_epochs: 5        # Warmup period
  gradient_clip: 5.0
  save_interval: 5
  log_interval: 10

# Model type identifier
model_type: 'vit'
```

---

## Training ViT Model

### 1. **Train from Scratch**
```bash
python trainer.py config_vit.yaml
```

### 2. **Resume Training**
```bash
python trainer.py config_vit.yaml --resume saved_models/run_2/htrnet_epoch_10.pt
```

### 3. **Expected Training Characteristics**
- **Memory Usage**: ~10-12GB GPU (batch_size=8)
- **Training Speed**: Slower than CNN-RNN (~2-3x longer per epoch)
- **Convergence**: May require more epochs (50+ recommended)
- **Regularization**: Benefits from weight decay and dropout

---

## Attention Visualization

The visualization system **automatically detects** model type and applies appropriate visualization.

### ViT-Specific Visualizations

#### 1. **Multi-Head Self-Attention**
- Shows attention patterns from **selected transformer layers**
- Default: Layer 0 (early), Layer 6 (middle), Layer 11 (final)
- Visualizes how patches attend to each other
- Shape: `[num_heads, num_patches, num_patches]`

#### 2. **Attention Rollout**
- Combines attention across **all 12 transformer layers**
- Shows end-to-end information flow
- Computed via recursive attention multiplication
- Results in single attention map highlighting important regions

### Usage Examples

#### Visualize ViT Model (Batch Mode)
```bash
python visualize_attention.py \
    --model saved_models/run_vit/htrnet_epoch_50.pt \
    --config config_vit.yaml \
    --num_samples 10 \
    --output_dir attention_vit
```

#### Visualize Specific Image (ViT)
```bash
python visualize_attention.py \
    --model saved_models/run_vit/htrnet_epoch_50.pt \
    --config config_vit.yaml \
    --image datasets/IAM/forms/a01-000u.png \
    --output_dir attention_vit_custom
```

### Output Files (ViT)
For each sample:
- `sample_0000_transformer_attention.png` - Multi-layer self-attention
- `sample_0000_attention_rollout.png` - Accumulated attention
- Both show: input image, prediction, ground truth (if available)

---

## Model Comparison: CNN-RNN vs ViT

| Aspect | CNN-RNN | ViT-B/16 |
|--------|---------|----------|
| **Parameters** | ~7M | ~80M |
| **Architecture** | ResNet + LSTM | Transformer |
| **Input** | 128×1024 image | 512 patches (16×16) |
| **Memory** | ~4-6GB (batch=16) | ~10-12GB (batch=8) |
| **Training Speed** | Fast | Slower (2-3x) |
| **Inductive Bias** | Strong (CNN locality) | Weak (learns from data) |
| **Data Efficiency** | Better for small datasets | Requires more data |
| **Attention** | CNN spatial + RNN temporal | Multi-head self-attention |
| **Visualization** | CNN feature maps + RNN states | Transformer attention weights |

### When to Use Each?

#### Use **CNN-RNN** when:
- Limited computational resources
- Small to medium datasets (<100k samples)
- Need faster training/inference
- Baseline model for comparison
- Strong inductive bias is beneficial

#### Use **ViT** when:
- Large datasets (100k+ samples)
- Sufficient GPU memory (12GB+)
- Want to explore transformer architectures
- Willing to train longer
- Need interpretable attention patterns

---

## Code Structure

### Files Modified/Created

#### New Files
- `models_vit.py` - ViT architecture implementation (430+ lines)
  - `PatchEmbedding` - Image to patch sequence
  - `MultiHeadSelfAttention` - Attention with weight storage
  - `TransformerBlock` - Single transformer layer
  - `ViTEncoder` - Stack of transformer blocks
  - `SequenceDecoder` - RNN/Linear decoder
  - `HTRViT` - Main ViT model class
  - `create_vit_htr_model()` - Factory function

- `config_vit.yaml` - ViT-specific configuration

#### Modified Files
- `trainer.py`
  - Added `create_model()` factory function
  - Supports both CNN-RNN and ViT instantiation
  - Automatically detects model type from config

- `visualize_attention.py`
  - Added `detect_model_type()` function
  - Conditional visualization logic (CNN-RNN vs ViT)
  - Imports from both `models.py` and `models_vit.py`

- `utils/visualizer.py`
  - Added `visualize_transformer_attention()` method
  - Added `visualize_attention_rollout()` method
  - Supports transformer attention weights `[B, H, N, N]`

---

## Implementation Details

### Attention Weight Storage
```python
class MultiHeadSelfAttention(nn.Module):
    def forward(self, x):
        # ... compute attention ...
        # Store for visualization
        self.attention_weights = attn_weights
        return out
```

### Attention Extraction
```python
# In HTRViT class
def get_attention_weights(self):
    """Extract attention weights from all transformer layers."""
    attention_list = []
    for block in self.encoder.blocks:
        if hasattr(block.attn, 'attention_weights'):
            attention_list.append(block.attn.attention_weights)
    return attention_list
```

### Model Type Detection
```python
def detect_model_type(model):
    """Detect if model is ViT or CNN-RNN."""
    if isinstance(model, HTRViT):
        return 'vit'
    elif hasattr(model, 'get_attention_weights'):
        return 'vit'
    else:
        return 'cnn-rnn'
```

---

## Testing

### Verify ViT Model Creation
```python
from models_vit import HTRViT

model = HTRViT(
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12,
    num_classes=80,
    decoder_type='rnn'
)

# Test forward pass
x = torch.randn(2, 1, 128, 1024)  # [B, C, H, W]
output = model(x)  # [T, B, num_classes]
print(output.shape)  # Should be [512, 2, 80]
```

### Verify Model Loading
```python
from trainer import create_model
from omegaconf import OmegaConf

config = OmegaConf.load('config_vit.yaml')
model = create_model(config, num_classes=80)
print(type(model))  # Should be HTRViT
```

---

## Troubleshooting

### Issue: Out of Memory (OOM)
**Solution**: Reduce batch size in `config_vit.yaml`
```yaml
data:
  batch_size: 4  # Reduce from 8
```

### Issue: Slow Training
**Solution**: This is expected for ViT. Optimize:
- Use mixed precision training (add to trainer)
- Reduce `num_workers` if CPU is bottleneck
- Use gradient accumulation for larger effective batch size

### Issue: Poor Initial Performance
**Solution**: Ensure proper warmup and LR scheduling
```yaml
train:
  warmup_epochs: 5
  scheduler: 'cosine'
```

### Issue: Attention Visualization Shows No Patterns
**Solution**: 
- Train for more epochs (ViT needs longer)
- Check if model is actually learning (CER/WER decreasing)
- Try visualizing intermediate layers, not just final layer

---

## Future Enhancements

### Potential Improvements
1. **Mixed Precision Training**: Add AMP support for faster training
2. **Larger ViT Variants**: ViT-L/16, ViT-H/14 for larger datasets
3. **Pre-training**: Use pre-trained ViT weights from ImageNet
4. **Hybrid Models**: Combine CNN feature extraction + Transformer encoder
5. **Decoder Improvements**: Try Transformer decoder instead of RNN
6. **Data Augmentation**: Add ViT-specific augmentations (RandAugment, etc.)

---

## References

1. **Vision Transformer Paper**: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2021)
2. **Attention Rollout**: "Quantifying Attention Flow in Transformers" (Abnar & Zuidema, 2020)
3. **CTC Loss**: "Connectionist Temporal Classification" (Graves et al., 2006)

---

## Summary

✅ **Implemented**: Complete ViT-B/16 architecture with RNN/Linear decoder  
✅ **Configured**: ViT-specific training configuration  
✅ **Integrated**: Model factory pattern in trainer  
✅ **Visualized**: Transformer attention (multi-head + rollout)  
✅ **Maintained**: Full modularity and backward compatibility with CNN-RNN  

The codebase now supports **multi-architecture experimentation** while keeping code clean and modular! 🚀
