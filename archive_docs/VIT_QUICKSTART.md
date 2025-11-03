# Quick Start: ViT-B/16 for HTR

## 🚀 Train ViT Model

```bash
# Activate virtual environment
.venv\Scripts\activate

# Train ViT from scratch
python trainer.py config_vit.yaml
```

## 📊 Visualize ViT Attention

```bash
# Batch visualization (10 test samples)
python visualize_attention.py --model saved_models/run_1/htrnet_epoch_50.pt --config config_vit.yaml --num_samples 10

# Single image visualization
python visualize_attention.py --model saved_models/run_1/htrnet_epoch_50.pt --config config_vit.yaml --image datasets/IAM/forms/a01-000u.png
```

## 📁 Output Structure

```
experiments/
└── run_1/                              # ViT training run
    ├── config.yaml                     # Copy of config_vit.yaml
    ├── htrnet_epoch_5.pt               # Model checkpoint
    ├── htrnet_epoch_10.pt
    ├── ...
    └── htrnet_epoch_50.pt              # Final model

attention_visualizations/                # ViT attention outputs
├── sample_0000_transformer_attention.png
├── sample_0000_attention_rollout.png
├── sample_0001_transformer_attention.png
├── sample_0001_attention_rollout.png
└── ...
```

## 🔍 What Gets Visualized?

### ViT Model (detected automatically)
1. **Transformer Attention**: Multi-head self-attention from 3 layers (early/mid/late)
2. **Attention Rollout**: Accumulated attention across all 12 layers

### CNN-RNN Model (detected automatically)
1. **CNN Spatial Attention**: Feature maps from convolutional layers
2. **RNN Temporal Attention**: Activation heatmaps from LSTM/GRU
3. **Sequence Attention**: Bar chart of attention over decoded sequence

## ⚙️ Key Configuration Differences

| Parameter | CNN-RNN (`config.yaml`) | ViT (`config_vit.yaml`) |
|-----------|-------------------------|-------------------------|
| Batch Size | 16 | 8 (higher memory) |
| Learning Rate | 0.001 | 0.0001 (lower) |
| Architecture | `arch` section | `arch_vit` section |
| Parameters | ~7M | ~80M |
| Training Time | 1x (baseline) | 2-3x slower |

## 📖 Full Documentation

- **`VIT_INTEGRATION.md`** - Complete ViT architecture and usage guide
- **`VISUALIZATION_README.md`** - Attention visualization system overview
- **`ARCHITECTURE_VISUALIZATION.md`** - Technical details on visualization

## 🛠️ Switching Between Models

The system automatically detects model type based on:
1. Config file (`model_type: 'vit'` or `arch_vit` section)
2. Model class (`HTRViT` vs `HTRNet`)

**No code changes needed** - just use the appropriate config file! ✨

---

**That's it!** The codebase now supports both architectures with automatic model detection and appropriate visualization. 🎉
