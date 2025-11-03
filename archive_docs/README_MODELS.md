# Multi-Architecture HTR System 🚀

A **modular Handwritten Text Recognition (HTR)** system supporting multiple deep learning architectures with automatic model detection and comprehensive attention visualization.

## 🌟 Features

- **Two Model Architectures**: CNN-RNN (ResNet + LSTM) and ViT-B/16 (Vision Transformer)
- **Automatic Model Detection**: Visualization adapts automatically to model type
- **Modular Design**: Factory pattern enables easy architecture extension
- **Comprehensive Attention Visualization**: Architecture-specific attention maps
- **CTC Loss**: Sequence-to-sequence alignment without forced alignment
- **IAM Dataset**: Handwritten text line recognition

---

## 📊 Supported Architectures

### 1. CNN-RNN (Original)
- **Encoder**: ResNet-style CNN with BasicBlocks
- **Decoder**: Bidirectional LSTM/GRU
- **Parameters**: ~7M
- **Best for**: Production, limited compute, smaller datasets

### 2. ViT-B/16 (New)
- **Encoder**: 12-layer Vision Transformer (768 dim, 12 heads)
- **Decoder**: RNN or Linear
- **Parameters**: ~80M
- **Best for**: Research, large datasets, interpretable attention

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone <repo-url>
cd HTR-best-practices

# Activate virtual environment
.venv\Scripts\activate

# Install dependencies (if needed)
pip install -r requirements.txt
```

### Training

#### Train CNN-RNN Model
```bash
python trainer.py config.yaml
```

#### Train ViT-B/16 Model
```bash
python trainer.py config_vit.yaml
```

### Visualization

#### Visualize Test Samples (Batch Mode)
```bash
# CNN-RNN model
python visualize_attention.py --model saved_models/run_1/htrnet.pt --config config.yaml --num_samples 10

# ViT model
python visualize_attention.py --model saved_models/run_2/htrnet.pt --config config_vit.yaml --num_samples 10
```

#### Visualize Specific Image
```bash
python visualize_attention.py \
    --model saved_models/run_2/htrnet.pt \
    --config config_vit.yaml \
    --image datasets/IAM/forms/a01-000u.png
```

---

## 📁 Project Structure

```
HTR-best-practices/
├── models.py                   # CNN-RNN architecture
├── models_vit.py               # ViT-B/16 architecture (NEW)
├── trainer.py                  # Training script (supports both models)
├── visualize_attention.py      # Visualization script (auto-detects model)
├── config.yaml                 # CNN-RNN configuration
├── config_vit.yaml             # ViT configuration (NEW)
│
├── utils/
│   ├── visualizer.py           # Attention visualization framework
│   ├── htr_dataset.py          # Dataset loader
│   ├── metrics.py              # CER/WER metrics
│   ├── preprocessing.py        # Image preprocessing
│   └── transforms.py           # Data augmentation
│
├── data/
│   └── IAM/processed_lines/    # Processed dataset
│
├── experiments/                # Training runs (auto-created)
│   ├── run_1/                  # First training run
│   ├── run_2/                  # Second training run
│   └── ...
│
├── attention_visualizations/   # Visualization outputs (auto-created)
│
└── docs/
    ├── VIT_QUICKSTART.md       # Quick start for ViT
    ├── VIT_INTEGRATION.md      # Complete ViT guide
    ├── IMPLEMENTATION_SUMMARY.md  # System overview
    ├── ARCHITECTURE_DIAGRAM.md    # Visual architecture comparison
    └── IMPLEMENTATION_CHECKLIST.md  # Verification checklist
```

---

## 🎯 Model Comparison

| Feature | CNN-RNN | ViT-B/16 |
|---------|---------|----------|
| **Parameters** | ~7M | ~80M |
| **GPU Memory (bs=8)** | ~3GB | ~10GB |
| **Training Speed** | Fast | 2-3x slower |
| **Convergence** | 20-30 epochs | 40-60 epochs |
| **CER (IAM test)** | 5-7% | 4-6% |
| **Best Use Case** | Production | Research |

---

## 📸 Attention Visualization

### CNN-RNN Model Outputs
For each sample:
- `sample_XXXX_combined.png` - All CNN layers overlaid
- `sample_XXXX_cnn_layer_N.png` - Individual CNN layer attention
- `sample_XXXX_rnn_heatmap.png` - RNN temporal attention
- `sample_XXXX_sequence_attention.png` - Character-level bar chart

### ViT Model Outputs
For each sample:
- `sample_XXXX_transformer_attention.png` - Multi-head self-attention (3 layers)
- `sample_XXXX_attention_rollout.png` - Accumulated attention across all layers

**All visualizations include:**
- Original input image
- Model prediction (decoded text)
- Ground truth (if available)
- Attention heatmaps/overlays

---

## ⚙️ Configuration

### CNN-RNN Configuration (`config.yaml`)
```yaml
preproc:
  image_height: 128
  image_width: 1024

arch:
  backbone: 'resnet'
  decoder: 'lstm'
  # ... other CNN-RNN parameters

data:
  batch_size: 16
  # ... data parameters

train:
  learning_rate: 0.001
  # ... training parameters
```

### ViT Configuration (`config_vit.yaml`)
```yaml
preproc:
  image_height: 128
  image_width: 1024

arch_vit:
  patch_size: 16
  embed_dim: 768
  depth: 12
  num_heads: 12
  decoder_type: 'rnn'
  # ... other ViT parameters

data:
  batch_size: 8  # Lower for ViT

train:
  learning_rate: 0.0001  # Lower for ViT
  # ... training parameters

model_type: 'vit'
```

---

## 🔧 Architecture Details

### CNN-RNN Pipeline
```
Input [B,1,128,1024]
  ↓
ResNet CNN Encoder (4 BasicBlocks)
  ↓
Feature Maps [B,512,8,64]
  ↓
Reshape → [512,B,512]
  ↓
2-layer Bidirectional LSTM
  ↓
Output [128,B,num_classes] → CTC Loss
```

### ViT-B/16 Pipeline
```
Input [B,1,128,1024]
  ↓
Patch Embedding (16×16 patches) → [B,512,768]
  ↓
Positional Embedding
  ↓
12 Transformer Blocks (12 heads, 768 dim)
  ↓
Sequence Decoder (RNN or Linear)
  ↓
Output [512,B,num_classes] → CTC Loss
```

---

## 📖 Documentation

### Quick References
- **`VIT_QUICKSTART.md`** - Get started with ViT in 5 minutes
- **`README_MODELS.md`** - This file (overview)

### Comprehensive Guides
- **`VIT_INTEGRATION.md`** - Complete ViT architecture and usage
- **`IMPLEMENTATION_SUMMARY.md`** - System design and implementation
- **`ARCHITECTURE_DIAGRAM.md`** - Visual architecture comparison

### Technical Details
- **`VISUALIZATION_README.md`** - Attention visualization framework
- **`ARCHITECTURE_VISUALIZATION.md`** - Visualization technical details
- **`IMPLEMENTATION_CHECKLIST.md`** - Verification checklist

---

## 🧪 Testing

### Verify Model Creation
```python
from models import HTRNet
from models_vit import HTRViT
import torch

# CNN-RNN model
cnn_model = HTRNet(config.arch, num_classes=80)
x = torch.randn(2, 1, 128, 1024)
out = cnn_model(x)
print(out.shape)  # [128, 2, 80]

# ViT model
vit_model = HTRViT(patch_size=16, embed_dim=768, depth=12, 
                   num_heads=12, num_classes=80)
out = vit_model(x)
print(out.shape)  # [512, 2, 80]
```

### Verify Factory Pattern
```python
from trainer import create_model
from omegaconf import OmegaConf

# Load CNN-RNN config → creates HTRNet
config = OmegaConf.load('config.yaml')
model = create_model(config, num_classes=80)
print(type(model))  # <class 'models.HTRNet'>

# Load ViT config → creates HTRViT
config_vit = OmegaConf.load('config_vit.yaml')
model_vit = create_model(config_vit, num_classes=80)
print(type(model_vit))  # <class 'models_vit.HTRViT'>
```

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory (OOM)
**Solution**: Reduce batch size in config file
```yaml
data:
  batch_size: 4  # Reduce from 8 or 16
```

### Issue: ViT Training is Slow
**Expected**: ViT is 2-3x slower than CNN-RNN due to higher parameter count and self-attention complexity.

**Optimization tips**:
- Ensure using GPU (`device: 'cuda:0'` in config)
- Reduce `num_workers` if CPU is bottleneck
- Consider gradient accumulation for larger effective batch size

### Issue: Model Not Converging
**For ViT**: Ensure proper warmup and learning rate scheduling
```yaml
train:
  warmup_epochs: 5
  scheduler: 'cosine'
  learning_rate: 0.0001
```

### Issue: Visualization Shows No Attention Patterns
**Solution**: 
- Train for more epochs (ViT needs 40-60 epochs)
- Check if model is learning (CER/WER should decrease)
- For ViT, visualize intermediate layers (not just final)

---

## 🚀 Advanced Usage

### Resume Training
```bash
python trainer.py config_vit.yaml --resume experiments/run_2/htrnet_epoch_10.pt
```

### Custom Visualization Layers (ViT)
Edit `visualize_attention.py`:
```python
layers_to_visualize=[0, 3, 6, 9, 11]  # Visualize 5 layers instead of 3
```

### Mixed Architecture Comparison
Train both models, then compare visualizations:
```bash
# Train both
python trainer.py config.yaml          # CNN-RNN
python trainer.py config_vit.yaml      # ViT

# Visualize same samples from both
python visualize_attention.py --model experiments/run_1/htrnet.pt --config config.yaml --num_samples 5
python visualize_attention.py --model experiments/run_2/htrnet.pt --config config_vit.yaml --num_samples 5
```

---

## 🤝 Contributing

### Adding New Architectures
1. Create new model file (e.g., `models_hybrid.py`)
2. Implement model class with CTC-compatible output `[T, B, C]`
3. Add to factory function in `trainer.py`:
   ```python
   elif model_type == 'hybrid':
       return create_hybrid_model(config, num_classes)
   ```
4. Add visualization method in `utils/visualizer.py` if needed
5. Update `detect_model_type()` in `visualize_attention.py`
6. Create config file (e.g., `config_hybrid.yaml`)

### Code Style
- Follow existing code structure
- Add type hints where possible
- Document all functions with docstrings
- Keep backward compatibility

---

## 📝 License

[Your License Here]

---

## 📚 References

1. **Vision Transformer (ViT)**: Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021
2. **CTC Loss**: Graves et al., "Connectionist Temporal Classification", ICML 2006
3. **IAM Dataset**: Marti & Bunke, "The IAM-database", IJDAR 2002
4. **Attention Rollout**: Abnar & Zuidema, "Quantifying Attention Flow in Transformers", ACL 2020

---

## 🙏 Acknowledgments

- Original CNN-RNN architecture based on HTR best practices
- ViT architecture adapted from "An Image is Worth 16x16 Words"
- Attention visualization inspired by various interpretability research

---

## 📞 Contact

[Your Contact Information]

---

**Last Updated**: [Current Date]

**Version**: 2.0 (Multi-Architecture Support)

---

## Quick Command Reference

| Task | Command |
|------|---------|
| Train CNN-RNN | `python trainer.py config.yaml` |
| Train ViT | `python trainer.py config_vit.yaml` |
| Visualize (batch) | `python visualize_attention.py --model MODEL --config CONFIG --num_samples 10` |
| Visualize (single) | `python visualize_attention.py --model MODEL --config CONFIG --image IMAGE` |
| Resume training | `python trainer.py CONFIG --resume CHECKPOINT` |

---

✨ **The system now supports multi-architecture experimentation with full modularity!** ✨
