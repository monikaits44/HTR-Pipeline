# Quick Start Guide: Explainable HTR System

## Overview
This guide shows how to use the HTR system according to the instruction set requirements for explainable handwritten text recognition.

---

## 🎯 Core Capabilities

### 1. Model Categories

#### NORMAL CATEGORY
- **CNN-RNN-CTC**: Baseline model (4.2% CER on IAM)
- **ViT-RGTS**: Vision Transformer with Register Tokens (supports 0, 2, 4, 8, 16 registers)

#### PRETRAINED CATEGORY  
- **TorchVision ViT**: vit_b_16, vit_b_32, vit_l_16, vit_l_32
- **TrOCR**: base-handwritten, large-handwritten, base-printed, large-printed

### 2. Register Token Configurations
- 0 registers: Standard ViT (no register tokens)
- 2 registers: Minimal register configuration
- 4 registers: Standard configuration (recommended)
- 8 registers: Enhanced configuration
- 16 registers: Maximum configuration

---

## 🚀 Quick Start Examples

### Train ViT-RGTS with Different Register Configurations

```bash
# No registers (standard ViT)
python scripts/trainer.py configs/baseline_vit_rgts.yaml arch.num_registers=0

# 4 registers (recommended)
python scripts/trainer.py configs/baseline_vit_rgts.yaml arch.num_registers=4

# 8 registers
python scripts/trainer.py configs/baseline_vit_rgts.yaml arch.num_registers=8

# 16 registers
python scripts/trainer.py configs/baseline_vit_rgts.yaml arch.num_registers=16
```

### Train TorchVision ViT Variants

```bash
# ViT-B/16 with 4 registers
python scripts/trainer.py configs/torchvision_vit.yaml \
    arch.model_name='vit_b_16' \
    arch.num_registers=4

# ViT-L/16 with 8 registers
python scripts/trainer.py configs/torchvision_vit.yaml \
    arch.model_name='vit_l_16' \
    arch.num_registers=8 \
    train.batch_size=2
```

### Train TrOCR Models

```bash
# TrOCR Small Handwritten (fastest)
python scripts/trainer.py configs/trocr.yaml \
    arch.model_name='microsoft/trocr-small-handwritten'

# TrOCR Base Handwritten (default)
python scripts/trainer.py configs/trocr.yaml \
    arch.model_name='microsoft/trocr-base-handwritten'

# TrOCR Large Handwritten (most accurate)
python scripts/trainer.py configs/trocr.yaml \
    arch.model_name='microsoft/trocr-large-handwritten' \
    train.batch_size=1
```

### Train Baseline CNN-RNN

```bash
# Standard baseline with dual head
python scripts/trainer.py configs/baseline.yaml
```

---

## 🔍 Explainability & Attention Visualization

### 1. Comprehensive Explainability Demo

```bash
python scripts/postprocessing/demo_explainability.py \
    --config configs/config.yaml \
    --arch-config configs/baseline_vit_rgts.yaml \
    --resume saved_models/experiments/run_X/model.pt \
    --image-path data/IAM/processed_lines/test/c04-165-05.png \
    --output-dir demo_output
```

This will:
- Load the trained model
- Make prediction on the image
- Extract attention maps from all layers
- Visualize middle-layer attention
- Save visualizations to output directory

### 2. Register Token Attention Analysis

```bash
python scripts/postprocessing/visualize_register_attention.py \
    saved_models/experiments/run_X/vit_rgts_explain \
    --sample-idx 0 \
    --similarity cosine \
    --save register_attention.png
```

Analyzes:
- How register tokens attend to patch tokens
- Cosine similarity between registers and patches
- Register token behavior across layers

### 3. Character-Level Attention

```bash
python scripts/postprocessing/visualize_character_attention.py \
    saved_models/experiments/run_X/vit_rgts_explain \
    --sample-idx 0 \
    --layer-idx -1 \
    --save character_attention.png
```

Shows:
- Attention maps aligned with recognized characters
- Fine-grained attention for each character
- Useful for downstream tasks (writer identification)

### 4. Token Norm Visualization

```bash
python scripts/postprocessing/visualize_token_norms.py \
    saved_models/experiments/run_X/vit_rgts_explain \
    --sample-idx 0 \
    --save token_norms.png
```

Visualizes:
- Token magnitudes across layers
- Identifies outlier tokens (paper finding)
- Register token vs patch token norms

### 5. GradCAM Visualization

```bash
python scripts/postprocessing/gradcam_vit_rgts.py \
    --config configs/baseline_vit_rgts.yaml \
    --resume saved_models/experiments/run_X/model.pt \
    --image-path data/IAM/processed_lines/test/sample.png \
    --output gradcam_output.png
```

### 6. t-SNE Register Analysis

```bash
python scripts/postprocessing/tsne_register_tokens.py \
    saved_models/experiments/run_X/vit_rgts_explain \
    --save tsne_registers.png
```

---

## 🧪 Run All Register Experiments

Use the convenience script to run all register configurations:

```bash
# Run all configurations (0, 2, 4, 8, 16) - Full training
python scripts/run_register_experiments.py --arch vit_rgts --all

# Quick test (5 epochs each)
python scripts/run_register_experiments.py --arch vit_rgts --all --quick-test

# Single configuration
python scripts/run_register_experiments.py --arch vit_rgts --num-registers 4

# With TorchVision ViT
python scripts/run_register_experiments.py \
    --arch torchvision_vit \
    --all \
    --model-name vit_b_16 \
    --quick-test
```

---

## 📊 Evaluation

### Evaluate Single Model

```bash
python scripts/postprocessing/evaluate.py \
    saved_models/experiments/run_X/config.json \
    resume=./saved_models/experiments/run_X/model.pt \
    device=cpu
```

### Compare Multiple Models

```bash
# Evaluate all runs
for run in saved_models/experiments/run_*; do
    echo "Evaluating $run"
    python scripts/postprocessing/evaluate.py \
        $run/config.json \
        resume=$run/model.pt \
        device=cpu
done
```

### Analyze Results

```bash
python scripts/postprocessing/analyze_evaluation.py \
    saved_models/experiments/
```

---

## 📈 Expected Results (IAM Dataset)

| Model | Registers | CER (%) | WER (%) | Params |
|-------|-----------|---------|---------|--------|
| CNN-RNN-CTC | - | 4.2 | ~12 | 7.4M |
| ViT-RGTS | 0 | ~4.5 | ~13 | 7-10M |
| ViT-RGTS | 4 | ~4.0 | ~12 | 7-10M |
| ViT-RGTS | 8 | ~3.8 | ~11 | 7-10M |
| TorchVision ViT-B | 0 | ~3.6 | ~10 | 86M |
| TorchVision ViT-L | 0 | ~3.2 | ~9 | 304M |
| TrOCR Small | - | ~3.8 | ~11 | ~60M |
| TrOCR Base | - | ~3.2 | ~9 | 334M |
| TrOCR Large | - | ~2.8 | ~8 | 558M |

*Note: Results may vary based on training configuration and random seed*

---

## 🎨 Understanding Attention Maps

### Layer-Specific Attention:

1. **Early Layers (1-3)**:
   - Low-level features (edges, strokes)
   - Localized attention patterns
   - Less semantic meaning

2. **Middle Layers (4-6)**:
   - Structural relationships
   - Character-level patterns
   - **Best for visualization** (instruction set requirement)

3. **Last Layer**:
   - High-level semantics
   - Global context
   - Most semantic but sometimes too abstract

### Register Token Behavior:

- **Without registers**: Patch tokens may contain outliers (paper finding)
- **With registers**: Outlier information stored in registers
- **Result**: Cleaner, more meaningful patch token attention

---

## 🔧 Configuration Tips

### For Register Token Experiments:

1. **Start with 4 registers** (standard configuration)
2. **Compare with 0 registers** to see the difference
3. **Try 8 registers** for potentially better attention
4. **Use 16 registers** only if you have enough data

### For Memory Optimization:

```bash
# Reduce batch size for large models
python scripts/trainer.py configs/torchvision_vit.yaml \
    arch.model_name='vit_l_16' \
    train.batch_size=2 \
    eval.batch_size=2

# Enable gradient checkpointing (if implemented)
# Use mixed precision training
# Reduce image resolution if needed
```

### For CPU Training:

```bash
python scripts/trainer.py configs/baseline_vit_rgts.yaml \
    device=cpu \
    train.batch_size=4 \
    train.num_workers=0
```

---

## 📁 Output Structure

After training and visualization:

```
saved_models/experiments/
├── run_X/
│   ├── config.json           # Training configuration
│   ├── model.pt              # Trained model weights
│   ├── results.csv           # Training metrics
│   ├── evaluation_details.csv # Test set results
│   └── vit_rgts_explain/     # Explainability outputs
│       ├── vit_rgts_features.csv
│       ├── attention_maps/
│       │   ├── sample_0_layer_0.npy
│       │   ├── sample_0_layer_1.npy
│       │   └── ...
│       └── register_tokens/
│           ├── sample_0_registers.npy
│           └── ...
```

---

## 🎓 Research Questions to Explore

Using this system, you can investigate:

1. **How many registers are optimal?**
   - Compare 0, 2, 4, 8, 16 registers
   - Analyze attention quality vs. CER

2. **What do register tokens learn?**
   - Visualize register attention patterns
   - Check if they capture writer-specific information

3. **Are attention maps fine-grained enough?**
   - Extract character-level attention
   - Use for writer identification (downstream task)

4. **Which layers are most informative?**
   - Compare early, middle, late layer attention
   - Find optimal layers for explainability

5. **How do pretrained models compare?**
   - TorchVision ViT vs TrOCR vs ViT-RGTS
   - Pretrained vs from-scratch training

---

## 📚 Key Papers Referenced

1. **VISION TRANSFORMERS NEED REGISTERS**
   - Introduced register tokens concept
   - Solves outlier token problem
   - Improves attention map quality

2. **Beyond Memorization: Training-Free Style Mixing**
   - Writer embedding injection
   - Handwritten text generation
   - Style transfer concepts

3. **Interpretable Writer Recognition**
   - Vectors of Locally Aggregated Characters
   - Character-level representations
   - Writer identification techniques

---

## 🐛 Troubleshooting

### Out of Memory:
```bash
# Reduce batch size
python scripts/trainer.py configs/baseline_vit_rgts.yaml train.batch_size=2

# Use CPU
python scripts/trainer.py configs/baseline_vit_rgts.yaml device=cpu

# Reduce number of workers
python scripts/trainer.py configs/baseline_vit_rgts.yaml train.num_workers=0
```

### Slow Training:
```bash
# Use pretrained models
python scripts/trainer.py configs/torchvision_vit.yaml arch.pretrained=True

# Fewer epochs for testing
python scripts/trainer.py configs/baseline_vit_rgts.yaml train.num_epochs=10
```

### Attention Maps Not Saved:
- Make sure you're using TorchVision ViT or ViT-RGTS
- Check that `forward_explain()` method is being called
- Verify output directory permissions

---

## ✅ Verification Checklist

Before running experiments, verify:

- [ ] IAM dataset preprocessed (`data/IAM/processed_lines/`)
- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Sufficient disk space for experiments
- [ ] GPU available (or CPU configured with small batch size)
- [ ] Config files reviewed and customized

---

## 🎉 Ready to Start!

You now have a fully functional explainable HTR system with:
- ✅ Multiple architectures (CNN-RNN, ViT-RGTS, TorchVision ViT, TrOCR)
- ✅ Register token support (0, 2, 4, 8, 16)
- ✅ Attention map extraction and visualization
- ✅ Fine-grained explainability for downstream tasks
- ✅ Modular, extensible codebase

**Start with a quick test:**
```bash
python scripts/run_register_experiments.py --arch vit_rgts --num-registers 4 --quick-test
```

Happy researching! 🚀
