# visualize_attention.py - Quick I/O Reference

## 📥 INPUTS

### Required (2)
```bash
--model     Path to trained model (.pt file)
--config    Path to configuration (.yaml file)
```

### Optional (5)
```bash
--num_samples    How many samples to visualize (default: 10)
--output_dir     Where to save outputs (default: ./attention_visualizations)
--dataset        Which split: train/val/test (default: test)
--device         cuda:0, cuda:1, or cpu (default: cuda:0)
--image          Single image path (overrides dataset sampling)
```

---

## 📤 OUTPUTS

### For EACH sample → 4 PNG files:

```
sample_0001_combined.png          ← All attention layers stacked
sample_0001_cnn_features.png      ← Spatial attention from CNN
sample_0001_rnn_features.png      ← Temporal attention from RNN
sample_0001_sequence_attention.png ← Bar chart of attention weights
```

### Example with 3 samples:
```
attention_visualizations/
├── sample_0000_combined.png
├── sample_0000_cnn_features.png
├── sample_0000_rnn_features.png
├── sample_0000_sequence_attention.png
├── sample_0001_combined.png
├── sample_0001_cnn_features.png
├── sample_0001_rnn_features.png
├── sample_0001_sequence_attention.png
├── sample_0002_combined.png
├── sample_0002_cnn_features.png
├── sample_0002_rnn_features.png
└── sample_0002_sequence_attention.png

Total: 4 files × 3 samples = 12 PNG files
```

---

## 🎨 Output Content Details

### 1️⃣ Combined View (`sample_XXXX_combined.png`)
```
┌────────────────────────────────────────────┐
│  Original Image                            │
│  GT: "ground truth text"                   │
│  Pred: "model prediction"                  │
├────────────────────────────────────────────┤
│  CNN Attention Heatmap Overlay             │
├────────────────────────────────────────────┤
│  RNN Attention Heatmap Overlay             │
└────────────────────────────────────────────┘
```

### 2️⃣ CNN Features (`sample_XXXX_cnn_features.png`)
```
┌────────────────────────────────────────────┐
│  Input Image + Spatial Heatmap             │
│  (Red/Yellow = high attention)             │
│  (Blue = low attention)                    │
│  GT: "..." | Pred: "..."                   │
└────────────────────────────────────────────┘
```

### 3️⃣ RNN Features (`sample_XXXX_rnn_features.png`)
```
┌────────────────────────────────────────────┐
│  Input Image + Temporal Heatmap            │
│  (Shows sequence-level attention)          │
│  GT: "..." | Pred: "..."                   │
└────────────────────────────────────────────┘
```

### 4️⃣ Sequence Attention (`sample_XXXX_sequence_attention.png`)
```
┌────────────────────────────────────────────┐
│  Original Image                            │
│  GT: "..." | Pred: "..."                   │
├────────────────────────────────────────────┤
│  Bar Chart:                                │
│    ║█████   = Attention weight per timestep│
│    ║  ████  = Annotated with characters    │
│    ║    ██                                 │
│    └─────────────────────────────         │
│     0  5  10  15  20  (timesteps)          │
└────────────────────────────────────────────┘
```

---

## 💻 Usage Examples

### Example 1: Basic (minimum)
```bash
python visualize_attention.py \
    --model saved_models/htrnet.pt \
    --config config.yaml
```
**Result**: 40 PNG files (10 samples × 4 files) in `./attention_visualizations/`

### Example 2: More samples, custom output
```bash
python visualize_attention.py \
    --model saved_models/htrnet.pt \
    --config config.yaml \
    --num_samples 50 \
    --output_dir ./my_visualizations
```
**Result**: 200 PNG files (50 samples × 4 files) in `./my_visualizations/`

### Example 3: Single image
```bash
python visualize_attention.py \
    --model saved_models/htrnet.pt \
    --config config.yaml \
    --image ./my_handwriting.png
```
**Result**: 4 PNG files in `./attention_visualizations/`

### Example 4: CPU mode
```bash
python visualize_attention.py \
    --model saved_models/htrnet.pt \
    --config config.yaml \
    --device cpu \
    --num_samples 5
```
**Result**: 20 PNG files (5 samples × 4 files), runs on CPU

---

## 📊 Visual Input/Output Flow

```
COMMAND LINE
└─> python visualize_attention.py --model X.pt --config Y.yaml --num_samples 3

INPUT FILES LOADED
├── Model: X.pt (trained weights)
├── Config: Y.yaml (architecture + preprocessing settings)
└── Dataset: data/IAM/processed_lines/test/ (or single image)

PROCESSING
├── Load model architecture
├── Load trained weights into model
├── Set model to eval mode
├── For each sample:
│   ├── Load image from dataset
│   ├── Run forward pass with hooks
│   ├── Capture CNN activations
│   ├── Capture RNN activations
│   ├── Generate attention heatmaps
│   ├── Decode prediction
│   └── Create 4 visualizations

OUTPUT FILES CREATED (per sample)
├── sample_0000_combined.png          (all layers stacked)
├── sample_0000_cnn_features.png      (CNN spatial attention)
├── sample_0000_rnn_features.png      (RNN temporal attention)
└── sample_0000_sequence_attention.png (bar chart)

├── sample_0001_combined.png
├── sample_0001_cnn_features.png
├── sample_0001_rnn_features.png
└── sample_0001_sequence_attention.png

├── sample_0002_combined.png
├── sample_0002_cnn_features.png
├── sample_0002_rnn_features.png
└── sample_0002_sequence_attention.png

CONSOLE OUTPUT
└─> ✓ Attention visualizations saved to: ./attention_visualizations
    [1/3] GT: "text1" | Pred: "pred1"
    [2/3] GT: "text2" | Pred: "pred2"
    [3/3] GT: "text3" | Pred: "pred3"
```

---

## 📏 File Specifications

| File | Size | Format | Resolution | Contains |
|------|------|--------|------------|----------|
| `*_combined.png` | 300-500 KB | PNG | ~1500×900 px | All layers stacked |
| `*_cnn_features.png` | 150-250 KB | PNG | ~1500×300 px | CNN spatial heatmap |
| `*_rnn_features.png` | 150-250 KB | PNG | ~1500×300 px | RNN temporal heatmap |
| `*_sequence_attention.png` | 200-300 KB | PNG | ~1500×600 px | Bar chart with annotations |

**Total per sample**: ~1-1.5 MB  
**For 10 samples**: ~10-15 MB  
**For 100 samples**: ~100-150 MB

---

## 🎯 What Each Output Shows

| Output File | Visualization Type | Shows | Use Case |
|-------------|-------------------|-------|----------|
| Combined | All-in-one view | Original + all attention layers | Quick overview |
| CNN Features | Spatial heatmap | Which image regions CNN focuses on | Debug CNN backbone |
| RNN Features | Temporal heatmap | Which sequence features RNN uses | Debug RNN decoder |
| Sequence Attention | Bar chart | Attention per timestep + chars | Correlate attention to predictions |

---

## 🔍 Reading the Outputs

### Heatmap Colors:
- 🔴 **Red/Yellow**: HIGH attention (model focuses here)
- 🟢 **Green**: MEDIUM attention
- 🔵 **Blue/Purple**: LOW attention (model ignores this)

### What to Look For:
✅ **Good attention**: Focuses on actual handwritten characters  
❌ **Bad attention**: Focuses on blank spaces or noise  
⚠️ **Interesting patterns**: Attention correlates with errors

### Example Interpretation:
```
GT:   "think"
Pred: "thine"
      ^^^^^ 
CNN Attention: Strong on "k" region but RNN didn't decode it correctly
→ Problem: RNN decoder, not feature extraction
```

---

## ⚡ Quick Command Reference

| What You Want | Command |
|---------------|---------|
| Visualize 10 test samples | `python visualize_attention.py --model model.pt --config config.yaml` |
| Visualize 50 samples | Add `--num_samples 50` |
| Use validation set | Add `--dataset val` |
| Save elsewhere | Add `--output_dir ./my_folder` |
| Single image | Add `--image ./my_image.png` |
| Use CPU | Add `--device cpu` |

---

## 📋 Checklist Before Running

- [ ] Model file exists: `saved_models/htrnet.pt` ✓
- [ ] Config file exists: `config.yaml` ✓
- [ ] Dataset path in config is correct ✓
- [ ] Python environment activated ✓
- [ ] Dependencies installed (torch, numpy, matplotlib, cv2) ✓
- [ ] CUDA available (or use `--device cpu`) ✓
- [ ] Enough disk space for outputs (~1.5 MB per sample) ✓

---

## 🎓 Summary

**Input**: Model + Config + (Dataset or Image)  
**Process**: Run inference with attention hooks  
**Output**: 4 PNG files per sample showing attention heatmaps  
**Time**: ~1-2 seconds per sample (GPU) or ~5-10 seconds (CPU)  
**Disk**: ~1.5 MB per sample

**Simplest command**:
```bash
python visualize_attention.py --model model.pt --config config.yaml
```

**That's it!** 🎉
