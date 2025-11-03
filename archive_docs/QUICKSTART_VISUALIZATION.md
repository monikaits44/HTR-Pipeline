# Attention Visualization - Quick Start Guide

## ✅ What Was Implemented

A **fully modular** attention visualization system that:
- Works independently of training code (zero coupling)
- Uses PyTorch hooks to capture intermediate activations
- Generates attention heatmaps overlaid on input images
- Supports CNN spatial attention and RNN temporal attention
- Can be used standalone or optionally integrated into training

## 📁 Files Created

```
HTR-best-practices/
├── utils/
│   └── visualizer.py                      # Core AttentionVisualizer class (342 lines)
├── visualize_attention.py                 # Standalone visualization script (247 lines)
├── config_with_visualization.yaml         # Config with visualization settings
├── trainer_visualization_integration.py   # Optional trainer integration examples
├── VISUALIZATION_README.md                # Complete documentation
└── QUICKSTART_VISUALIZATION.md            # This file
```

## 🚀 Quick Start (3 Steps)

### Step 1: Visualize Your Trained Model

```bash
# Activate your environment
.venv\Scripts\Activate.ps1

# Generate attention maps for 10 test samples
python visualize_attention.py \
    --model saved_models/htrnet.pt \
    --config config.yaml \
    --num_samples 10 \
    --output_dir ./attention_maps
```

### Step 2: Check the Output

Navigate to `./attention_maps/` and you'll find:

For each sample:
- `sample_0001_combined.png` - All layers stacked vertically
- `sample_0001_cnn_features.png` - CNN spatial attention
- `sample_0001_rnn_features.png` - RNN temporal attention  
- `sample_0001_sequence_attention.png` - Sequence attention bar chart

### Step 3: Visualize a Specific Image

```bash
python visualize_attention.py \
    --model saved_models/htrnet.pt \
    --config config.yaml \
    --image path/to/your/handwritten_image.png \
    --output_dir ./custom_attention
```

## 🎨 What You'll See

Each visualization shows:
1. **Original Image**: The input handwritten text line
2. **Ground Truth**: The actual transcription
3. **Prediction**: What the model predicted
4. **CNN Attention**: Spatial heatmap showing which image regions the CNN focuses on
5. **RNN Attention**: Temporal attention showing which sequence positions are important
6. **Sequence Chart**: Bar chart of attention weights across timesteps

## 🔧 Modular Design Benefits

### No Training Code Changes Required
- The visualizer uses **PyTorch hooks** (non-invasive)
- Training code remains unchanged
- Zero overhead when visualization is disabled

### Three Usage Modes

**Mode 1: Standalone Script** (Recommended)
```bash
# Works with any saved model
python visualize_attention.py --model <model.pt> --config <config.yaml>
```

**Mode 2: Python API**
```python
from utils.visualizer import AttentionVisualizer, AttentionHook

visualizer = AttentionVisualizer(save_dir='./maps', enabled=True)

with AttentionHook(model, visualizer) as viz:
    output = model(image)
    viz.visualize_and_save(image, "GT text", "Pred text", "sample_1")
```

**Mode 3: Training Integration** (Optional)
```python
# Add to trainer.py if you want visualization during training
# See trainer_visualization_integration.py for examples
```

## 📊 Architecture

### How It Works

```
Input Image → Model → [Hooks Capture Activations] → Generate Heatmaps → Overlay on Image
                ↓
            CNN Features (spatial attention)
            RNN Features (temporal attention)
```

### Hook Registration
```python
visualizer.register_hooks(model)  # Automatically hooks CNN and RNN layers
# or
visualizer.register_hooks(model, layer_names=['features.cnv3', 'top.rec'])  # Custom layers
```

### Activation Capture
During forward pass:
- **CNN**: Captures feature maps `[Batch, Channels, Height, Width]`
- **RNN**: Captures sequence outputs `[Time, Batch, Features]`

### Attention Map Generation
- **CNN**: Average across channels → 2D spatial heatmap
- **RNN**: Aggregate sequence → 1D temporal attention or 2D visualization
- Normalize to [0, 1] → Apply jet colormap → Overlay on image (40% img + 60% heatmap)

## 📝 Example Use Cases

### 1. Debug Model Performance
```bash
# Visualize failure cases
python visualize_attention.py --model model.pt --config config.yaml --dataset test --num_samples 50
# Then manually inspect where attention went wrong
```

### 2. Compare Model Versions
```bash
# Visualize run_1
python visualize_attention.py --model saved_models/experiments/run_1/model.pt \
    --config saved_models/experiments/run_1/config.json --output_dir ./run1_attention

# Visualize run_2
python visualize_attention.py --model saved_models/experiments/run_2/model.pt \
    --config saved_models/experiments/run_2/config.json --output_dir ./run2_attention
```

### 3. Create Paper Figures
```bash
# Generate high-quality visualizations
python visualize_attention.py --model model.pt --config config.yaml --num_samples 5 \
    --output_dir ./paper_figures
# Edit save_path in visualizer.py to increase DPI if needed (currently 150 DPI)
```

### 4. Visualize During Training (Optional)
```python
# In your trainer.py (see trainer_visualization_integration.py for full example):
if epoch % 50 == 0:
    visualizer = AttentionVisualizer(save_dir=f'./attention_epoch_{epoch}')
    with AttentionHook(self.net, visualizer):
        # Run inference and visualize
        pass
```

## 🎯 Key Features

✅ **Zero Training Overhead**: Completely disabled when `enabled=False`  
✅ **Non-Invasive**: Uses PyTorch hooks (no model code changes)  
✅ **Flexible**: Hook any layer by name  
✅ **Clean API**: Context manager for automatic cleanup  
✅ **Multiple Outputs**: Combined, individual layers, sequence charts  
✅ **Customizable**: Change colormaps, overlay ratios, DPI in `visualizer.py`

## 🛠️ Advanced Usage

### Custom Colormap
Edit `utils/visualizer.py` line 165:
```python
heatmap = plt.cm.jet(attn_resized)[:, :, :3]  # Change jet to viridis, plasma, etc.
```

### Adjust Overlay Ratio
Edit line 169:
```python
overlay = 0.4 * img_np + 0.6 * heatmap  # Change ratios (must sum to 1.0)
```

### Higher Resolution
Edit line 181:
```python
individual_fig.savefig(individual_path, bbox_inches='tight', dpi=300)  # Increase from 150
```

### Hook Specific Layers
```python
visualizer.register_hooks(model, layer_names=['features.cnv1', 'features.cnv2', 'top.rec.layer0'])
```

## 📦 Dependencies

Already installed if you have the HTR project running:
- `torch` - PyTorch for hooks
- `numpy` - Array operations
- `matplotlib` - Plotting
- `opencv-python` (cv2) - Image resizing

If missing:
```bash
pip install torch numpy matplotlib opencv-python
```

## 🐛 Troubleshooting

**Problem**: "No attention maps generated"  
**Solution**: Ensure `enabled=True` and model is in eval mode: `model.eval()`

**Problem**: "CUDA out of memory"  
**Solution**: Use CPU: `--device cpu` or reduce batch size

**Problem**: "Wrong image dimensions"  
**Solution**: Attention maps are auto-resized to match input. Check input preprocessing.

**Problem**: "Hooks not capturing"  
**Solution**: Print `visualizer.activations.keys()` to debug which layers were hooked

## 📚 Documentation

- **VISUALIZATION_README.md**: Complete documentation with examples
- **visualize_attention.py**: Standalone script with CLI
- **trainer_visualization_integration.py**: Optional training integration examples
- **utils/visualizer.py**: Core implementation with docstrings

## 🎓 Understanding the Output

### CNN Attention (Spatial)
- **Hot colors (red/yellow)**: Important spatial regions
- **Cool colors (blue/purple)**: Less important regions
- Shows which parts of the handwritten line the CNN focuses on

### RNN Attention (Temporal)
- Shows attention across the sequence
- Useful for understanding which timesteps are important for decoding

### Sequence Bar Chart
- X-axis: Sequence timestep
- Y-axis: Attention weight (normalized)
- Annotated with decoded characters
- Helps correlate attention to specific character predictions

## ✨ Example Results

From our test run (3 samples):

**Sample 0:**
- GT: `"Become a success with a disc and hey presto! You're a star.... Rolly sings with"`
- Pred: `"Become a success with a dise and hey presto! You're a starwo Rolly sings with"`
- Error: "disc" → "dise", missing period, "star...." → "starwo"

**Sample 1:**
- GT: `"assuredness "Bella Bella Marie" (Parlophone), a lively song that changes tempo mid-way."`
- Pred: `"assuredness "Bella Bella Marie" (Parlophone) a lively song that changes tempo mitway."`
- Error: missing comma, "mid-way" → "mitway"

**Sample 2:**
- GT: `"I don't think he will storm the charts with this one, but it's a good start."`
- Pred: `"I don't thine he will storm the charts with this one, but it's a good start."`
- Error: "think" → "thine"

The attention maps help visualize **where** the model focused when making these errors!

## 🚀 Next Steps

1. **Visualize more samples**: `--num_samples 50`
2. **Try different models**: `--model saved_models/experiments/run_2/model.pt`
3. **Customize visualizations**: Edit `utils/visualizer.py` for different styles
4. **Optional training integration**: See `trainer_visualization_integration.py`
5. **Use for debugging**: Focus on failure cases to improve model

## 💡 Tips

- Start with 5-10 samples to get a feel for the output
- Use validation set (`--dataset val`) for visualization during development
- Compare attention between correct and incorrect predictions
- Look for patterns: Does the model struggle with certain characters or positions?
- Use attention maps to justify architectural choices in papers

---

**Questions?** Check `VISUALIZATION_README.md` for detailed documentation.

**Enjoy visualizing your HTR model's attention! 🎨✨**
