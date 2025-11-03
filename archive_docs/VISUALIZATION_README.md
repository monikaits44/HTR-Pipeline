# Attention Visualization Module

Modular attention map visualization for Handwritten Text Recognition (HTR) models.

## Overview

This module provides a **clean, modular approach** to visualizing attention maps without modifying core training code. It uses PyTorch hooks to capture intermediate activations and generates heatmap overlays on input images.

## Architecture

```
├── utils/visualizer.py              # Core AttentionVisualizer class
├── visualize_attention.py           # Standalone visualization script
├── trainer_visualization_integration.py  # Optional trainer integration examples
└── config_with_visualization.yaml   # Config with visualization settings
```

## Key Features

✅ **Zero Training Overhead**: Visualization can be completely disabled with `enabled=False`  
✅ **Standalone Script**: Visualize any saved model without touching training code  
✅ **PyTorch Hooks**: Captures intermediate activations without modifying forward pass  
✅ **Multiple Layers**: Visualize CNN features, RNN attention, or custom layers  
✅ **Sequence Visualization**: Special handling for RNN/LSTM sequential attention  
✅ **Context Manager**: Clean hook registration/cleanup with `AttentionHook`

## Usage

### Option 1: Standalone Visualization (Recommended)

Use the standalone script to visualize attention for already-trained models:

```bash
# Visualize 10 samples from test set
python visualize_attention.py \
    --model saved_models/experiments/run_1/model.pt \
    --config config.yaml \
    --num_samples 10 \
    --output_dir ./attention_visualizations

# Visualize specific image file
python visualize_attention.py \
    --model saved_models/experiments/run_1/model.pt \
    --config config.yaml \
    --image path/to/handwritten_image.png \
    --output_dir ./attention_visualizations
```

**Arguments:**
- `--model`: Path to trained model checkpoint (.pt file)
- `--config`: Path to configuration file (.yaml)
- `--num_samples`: Number of samples to visualize from dataset (default: 10)
- `--dataset`: Which split to use: 'train', 'val', or 'test' (default: 'test')
- `--image`: Path to a specific image to visualize (overrides --num_samples)
- `--output_dir`: Directory to save visualizations (default: './attention_visualizations')
- `--device`: Device to run on (default: 'cuda:0')

### Option 2: Programmatic Usage

Use the `AttentionVisualizer` class directly in your Python code:

```python
from utils.visualizer import AttentionVisualizer, AttentionHook
from models import HTRNet
import torch

# Initialize model and visualizer
model = HTRNet(config.arch, num_classes)
model.load_state_dict(torch.load('model.pt'))
model.eval()

visualizer = AttentionVisualizer(save_dir='./attention_maps', enabled=True)

# Use context manager for clean hook management
with AttentionHook(model, visualizer) as viz:
    # Run inference
    output = model(input_image)
    
    # Generate and save attention maps
    viz.visualize_and_save(
        input_image=input_image,
        transcription="ground truth text",
        prediction="predicted text",
        sample_id="sample_001",
        save_individual=True
    )
```

### Option 3: Training Integration (Optional)

If you want to visualize attention *during* training, you can integrate the visualizer into your trainer. See `trainer_visualization_integration.py` for examples.

**Minimal integration steps:**

1. Add to config (`config_with_visualization.yaml`):
```yaml
visualization:
  enabled: True
  save_dir: './attention_maps'
  save_every_k_epochs: 50
  num_samples_per_save: 5
```

2. Import and initialize in trainer:
```python
from utils.visualizer import AttentionVisualizer

# In __init__:
if hasattr(config, 'visualization') and config.visualization.enabled:
    self.visualizer = AttentionVisualizer(
        save_dir=config.visualization.save_dir, 
        enabled=True
    )
```

3. Call visualization method periodically:
```python
# In training loop:
if epoch % config.visualization.save_every_k_epochs == 0:
    self.visualize_attention_samples(epoch, num_samples=5)
```

See `trainer_visualization_integration.py` for complete integration examples.

## How It Works

### 1. Hook Registration
The visualizer uses PyTorch's `register_forward_hook()` to capture intermediate activations:

```python
visualizer.register_hooks(model)  # Hooks CNN and RNN layers
```

### 2. Activation Capture
During forward pass, hooks save activations to a dictionary:

```python
self.activations = {
    'cnn_features': tensor,  # Shape: [B, C, H, W]
    'rnn_features': tensor   # Shape: [T, B, C]
}
```

### 3. Attention Map Generation
Activations are processed into spatial attention maps:
- **CNN**: Average across channels → 2D spatial map
- **RNN**: Aggregate across sequence → 1D temporal attention or 2D heatmap

### 4. Visualization
Attention maps are:
- Normalized to [0, 1]
- Resized to match input image dimensions
- Converted to heatmaps using colormap (jet)
- Overlaid on original images (40% image + 60% heatmap)

## Output Examples

The visualizer generates several types of outputs:

### 1. Combined Attention Maps
`sample_0001_combined.png` - Shows original image + all layer attention maps stacked vertically

### 2. Individual Layer Maps
- `sample_0001_cnn_features.png` - CNN spatial attention
- `sample_0001_rnn_features.png` - RNN temporal attention

### 3. Sequence Attention
`sample_0001_sequence_attention.png` - Bar chart showing attention weights across sequence timesteps

Each visualization includes:
- Original input image
- Ground truth transcription (if available)
- Model prediction
- Attention heatmap overlay

## Advanced Usage

### Custom Layer Hooks

Hook specific layers by name:

```python
visualizer = AttentionVisualizer(save_dir='./maps')
visualizer.register_hooks(model, layer_names=['features.cnv3', 'top.rec'])
```

### Manual Attention Map Generation

Generate maps without automatic hooks:

```python
# Capture activations manually
with torch.no_grad():
    features = model.features(input_image)

# Generate attention maps
attention_maps = visualizer.generate_attention_maps(
    input_image=input_image,
    activations={'custom_layer': features}
)
```

### Sequence-Specific Visualization

Visualize attention at specific timesteps:

```python
visualizer.visualize_sequence_attention(
    input_image=image,
    rnn_activations=rnn_output,
    decoded_sequence="predicted text",
    ground_truth="actual text",
    sample_id="sample_001"
)
```

## Performance Considerations

- **Disabled Mode**: Zero overhead when `enabled=False` (hooks not registered)
- **Memory**: Activations are detached and stored separately (no gradient computation)
- **Inference Only**: Designed for evaluation/inference, not training (use sparingly during training)

## Requirements

```python
torch
numpy
matplotlib
opencv-python (cv2)
```

Install with:
```bash
pip install torch numpy matplotlib opencv-python
```

## Architecture Notes

### Why Hooks?
- **Non-invasive**: No need to modify model forward pass
- **Flexible**: Can hook any layer dynamically
- **Clean**: Easy to enable/disable without code changes

### Why Context Manager?
The `AttentionHook` context manager ensures:
- Hooks are registered only when needed
- Automatic cleanup after use
- No memory leaks from lingering hooks

### Why Separate Module?
- **Modularity**: Visualization is optional, not core functionality
- **Testing**: Can test visualization independently
- **Reusability**: Can use same visualizer for different models/tasks

## Troubleshooting

**Q: No attention maps generated?**  
A: Check that `enabled=True` and hooks are registered. Print `visualizer.activations.keys()` to see captured layers.

**Q: CUDA out of memory?**  
A: Reduce batch size or disable visualization during training. Activations are stored in GPU memory.

**Q: Wrong image size?**  
A: Attention maps are resized to match input image dimensions using `cv2.resize()`.

**Q: Hooks not capturing?**  
A: Ensure model is in eval mode: `model.eval()`. Hooks work for both train/eval but eval is recommended.

## Examples

See `visualize_attention.py` for complete working examples of:
- Batch visualization from dataset
- Single image visualization
- Custom output directories
- Different dataset splits

## Citation

If you use this visualization module in your research, please cite your HTR work and mention the visualization approach.
