# Modular Attention Visualization Architecture

## System Design Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HTR Model (models.py)                                │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐         │
│  │     CNN      │  →   │     RNN      │  →   │    Output    │         │
│  │   Features   │      │  (LSTM/GRU)  │      │   (Logits)   │         │
│  └──────────────┘      └──────────────┘      └──────────────┘         │
│         ↓                      ↓                                        │
│    [Hook Capture]         [Hook Capture]                               │
└─────────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────────┐
│              AttentionVisualizer (utils/visualizer.py)                  │
│                                                                         │
│  ┌──────────────────────┐                                              │
│  │  Activation Storage  │  {                                           │
│  │                      │    'cnn_features': Tensor[B,C,H,W],          │
│  │  self.activations    │    'rnn_features': Tensor[T,B,C]             │
│  │                      │  }                                            │
│  └──────────────────────┘                                              │
│            ↓                                                            │
│  ┌──────────────────────────────────────────────────────┐              │
│  │         Attention Map Generation                     │              │
│  │  - Average CNN channels → Spatial map [H, W]        │              │
│  │  - Aggregate RNN sequence → Temporal attention [T]   │              │
│  │  - Normalize to [0, 1]                               │              │
│  │  - Resize to match input image dimensions            │              │
│  └──────────────────────────────────────────────────────┘              │
│            ↓                                                            │
│  ┌──────────────────────────────────────────────────────┐              │
│  │         Heatmap Overlay & Visualization              │              │
│  │  - Apply colormap (jet) to attention map             │              │
│  │  - Overlay: 40% image + 60% heatmap                  │              │
│  │  - Save combined + individual layer maps             │              │
│  │  - Generate sequence attention bar charts            │              │
│  └──────────────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                     Output Files (PNG)                                  │
│                                                                         │
│  - sample_0001_combined.png           (all layers stacked)             │
│  - sample_0001_cnn_features.png       (spatial attention)              │
│  - sample_0001_rnn_features.png       (temporal attention)             │
│  - sample_0001_sequence_attention.png (bar chart)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

## Usage Modes

### Mode 1: Standalone Script (Recommended)
```
┌──────────────────┐      ┌──────────────────┐      ┌────────────────┐
│  Trained Model   │  →   │  visualize_      │  →   │  Attention     │
│  (.pt file)      │      │  attention.py    │      │  Maps (PNG)    │
└──────────────────┘      └──────────────────┘      └────────────────┘
        ↓
   ┌────────────┐
   │ config.yaml│
   └────────────┘

Command: python visualize_attention.py --model model.pt --config config.yaml
```

### Mode 2: Python API
```python
# In your Python code
from utils.visualizer import AttentionVisualizer, AttentionHook

visualizer = AttentionVisualizer(save_dir='./maps')

with AttentionHook(model, visualizer) as viz:
    output = model(image)
    viz.visualize_and_save(image, "GT", "Pred", "sample_1")
```

### Mode 3: Training Integration (Optional)
```
┌──────────────────┐      ┌──────────────────┐      ┌────────────────┐
│  trainer.py      │  →   │  AttentionHook   │  →   │  Attention     │
│                  │      │  (conditional)    │      │  Maps (epoch)  │
│  - if epoch % 50 │      │                  │      │                │
└──────────────────┘      └──────────────────┘      └────────────────┘
```

## Hook Mechanism

```
Model Forward Pass:
┌────────┐
│ Input  │
│ Image  │
└────┬───┘
     │
     ↓
┌────────────────┐     ┌───────────────────┐
│  CNN Layer     │ →   │ Forward Hook      │ → Store activation
│                │     │ (registered)      │   in dict
└────┬───────────┘     └───────────────────┘
     │
     ↓
┌────────────────┐     ┌───────────────────┐
│  RNN Layer     │ →   │ Forward Hook      │ → Store activation
│                │     │ (registered)      │   in dict
└────┬───────────┘     └───────────────────┘
     │
     ↓
┌────────────────┐
│   CTC Output   │
└────────────────┘

After forward pass: visualizer.activations contains all captured tensors
```

## Key Design Principles

### 1. Modularity
- **Separated concerns**: Visualization is isolated in `utils/visualizer.py`
- **No model changes**: Uses PyTorch hooks (non-invasive)
- **Optional**: Can be completely disabled (`enabled=False`)

### 2. Flexibility
- **Standalone or integrated**: Works independently or in training loop
- **Custom layers**: Hook any layer by name
- **Multiple visualizations**: CNN spatial, RNN temporal, sequence charts

### 3. Clean API
- **Context manager**: `AttentionHook` ensures automatic cleanup
- **Simple interface**: 3-line usage for basic visualization
- **Configuration**: Can be controlled via config file

### 4. Zero Overhead When Disabled
```python
if not self.enabled:
    return  # Early exit in all methods
```

## File Organization

```
HTR-best-practices/
├── utils/
│   └── visualizer.py              # Core module (342 lines)
│       ├── AttentionVisualizer    # Main class
│       │   ├── register_hooks()   # Hook registration
│       │   ├── generate_attention_maps()  # Map generation
│       │   ├── visualize_and_save()       # Save outputs
│       │   └── visualize_sequence_attention()
│       └── AttentionHook          # Context manager
│
├── visualize_attention.py         # CLI script (247 lines)
│   ├── visualize_samples()        # Batch visualization
│   ├── visualize_specific_sample() # Single image
│   └── main()                     # CLI entry point
│
├── trainer_visualization_integration.py  # Integration examples
│   ├── setup_visualizer()         # Optional trainer method
│   └── visualize_attention_samples()     # Call during training
│
├── config_with_visualization.yaml # Config with viz settings
│
└── Documentation
    ├── VISUALIZATION_README.md    # Complete docs
    └── QUICKSTART_VISUALIZATION.md # Quick start guide
```

## Data Flow

```
Input Image (1×H×W)
      ↓
Model Forward Pass
      ↓
┌─────────────────────────────────┐
│  CNN Features (B×C×H'×W')       │ ← Hook captures
│  - Average channels              │
│  - Spatial attention map (H'×W') │
└─────────────────────────────────┘
      ↓
┌─────────────────────────────────┐
│  RNN Features (T×B×C)            │ ← Hook captures
│  - Aggregate sequence            │
│  - Temporal attention (T)        │
└─────────────────────────────────┘
      ↓
Normalize & Colormap
      ↓
Overlay on Input Image
      ↓
Save PNG Files
```

## Attention Map Processing

### CNN Spatial Attention
```python
activation: [B, C, H, W]
    ↓
mean(dim=0)  # Average across channels
    ↓
attn_map: [H, W]
    ↓
normalize to [0, 1]
    ↓
resize to input dimensions
    ↓
apply colormap (jet)
    ↓
overlay: 0.4 * image + 0.6 * heatmap
```

### RNN Temporal Attention
```python
activation: [T, B, C]
    ↓
abs().mean(dim=0)  # Magnitude, average features
    ↓
attn_map: [T]
    ↓
normalize to [0, 1]
    ↓
visualize as:
  - Heatmap (expand to 2D)
  - Bar chart with character labels
```

## Integration Patterns

### Pattern 1: Post-Training Analysis (Recommended)
```bash
# After training completes
python visualize_attention.py --model run_1/model.pt --config config.yaml
```
**Pros**: No training code changes, no overhead during training  
**Cons**: Can only visualize after training completes

### Pattern 2: Periodic During Training
```python
# In training loop
if epoch % 50 == 0:
    with AttentionHook(model, visualizer):
        # Visualize validation samples
```
**Pros**: Track attention evolution during training  
**Cons**: Slight overhead every N epochs

### Pattern 3: On-Demand API
```python
# In Jupyter notebook or custom script
from utils.visualizer import AttentionVisualizer
viz = AttentionVisualizer('./maps')
# ... run inference and visualize
```
**Pros**: Maximum flexibility  
**Cons**: Requires custom code

## Performance Characteristics

- **Hook Registration**: O(1) per layer (one-time setup)
- **Activation Capture**: O(1) per forward pass (tensor reference)
- **Attention Generation**: O(H×W×C) for CNN, O(T×C) for RNN
- **Visualization**: O(H×W) image processing
- **File I/O**: ~100-500 KB per PNG (depends on resolution)

**Memory Impact**:
- Disabled: 0 bytes
- Enabled: ~few MB per sample (activation tensors)
- Activations are `.detach()`ed (no gradients stored)

## Extension Points

Want to customize? Edit these in `visualizer.py`:

1. **Colormap** (line 165): Change `plt.cm.jet` to `viridis`, `plasma`, etc.
2. **Overlay ratio** (line 169): Change `0.4 * img + 0.6 * heatmap`
3. **Resolution** (line 181): Change `dpi=150` to higher value
4. **Hook layers** (line 62-75): Add custom layer names
5. **Aggregation** (line 128-148): Change how attention is computed

## Testing Checklist

✅ Standalone script works  
✅ Hooks capture CNN features  
✅ Hooks capture RNN features  
✅ Attention maps generated correctly  
✅ Overlays render properly  
✅ Files saved to correct directory  
✅ Context manager cleans up hooks  
✅ Zero overhead when disabled  
✅ Works with 'both', 'cnn', 'rnn' head types  
✅ Handles batch and single image inputs  

---

This modular architecture ensures:
- **Clean separation**: Visualization is independent of core model/training
- **Easy maintenance**: Changes to one component don't affect others
- **Extensibility**: Easy to add new visualization types
- **User-friendly**: Multiple usage modes for different needs
