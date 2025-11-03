# visualize_attention.py - Input/Output Reference

## Overview
`visualize_attention.py` is a standalone script that generates attention map visualizations for trained HTR models. It takes a trained model and generates heatmap overlays showing where the model focuses during inference.

---

## Required Inputs

### 1. **Model Checkpoint** (Required)
- **Parameter**: `--model` or `-m`
- **Type**: File path to `.pt` file
- **Example**: `saved_models/htrnet.pt` or `saved_models/experiments/run_1/model.pt`
- **Description**: PyTorch model state_dict containing trained weights
- **What it needs**: Compatible with HTRNet architecture in `models.py`

### 2. **Configuration File** (Required)
- **Parameter**: `--config` or `-c`
- **Type**: File path to `.yaml` file
- **Example**: `config.yaml` or `saved_models/experiments/run_1/config.json`
- **Description**: Model configuration with architecture settings, preprocessing params
- **Must contain**:
  - `arch`: Architecture configuration (CNN, RNN settings)
  - `preproc`: Image preprocessing (height, width)
  - `data.path`: Path to dataset directory

---

## Optional Inputs

### 3. **Number of Samples** (Optional)
- **Parameter**: `--num_samples`
- **Type**: Integer
- **Default**: `10`
- **Example**: `--num_samples 50`
- **Description**: How many samples from the dataset to visualize
- **Range**: 1 to dataset size

### 4. **Output Directory** (Optional)
- **Parameter**: `--output_dir`
- **Type**: Directory path (string)
- **Default**: `./attention_visualizations`
- **Example**: `--output_dir ./my_visualizations`
- **Description**: Where to save generated PNG files
- **Note**: Directory will be created if it doesn't exist

### 5. **Dataset Split** (Optional)
- **Parameter**: `--dataset`
- **Type**: String
- **Choices**: `train`, `val`, `test`
- **Default**: `test`
- **Example**: `--dataset val`
- **Description**: Which dataset split to sample from

### 6. **Device** (Optional)
- **Parameter**: `--device`
- **Type**: String
- **Default**: `cuda:0`
- **Example**: `--device cpu` or `--device cuda:1`
- **Description**: PyTorch device for inference (GPU or CPU)

### 7. **Specific Image** (Optional)
- **Parameter**: `--image`
- **Type**: File path to image
- **Default**: `None`
- **Example**: `--image ./my_handwriting.png`
- **Description**: Visualize a single specific image file instead of dataset samples
- **Note**: When provided, overrides `--num_samples` and `--dataset`
- **Supported formats**: PNG, JPG, any format readable by OpenCV

---

## Usage Examples

### Example 1: Basic Usage (Dataset Samples)
```bash
python visualize_attention.py \
    --model saved_models/htrnet.pt \
    --config config.yaml
```
**Inputs**:
- Model: `saved_models/htrnet.pt`
- Config: `config.yaml`
- Uses defaults: 10 samples from test set, output to `./attention_visualizations`, runs on CUDA

### Example 2: Custom Number of Samples
```bash
python visualize_attention.py \
    --model saved_models/experiments/run_1/model.pt \
    --config config.yaml \
    --num_samples 50 \
    --output_dir ./run1_attention
```
**Inputs**:
- Model: `saved_models/experiments/run_1/model.pt`
- Config: `config.yaml`
- Visualize 50 samples
- Save to `./run1_attention`

### Example 3: Validation Set
```bash
python visualize_attention.py \
    --model saved_models/htrnet.pt \
    --config config.yaml \
    --dataset val \
    --num_samples 20
```
**Inputs**:
- Model: `saved_models/htrnet.pt`
- Config: `config.yaml`
- Use validation set (not test)
- Visualize 20 samples

### Example 4: CPU Inference
```bash
python visualize_attention.py \
    --model saved_models/htrnet.pt \
    --config config.yaml \
    --device cpu \
    --num_samples 5
```
**Inputs**:
- Model: `saved_models/htrnet.pt`
- Config: `config.yaml`
- Run on CPU (no CUDA)
- Visualize 5 samples

### Example 5: Single Specific Image
```bash
python visualize_attention.py \
    --model saved_models/htrnet.pt \
    --config config.yaml \
    --image ./my_handwritten_text.png \
    --output_dir ./single_image_attention
```
**Inputs**:
- Model: `saved_models/htrnet.pt`
- Config: `config.yaml`
- Image file: `./my_handwritten_text.png`
- Save to `./single_image_attention`

---

## Outputs Generated

### For Each Sample, 4 PNG Files Are Created:

#### 1. **Combined View** - `sample_XXXX_combined.png`
**Contains**:
- Original input image (top)
- Ground truth transcription
- Model prediction
- CNN attention heatmap overlay (middle)
- RNN attention heatmap overlay (bottom)

**Layout**: Vertical stack of all visualizations

**Example**: `sample_0001_combined.png`

**Size**: ~300-500 KB, dimensions ~1500x900 pixels (depends on image size)

---

#### 2. **CNN Spatial Attention** - `sample_XXXX_cnn_features.png`
**Contains**:
- Input image with CNN attention heatmap overlay
- Shows spatial regions the CNN backbone focuses on
- Heatmap: Red/Yellow = high attention, Blue = low attention

**What it shows**: Which parts of the handwritten line image are most important for the CNN feature extraction

**Example**: `sample_0001_cnn_features.png`

**Size**: ~150-250 KB

---

#### 3. **RNN Temporal Attention** - `sample_XXXX_rnn_features.png`
**Contains**:
- Input image with RNN attention heatmap overlay
- Shows temporal/sequential attention from LSTM/GRU
- Heatmap indicates which sequence positions are important

**What it shows**: Which temporal features the RNN focuses on during sequence processing

**Example**: `sample_0001_rnn_features.png`

**Size**: ~150-250 KB

---

#### 4. **Sequence Attention Bar Chart** - `sample_XXXX_sequence_attention.png`
**Contains**:
- Original input image (top)
- Bar chart showing attention weights across timesteps (bottom)
- X-axis: Sequence timesteps
- Y-axis: Attention weight (normalized 0-1)
- Bars annotated with decoded characters

**What it shows**: How attention varies across the sequence, correlated with character predictions

**Example**: `sample_0001_sequence_attention.png`

**Size**: ~200-300 KB

---

## Complete Output Example

For `--num_samples 3`, you get:

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
```

**Total files**: 4 files × N samples = 12 files for 3 samples

---

## Output Specifications

### Image Format
- **Format**: PNG (lossless)
- **Color**: RGB
- **DPI**: 150 (can be changed in `visualizer.py`)

### Heatmap Colors
- **Colormap**: Jet (red-yellow-blue)
- **Red/Yellow**: High attention (important regions)
- **Green**: Medium attention
- **Blue/Purple**: Low attention (less important)

### Overlay Ratio
- **Image**: 40% opacity
- **Heatmap**: 60% opacity
- Total: Blended overlay showing both image content and attention

### Text Annotations
Each visualization includes:
- **GT**: Ground truth transcription
- **Pred**: Model prediction
- **Title**: Layer name or visualization type

---

## Console Output

During execution, the script prints:

```
Loading model from saved_models/htrnet.pt...
Loading test dataset...
Character classes: [' ', '!', '"', '#', ...] (75 different characters)
Generating attention maps for 3 samples...
Attention maps saved to attention_visualizations_test
  [1/3] GT: "Original text here" | Pred: "Predicted text here"
  [2/3] GT: "..." | Pred: "..."
  [3/3] GT: "..." | Pred: "..."

✓ Attention visualizations saved to: ./attention_visualizations_test
```

---

## Data Flow Diagram

```
INPUTS                                    OUTPUTS
┌──────────────────┐                     
│ Model (.pt)      │ ──┐                 ┌──────────────────────────┐
└──────────────────┘   │                 │ sample_0001_combined.png │
                       │                 │   - Original image       │
┌──────────────────┐   │                 │   - GT & Prediction      │
│ Config (.yaml)   │ ──┤                 │   - CNN attention        │
└──────────────────┘   │                 │   - RNN attention        │
                       ├─> PROCESSING ──>└──────────────────────────┘
┌──────────────────┐   │                 
│ Dataset or       │ ──┤                 ┌──────────────────────────┐
│ Single Image     │   │                 │ sample_0001_cnn_...png   │
└──────────────────┘   │                 │   - Spatial attention    │
                       │                 └──────────────────────────┘
┌──────────────────┐   │                 
│ Parameters:      │ ──┘                 ┌──────────────────────────┐
│ - num_samples    │                     │ sample_0001_rnn_...png   │
│ - device         │                     │   - Temporal attention   │
│ - output_dir     │                     └──────────────────────────┘
└──────────────────┘                     
                                         ┌──────────────────────────┐
                                         │ sample_0001_sequence.png │
                                         │   - Bar chart with chars │
                                         └──────────────────────────┘
```

---

## Dependencies

The script requires these to be installed:
```python
torch              # Model loading and inference
numpy              # Array operations
matplotlib         # Plotting and visualization
cv2 (opencv-python) # Image loading and resizing
omegaconf          # Config file parsing
```

---

## Error Handling

### Common Input Errors:

**Missing model file**:
```
FileNotFoundError: Model not found at <path>
```

**Missing config file**:
```
FileNotFoundError: Config not found at <path>
```

**Invalid image file**:
```
ValueError: Could not load image from <path>
```

**CUDA not available**:
```
RuntimeError: CUDA error... (automatically falls back to CPU or specify --device cpu)
```

---

## Quick Reference Table

| Parameter | Required | Default | Example Value | Purpose |
|-----------|----------|---------|---------------|---------|
| `--model` | ✅ Yes | - | `saved_models/htrnet.pt` | Model checkpoint path |
| `--config` | ✅ Yes | - | `config.yaml` | Configuration file |
| `--num_samples` | ❌ No | `10` | `50` | Number of samples to visualize |
| `--output_dir` | ❌ No | `./attention_visualizations` | `./my_viz` | Output directory |
| `--dataset` | ❌ No | `test` | `val` | Dataset split (train/val/test) |
| `--device` | ❌ No | `cuda:0` | `cpu` | Inference device |
| `--image` | ❌ No | `None` | `./image.png` | Single image to visualize |

---

## File Size Estimates

**Per sample** (approximate):
- Combined: 300-500 KB
- CNN attention: 150-250 KB
- RNN attention: 150-250 KB
- Sequence chart: 200-300 KB
- **Total per sample**: ~1-1.5 MB

**For 10 samples**: ~10-15 MB
**For 100 samples**: ~100-150 MB

---

## Use Cases

1. **Debug model failures**: See where attention went wrong on incorrect predictions
2. **Compare models**: Visualize attention from different training runs
3. **Create paper figures**: Generate high-quality attention visualizations
4. **Understand model**: See what features the model focuses on
5. **Validate architecture**: Check if attention patterns make sense

---

## Summary

**Minimum command**:
```bash
python visualize_attention.py --model <model.pt> --config <config.yaml>
```

**Outputs**: 4 PNG files per sample showing attention maps with ground truth and predictions

**Total control**: Customize samples, device, output location, and data source via command-line arguments
