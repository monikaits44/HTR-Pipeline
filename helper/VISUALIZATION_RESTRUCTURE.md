# HTR Visualization Restructure Summary

## Overview
The HTR register attention visualization script has been completely restructured to match the reference style with organized output folders and improved visual aesthetics.

## Key Changes

### 1. **Organized Output Structure**
Visualizations are now organized into 4 separate folders:

```
visualizations/register_analysis/
├── character_attention/      # Detailed character-level attention maps
├── word_attention/           # Simplified word-level attention (3-panel view)
├── register_comparison/      # Side-by-side comparison across register configurations
└── attention_heads/          # Per-head attention visualizations
```

### 2. **New Visualization Functions**

#### `visualize_character_attention()`
- Detailed character-level attention visualization
- Renamed from `visualize_single_model_attention()`
- Shows comprehensive attention analysis for each register configuration

#### `visualize_word_attention()`
- **NEW**: Simplified 3-panel word-level view
- Panels:
  1. Original input image with text
  2. Attention heatmap
  3. Attention overlay
- Cleaner, more focused presentation

#### `visualize_attention_heads()`
- **NEW**: Per-head attention grid visualization
- Shows individual attention patterns for each attention head
- Uses 4-column grid layout
- Uses `viridis` colormap for purple/blue style

### 3. **Register Comparison Updates**
- **Layout**: Changed from 3-row to 2-row grid matching reference
- **Colormap**: Changed from `hot` (red/yellow) to `viridis` (purple/blue)
- **Labels**: Clean "Input" and "N [reg]" labels for each column
- **Style**: Minimalist design matching reference image
- Row 1: Labels ("Input", "0 [reg]", "2 [reg]", etc.)
- Row 2: Images (original + attention heatmaps)

### 4. **Visual Improvements**
- **Colormap**: Changed from `hot` to `viridis` for purple/blue aesthetic
- **Layout**: Cleaner, more publication-ready presentations
- **Labels**: Simplified and standardized text labels
- **Consistency**: All visualizations follow same style guide

## Generated Files

For each test image (e.g., `c04-116-02.png`), the script generates:

### Register Comparison (1 file)
```
register_comparison/c04-116-02_register_comparison.png
```
Side-by-side comparison showing attention across all register configurations (0, 2, 4, 8, 16)

### Character-Level Attention (5 files)
```
character_attention/c04-116-02_0regs_character.png
character_attention/c04-116-02_2regs_character.png
character_attention/c04-116-02_4regs_character.png
character_attention/c04-116-02_8regs_character.png
character_attention/c04-116-02_16regs_character.png
```
Detailed character-level attention for each configuration

### Word-Level Attention (5 files)
```
word_attention/c04-116-02_0regs_word.png
word_attention/c04-116-02_2regs_word.png
word_attention/c04-116-02_4regs_word.png
word_attention/c04-116-02_8regs_word.png
word_attention/c04-116-02_16regs_word.png
```
Simplified word-level attention for each configuration

### Attention Heads (5 files)
```
attention_heads/c04-116-02_0regs_heads.png
attention_heads/c04-116-02_2regs_heads.png
attention_heads/c04-116-02_4regs_heads.png
attention_heads/c04-116-02_8regs_heads.png
attention_heads/c04-116-02_16regs_heads.png
```
Per-head attention grid for each configuration

**Total**: 16 visualizations per test image

## Usage

```bash
python scripts/postprocessing/visualize_htr_register_attention.py \
  --image-path data/IAM/processed_lines/test/<image>.png \
  --runs run_21 run_22 run_23 run_24 run_25 \
  --register-counts 0 2 4 8 16
```

## Technical Details

### Attention Extraction Mechanism
- Uses `MultiheadAttentionWrapper` that inherits from `nn.MultiheadAttention`
- Forces `need_weights=True` in forward pass
- **Critical**: Requires no-op forward hook registration for PyTorch to properly invoke wrapper's forward method
- This workaround is necessary because `TransformerEncoderLayer._sa_block()` explicitly passes `need_weights=False`

### Color Schemes
- **Register Comparison**: `viridis` (purple → blue gradient)
- **Character Attention**: `hot` (black → red → yellow)
- **Word Attention**: `hot` (black → red → yellow)
- **Attention Heads**: `viridis` (purple → blue gradient)

### Layout Specifications
- **Register Comparison**: 2 rows × (N+1) columns where N = number of configurations
- **Character Attention**: 3 rows × 3 columns (detailed multi-panel view)
- **Word Attention**: 1 row × 3 columns (simplified view)
- **Attention Heads**: Dynamic grid based on number of heads (typically 2 rows × 4 columns for 8 heads)

## Validation
✅ All 5 models (run_21-25) load successfully  
✅ Attention extraction works reliably with no-op hook trick  
✅ All 4 visualization types generate correctly  
✅ Outputs organized in separate folders  
✅ Visual style matches reference image  

## Notes
- The attention extraction mechanism preserves the critical no-op hook trick that makes attention capture work
- All colormap choices are designed to match publication-quality standards
- File naming follows consistent pattern: `<image>_<N>regs_<type>.png`
- Layout dimensions are optimized for readability and comparison
