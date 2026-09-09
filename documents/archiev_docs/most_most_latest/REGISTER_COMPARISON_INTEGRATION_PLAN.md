# Integration Plan: Character Attention Maps + Register Comparison Framework

## Executive Summary

**Core Research Question:** *Do attention maps become cleaner with registers?*

This document provides a concrete, step-by-step plan to:
1. Generate character-specific attention maps from your ViT-RGTS models
2. Systematically compare attention quality across models with different register counts (0, 4, 7, 16)
3. Quantify "cleanness" of attention maps with rigorous metrics
4. Produce publication-ready figures for your thesis

---

## 1. How Character-Specific Attention Maps Work in Your Pipeline

### 1.1 The Key Difference from Beyond-Memorization

| Aspect | Beyond-Memorization | Your Pipeline (ViT-RGTS + CTC) |
|--------|--------------------|---------------------------------|
| Attention type | Cross-attention (image→text) | Self-attention (token↔token) |
| Character mapping | Direct — decoder queries one char at a time | Indirect — CTC aligns chars to time positions |
| Token semantics | Each decoder query = one character | Each column token = ~8px horizontal strip |
| Register tokens | None | 0–16, prepended to sequence |
| Spatial layout | 2D patch grid (14×14 or similar) | 1D column sequence (128 tokens) |

### 1.2 Your Architecture's Attention Flow

```
Input Image [1, 1, 128, 1024]
        │
        ▼
   CNN Stem → Feature Map [B, 256, 1, 128]
        │
        ▼
   Squeeze → Column Tokens [B, 128, 256]
        │
        ▼
   Prepend Registers → [B, R+128, 256]
        │
        ▼
   6× Transformer Layers (8 heads each)
   Each layer produces attention: [B, 8, R+128, R+128]
        │
        ▼
   Split: reg_tokens [B, R, 256] + patch_tokens [B, 128, 256]
        │
        ▼
   CTC Head → logits [128, B, 54]
        │
        ▼
   CTC Decode → text + character-to-position mapping
```

### 1.3 Generating Per-Character Attention Maps

**Step 1:** Run `forward_explain()` → get `attn_maps` (6 layers × [B, 8, S, S]) and `logits` [T, B, C]

**Step 2:** CTC greedy decode with position tracking:
- For each time step `t`, decode the most probable character
- Collapse consecutive same predictions (CTC rule)
- Record which time positions `t` produced each character

**Step 3:** For each character, extract its attention pattern:
- Select rows from the attention matrix at that character's time positions
- Average across those rows → get a 128-dim attention vector
- This tells you: "When producing character X, the model attended to these other columns"

**Step 4:** Overlay as heatmap on input image (each column = 8 pixels wide)

**Already implemented in:** `scripts/postprocessing/character_attention_viz.py`

---

## 2. Integration into Your Repository

### 2.1 What Already Exists

| Component | Location | Status |
|-----------|----------|--------|
| `forward_explain()` | `models.py:385` | ✅ Working |
| `extract_attention_weights()` | `scripts/trainer.py:748` | ✅ Saves .npy every 5 epochs |
| `AttentionExtractor` | `utils/attention_extractor.py` | ✅ Has entropy metrics |
| `character_attention_viz.py` | `scripts/postprocessing/` | ✅ All 3 approaches |
| Register comparison script | — | ❌ **Needs creation** |
| Quantitative metrics module | — | ❌ **Needs creation** |
| Batch evaluation script | — | ❌ **Needs creation** |

### 2.2 What Needs to Be Created

#### File 1: `scripts/postprocessing/register_comparison.py`
**Purpose:** Load multiple models (different register counts), run same images, compare attention maps side-by-side with quantitative metrics.

#### File 2: `utils/attention_metrics.py`
**Purpose:** Quantitative metrics for attention map "cleanness" — entropy, sparsity, character localization accuracy, peak sharpness.

#### File 3: `scripts/postprocessing/batch_attention_eval.py`
**Purpose:** Run metrics over entire test set (not just 5 samples) to get statistically significant results.

---

## 3. Quantitative Metrics: Defining "Cleaner" Attention

### 3.1 Metric Suite

| Metric | What it measures | "Cleaner" = | Formula |
|--------|-----------------|-------------|---------|
| **Attention Entropy** | How spread out the attention is | Lower | $H = -\sum_j a_j \log(a_j)$ |
| **Sparsity (Gini)** | How concentrated on few positions | Higher | Gini coefficient of attention vector |
| **Peak Sharpness** | How dominant the peak is | Higher | $\max(a) / \text{mean}(a)$ |
| **Character Localization** | Does attention focus on correct spatial position | Higher | Attention mass within ±2 positions of character |
| **Cross-char Overlap** | Do different characters have distinct attention | Lower | IoU between adjacent characters' attention |
| **Register Utilization** | How much patch→register attention flows | Varies | Mean attention from patches to registers |
| **Register Specialization** | Do registers develop distinct roles | Higher | Variance across registers' attention patterns |

### 3.2 What "Cleaner" Means Operationally

A "clean" attention map for HTR should:
1. **Focus on the character's spatial location** (not spread uniformly)
2. **Include relevant context** (neighboring characters for disambiguation)
3. **Distinguish different characters** (char A's attention ≠ char B's attention)
4. **Be consistent** (same character in different contexts → similar attention pattern)

With registers, we hypothesize:
- Registers absorb "global information gathering" that would otherwise pollute character-specific attention
- Patch-to-patch attention becomes more locally focused (cleaner)
- More registers → more capacity for global storage → cleaner character attention

---

## 4. Concrete Integration Steps

### Step 1: Create the Metrics Module

**File:** `utils/attention_metrics.py`

```python
"""
Quantitative metrics for attention map quality analysis.
Used to answer: "Do attention maps become cleaner with registers?"
"""

import numpy as np
from typing import Dict, List, Tuple


def attention_entropy(attn_vector: np.ndarray) -> float:
    """
    Shannon entropy of attention distribution.
    Lower = more focused = "cleaner".
    
    Args:
        attn_vector: [P] attention weights (already softmaxed, sums to 1)
    Returns:
        Entropy in nats
    """
    a = attn_vector + 1e-10  # avoid log(0)
    a = a / a.sum()  # ensure normalization
    return -np.sum(a * np.log(a))


def attention_sparsity_gini(attn_vector: np.ndarray) -> float:
    """
    Gini coefficient of attention distribution.
    Higher = more concentrated = "cleaner".
    Range: [0, 1] where 1 = all attention on single position.
    """
    a = np.sort(attn_vector)
    n = len(a)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * a) / (n * np.sum(a))) - (n + 1) / n


def peak_sharpness(attn_vector: np.ndarray) -> float:
    """
    Ratio of maximum attention to mean attention.
    Higher = sharper peak = "cleaner".
    """
    return attn_vector.max() / (attn_vector.mean() + 1e-10)


def character_localization_accuracy(
    attn_vector: np.ndarray, 
    char_positions: List[int],
    tolerance: int = 2
) -> float:
    """
    Fraction of attention mass within ±tolerance positions of the character.
    Higher = better localization = "cleaner".
    
    Args:
        attn_vector: [P] attention weights
        char_positions: Time positions assigned to this character
        tolerance: Number of positions around the character to consider "correct"
    """
    P = len(attn_vector)
    mask = np.zeros(P, dtype=bool)
    for pos in char_positions:
        start = max(0, pos - tolerance)
        end = min(P, pos + tolerance + 1)
        mask[start:end] = True
    
    return attn_vector[mask].sum() / (attn_vector.sum() + 1e-10)


def cross_character_overlap(
    attn_vectors: List[np.ndarray],
    threshold: float = 0.1
) -> float:
    """
    Average IoU between adjacent characters' attention maps.
    Lower = more distinct character attention = "cleaner".
    
    Args:
        attn_vectors: List of [P] attention vectors, one per character
        threshold: Binarization threshold for IoU computation
    """
    if len(attn_vectors) < 2:
        return 0.0
    
    overlaps = []
    for i in range(len(attn_vectors) - 1):
        a = (attn_vectors[i] > threshold).astype(float)
        b = (attn_vectors[i + 1] > threshold).astype(float)
        intersection = (a * b).sum()
        union = np.maximum(a, b).sum()
        if union > 0:
            overlaps.append(intersection / union)
    
    return np.mean(overlaps) if overlaps else 0.0


def register_specialization(
    attn_maps: np.ndarray,
    num_registers: int
) -> float:
    """
    Measure how specialized registers are (high variance across their patterns).
    Higher = more specialized = registers serving distinct roles.
    
    Args:
        attn_maps: [H, S, S] attention map (one layer, one sample, all heads averaged)
        num_registers: R
    """
    if num_registers == 0:
        return 0.0
    
    # Register-to-patch attention: [R, P]
    reg_to_patch = attn_maps[:num_registers, num_registers:]
    
    # Compute pairwise cosine distances between registers
    norms = np.linalg.norm(reg_to_patch, axis=1, keepdims=True) + 1e-10
    normalized = reg_to_patch / norms
    similarity_matrix = normalized @ normalized.T  # [R, R]
    
    # Average off-diagonal similarity (lower = more specialized)
    R = num_registers
    off_diag = similarity_matrix[~np.eye(R, dtype=bool)]
    return 1.0 - np.mean(off_diag)  # Convert to specialization score


def compute_all_metrics(
    patch_attn: np.ndarray,
    char_positions: Dict[int, List[int]],
    full_attn: np.ndarray = None,
    num_registers: int = 0
) -> Dict[str, float]:
    """
    Compute all attention quality metrics for a single sample.
    
    Args:
        patch_attn: [P, P] patch-to-patch attention (registers already stripped)
        char_positions: Dict mapping char_idx → list of time positions
        full_attn: [S, S] full attention including registers (for register metrics)
        num_registers: Number of register tokens
    
    Returns:
        Dictionary of metric_name → value
    """
    metrics = {}
    
    # Per-character metrics
    entropies = []
    sparsities = []
    sharpnesses = []
    localizations = []
    char_attn_vectors = []
    
    for char_idx, positions in char_positions.items():
        # Average attention FROM this character's positions TO all patches
        char_attn = patch_attn[positions, :].mean(axis=0)  # [P]
        char_attn_vectors.append(char_attn)
        
        entropies.append(attention_entropy(char_attn))
        sparsities.append(attention_sparsity_gini(char_attn))
        sharpnesses.append(peak_sharpness(char_attn))
        localizations.append(
            character_localization_accuracy(char_attn, positions, tolerance=3)
        )
    
    metrics['entropy_mean'] = np.mean(entropies)
    metrics['entropy_std'] = np.std(entropies)
    metrics['sparsity_gini_mean'] = np.mean(sparsities)
    metrics['peak_sharpness_mean'] = np.mean(sharpnesses)
    metrics['localization_accuracy_mean'] = np.mean(localizations)
    metrics['cross_char_overlap'] = cross_character_overlap(char_attn_vectors)
    
    # Register-specific metrics
    if full_attn is not None and num_registers > 0:
        metrics['register_specialization'] = register_specialization(
            full_attn, num_registers
        )
        # Patch-to-register attention (how much patches "use" registers)
        patch_to_reg = full_attn[num_registers:, :num_registers]  # [P, R]
        metrics['register_utilization'] = patch_to_reg.mean()
    
    return metrics
```

### Step 2: Create the Register Comparison Script

**File:** `scripts/postprocessing/register_comparison.py`

```python
#!/usr/bin/env python3
"""
Register Comparison: Systematic attention map quality comparison.
Answers: "Do attention maps become cleaner with registers?"

Loads models with different register counts, evaluates on same images,
computes quantitative metrics, and produces comparison visualizations.

Usage:
    python scripts/postprocessing/register_comparison.py \
        --models run_76:0 run_84:4 run_68:7 run_71:16 \
        --num-samples 50 \
        --save-dir visualizations/register_comparison
"""
```

**Key functionality:**
- Load 4+ models with register counts: 0, 4, 7, 16
- Run each model on the same N test images
- Compute all metrics for each model
- Statistical tests (paired t-test / Wilcoxon) between register counts
- Generate comparison plots (box plots, line plots, example heatmaps)

### Step 3: Integrate into Training Pipeline

Add to `scripts/trainer.py` evaluation loop (at epoch milestones):

```python
# In the evaluation method, after computing CER:
if epoch % 10 == 0 or epoch == max_epochs:
    self.run_attention_quality_evaluation(epoch)
```

This allows tracking attention quality *during training*, not just post-hoc.

---

## 5. Available Models for Comparison

Based on your `saved_models/experiments/`:

| Run | Registers | Architecture | Status |
|-----|-----------|--------------|--------|
| run_76 | 0 | vit_rgts | ✅ model.pt exists |
| run_80 | 1 | vit_rgts | Check |
| run_82 | 2 | vit_rgts | Check |
| run_66/78 | 3 | vit_rgts | Check |
| run_84 | 4 | vit_rgts | ✅ model.pt exists |
| run_65/85 | 5 | vit_rgts | Check |
| run_67/83 | 6 | vit_rgts | Check |
| run_68 | 7 | vit_rgts | ✅ model.pt exists (best CER: 5.99%) |
| run_63/87 | 8 | vit_rgts | Check |
| run_69 | 9 | vit_rgts | Check |
| run_61 | 10 | vit_rgts | Check |
| run_86 | 11 | vit_rgts | Check |
| run_60 | 13 | vit_rgts | Check |
| run_62/79 | 14 | vit_rgts | Check |
| run_70/81 | 15 | vit_rgts | Check |
| run_71/77 | 16 | vit_rgts | ✅ model.pt exists |

**Recommended comparison set:** run_76 (0), run_84 (4), run_68 (7), run_71 (16)

---

## 6. Expected Results and Hypothesis

### Hypothesis (from Registers paper)

1. **With 0 registers:** Patch attention will be noisy — some patches develop artificially high norms ("artifacts") because they're used as global information sinks
2. **With 4+ registers:** Registers absorb global information → patch attention becomes cleaner and more locally focused
3. **Diminishing returns:** Beyond ~4-8 registers, additional registers don't significantly improve attention quality
4. **Register specialization:** Each register develops a unique attention pattern (some attend to strokes, others to spacing)

### Expected Metrics Trends (registers: 0 → 4 → 7 → 16)

| Metric | Expected trend | Why |
|--------|---------------|-----|
| Entropy | ↓ decrease | Attention becomes more focused |
| Sparsity (Gini) | ↑ increase | Fewer positions receive high attention |
| Peak sharpness | ↑ increase | Dominant peaks become more prominent |
| Character localization | ↑ increase | Attention focuses on correct spatial region |
| Cross-char overlap | ↓ decrease | Characters develop distinct patterns |
| Register specialization | ↑ increase | Registers differentiate their roles |

---

## 7. Visualization Plan for Thesis

### Figure Set 1: Per-Character Attention Maps (qualitative)

For 2-3 sample words, show:
- Row 1: Input image with text annotation
- Row 2: Character attention heatmap (0 registers)
- Row 3: Character attention heatmap (4 registers)
- Row 4: Character attention heatmap (7 registers)
- Row 5: Character attention heatmap (16 registers)

**Purpose:** Visual evidence that attention becomes more focused with registers.

### Figure Set 2: Register Token Specialization

For models with 4 and 7 registers:
- One row per register, showing its attention pattern over the image
- Annotate what each register "specializes" in

**Purpose:** Show that registers develop meaningful, distinct roles.

### Figure Set 3: Quantitative Comparison (publication plots)

- Box plot: Entropy vs. Register Count (over N test samples)
- Line plot: All metrics vs. Register Count (mean ± std)
- Scatter plot: CER vs. Average Attention Entropy (correlation)
- Bar chart: Register specialization score vs. Register Count

**Purpose:** Statistical evidence for the hypothesis.

### Figure Set 4: Attention Flow Evolution

- Self-attention matrix [128×128] for models with 0, 4, 7, 16 registers
- Show how the attention pattern changes (expect more diagonal/local structure with registers)

---

## 8. Implementation Priority Order

### Phase 1: Core Infrastructure (Do First)

1. **Create `utils/attention_metrics.py`** — the metrics module (code above)
2. **Create `scripts/postprocessing/register_comparison.py`** — the comparison script
3. **Test on 3-5 samples** — verify everything works end-to-end

### Phase 2: Full Evaluation

4. **Run comparison over full test set** (~2,915 samples)
5. **Compute statistics** — mean, std, confidence intervals
6. **Statistical significance tests** — paired t-test between register counts

### Phase 3: Visualization & Thesis Figures

7. **Generate Figure Set 1-4** — publication-ready
8. **Create summary table** — CER + attention metrics for each register count
9. **Write analysis section** — interpret results in context of thesis

### Phase 4: Training Integration (Optional)

10. **Add attention quality tracking during training** — monitor how attention evolves
11. **Early stopping based on attention quality** — experimental

---

## 9. How to Run (End-to-End)

### Quick Test (5 minutes)
```bash
# Single model, single image — verify pipeline works
python scripts/postprocessing/character_attention_viz.py \
    --config configs/baseline_vit_rgts_v2.yaml \
    --model saved_models/experiments/run_68/model.pt \
    --image data/IAM/test/a01-000u-00.png \
    --save-dir visualizations/quick_test
```

### Register Comparison (after creating register_comparison.py)
```bash
# Compare 4 register counts on 50 test samples
python scripts/postprocessing/register_comparison.py \
    --models \
        saved_models/experiments/run_76/model.pt:0 \
        saved_models/experiments/run_84/model.pt:4 \
        saved_models/experiments/run_68/model.pt:7 \
        saved_models/experiments/run_71/model.pt:16 \
    --config configs/baseline_vit_rgts_v2.yaml \
    --num-samples 50 \
    --save-dir visualizations/register_comparison
```

### Full Evaluation (submit as SLURM job)
```bash
# Full test set — needs GPU, ~30 min
sbatch scripts/postprocessing/register_comparison.slurm
```

---

## 10. Connection to Thesis

This work directly addresses your thesis requirement:

> *"Make HTR explainable... Should have attention maps that are fine grained and meaningful... Integrate tokens for better attention maps... use idea from paper VISION TRANSFORMERS NEED REGISTERS"*

The deliverables are:
1. **Character-specific attention maps** → "fine grained and meaningful"
2. **Register comparison** → "Integrate tokens for better attention maps"
3. **Quantitative proof** → "Do attention maps become cleaner with registers?"
4. **Publication-ready figures** → thesis chapters on explainability

---

## 11. Summary of Files to Create/Modify

| Action | File | Purpose |
|--------|------|---------|
| **CREATE** | `utils/attention_metrics.py` | Quantitative metrics for attention quality |
| **CREATE** | `scripts/postprocessing/register_comparison.py` | Multi-model comparison script |
| **CREATE** | `scripts/postprocessing/register_comparison.slurm` | SLURM job for full evaluation |
| **MODIFY** | `scripts/trainer.py` | Add attention quality tracking during training |
| **EXISTING** | `scripts/postprocessing/character_attention_viz.py` | Already implements per-char visualization |
| **EXISTING** | `utils/attention_extractor.py` | Already has extraction + entropy |
| **EXISTING** | `models.py:forward_explain()` | Already returns everything needed |
