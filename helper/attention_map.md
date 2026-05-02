
# HTR Register-Augmented Attention Visualisation

> **Making Handwritten Text Recognition explainable for downstream tasks such as writer identification**

---

## Overview

This toolset adds **register tokens** (Darcet et al., ICLR 2024) to a ViT-based CTC HTR model and exposes two complementary visualisation routines that map the model's internal attention to human-readable spatial heat-maps over the handwriting image.

| Figure | Paper reference | Purpose |
|--------|-----------------|---------|
| **Fig 1 – Register Impact Grid** | Darcet et al., Fig. 1 / Fig. 19 | Show how register tokens eliminate attention artifacts and produce cleaner CLS-to-patch maps |
| **Fig 2 – Character-Level Attention Grid** | "Beyond Memorization", Fig. 5 + VLAC paper Fig. 4 | Show which spatial region the model attends to for *each decoded character* — the core explainability signal for writer ID |

---

## The Problem Being Solved

```
Without registers                     With registers
─────────────────────────────────────────────────────
[CLS]→patch attention is polluted     Attention is clean and spatially
by high-norm "outlier" tokens          coherent → reliable for writer ID
(background artefacts)
```

In a standard ViT-HTR backbone large, well-trained models spontaneously *recycle* patch tokens from low-information regions (inter-character gaps, ascender/descender white-space) to store global information. This creates `high-norm outlier tokens` that corrupt attention maps and prevent fine-grained interpretability.

**Register tokens** give the model explicit scratch-pad tokens so the outlier behaviour is cleanly absorbed, leaving every patch token with purely local spatial information and producing artefact-free attention maps.

---

## Architecture

```
Input image (1×64×256)
       │
  PatchEmbed (4×4 patches → 16×64 = 1024 patch tokens, dim=128)
       │
  [CLS] │ patch_1 … patch_1024 │ [REG_1] … [REG_R]
       │           ─────────────────────────────────
       │           appended after pos-embedding; no position assigned
       │
  6× AttentionBlock  ←── last block captures weights for visualisation
       │
  LayerNorm
       │
  ┌────┴────────────────────────────────────┐
  │                                         │
 [CLS] (writer-level)              patch_1…1024 (character-level)
  for classification                   ↓
                               column-average → CTC head
                               logits (B, n_w, n_classes)
                               
  [REG_1]…[REG_R]  ← DISCARDED at output (Darcet et al. §2.2)
```

### Token sequence sizes

| Config        | Total tokens | [CLS] | Patches | Registers |
|---------------|-------------|-------|---------|-----------|
| 0-register    | 1025        | 1     | 1024    | 0         |
| 4-register    | 1029        | 1     | 1024    | 4         |
| 8-register    | 1033        | 1     | 1024    | 8         |

FLOP increase for 4 registers: **< 2%** (matches Darcet et al. Fig. 12).

---

## Key Components

### `PatchEmbed`
Convolutional patch splitter. Patch size `4×4` matches the HTR best-practices setup where each vertical slice corresponds to approximately one character stroke column.

### `AttentionBlock`
Standard pre-LN transformer block with a **weight capture hook** (`enable_capture()` / `disable_capture()`). The last active block stores `last_attn_weights` of shape `(B, T_seq, T_seq)` for downstream visualisation.

### `HTRViT`
Full ViT-CTC encoder. Key design choices:
- Register tokens have **no positional embedding** (intentional — they must be position-agnostic scratch pads)
- Only `[CLS]` + patch tokens are returned as output; registers are sliced off
- `enable_attention_capture(layer=-1)` arms the last block for one forward pass

### `extract_cls_to_patch_attn`
Extracts the `[CLS]`-row from the attention weight tensor and reshapes it to the `(n_h, n_w)` patch grid. Normalised to [0, 1] per sample.

```python
# Token layout:  [CLS | p1 … pN | REG_1 … REG_R]
#                  0     1…N      N+1…N+R
cls_attn = attn_weights[:, 0, 1 : 1 + N]   # (B, N)
```

### `extract_ctc_char_attn`
Character-level attention extractor:

1. **Greedy CTC decode** `logits → (chars, peak_columns)`
2. For each peak column `c` look up attention row `attn[1+c, 1:1+N]`
   (the patch token at column `c` attending to all other patches)
3. Reshape to `(n_h, n_w)` and normalise

This gives a one-to-one spatial map per decoded character.

---

## Visualisation Functions

### `plot_register_impact_grid`

```
┌─────────┬──────────┬─────────┬─────────┬─────────┬─────────┐
│  Input  │  No Reg  │  1 Reg  │  2 Reg  │  4 Reg  │  8 Reg  │
├─────────┼──────────┼─────────┼─────────┼─────────┼─────────┤
│         │ CLS attn │ CLS attn│ CLS attn│ CLS attn│ CLS attn│  row 0
├─────────┼──────────┼─────────┼─────────┼─────────┼─────────┤
│  image  │ PCA feat │ PCA feat│ PCA feat│ PCA feat│ PCA feat│  row 1
├─────────┼──────────┼─────────┼─────────┼─────────┼─────────┤
│         │ tok norm │ tok norm│ tok norm│ tok norm│ tok norm│  row 2
└─────────┴──────────┴─────────┴─────────┴─────────┴─────────┘
```

Artifact hot-spots in the 0-register norm map are highlighted with a **cyan bounding box**.

### `plot_char_attention_grid`

```
┌──────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ Original │ Char 1  │ Char 2  │ Char 3  │ Char 4  │ Char 5  │ Char 6  │ Char 7  │
├──────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│  word 1  │ attn+im │ attn+im │ attn+im │ attn+im │ attn+im │ attn+im │ attn+im │
├──────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│  word 2  │  ...                                                                  │
└──────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
```

Each cell overlays the greyscale handwriting with the character's spatial attention heat-map. The decoded character is annotated in white.

---

## Quick Start

### Demo mode (no checkpoints needed)

```bash
pip install torch torchvision matplotlib pillow numpy

# uses the provided sample image (or auto-generates a synthetic one)
python htr_register_attention_viz.py --image sample.png --out_dir outputs/
```

Output files:
```
outputs/
├── fig1_register_impact_grid.pdf
└── fig2_char_attention_grid.pdf
```

### With real HTR-best-practices checkpoints

```bash
# Train a model using https://github.com/georgeretsi/HTR-best-practices
# then export two checkpoints: baseline and +4reg

python htr_register_attention_viz.py \
    --image          path/to/line.png \
    --checkpoint_no_reg   models/htr_baseline.pt \
    --checkpoint_with_reg models/htr_4reg.pt \
    --out_dir         outputs/
```

### Programmatic use

```python
from htr_register_attention_viz import HTRViT, plot_char_attention_grid

model = HTRViT(n_registers=4, n_classes=80)
model.load_state_dict(torch.load("my_model.pt"))

charset = [" "] + list("abcdefghijklmnopqrstuvwxyz...")

plot_char_attention_grid(
    image_paths=["line1.png", "line2.png", "line3.png"],
    model=model,
    charset=charset,
    max_chars=8,
    out_path="char_maps.pdf",
)
```

---

## Integration with HTR-best-practices

The `HTRViT` class is designed to be a drop-in augmentation of the ViT backbone in [georgeretsi/HTR-best-practices](https://github.com/georgeretsi/HTR-best-practices).

**Step-by-step integration:**

```python
# 1. Locate the ViT backbone in the repo, typically in model.py or similar
# 2. Replace / wrap the patch-embedding layer with PatchEmbed from this module
# 3. After the patch embedding, inject register tokens:

class YourHTRModel(nn.Module):
    def __init__(self, n_registers=4, ...):
        super().__init__()
        # existing code ...
        if n_registers > 0:
            self.reg_tokens = nn.Parameter(
                torch.zeros(1, n_registers, embed_dim)
            )

    def forward(self, x):
        tokens = self.patch_embed(x)
        cls    = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed

        # inject registers (NO positional embedding)
        if hasattr(self, "reg_tokens"):
            reg = self.reg_tokens.expand(x.size(0), -1, -1)
            tokens = torch.cat([tokens, reg], dim=1)

        tokens = self.transformer(tokens)

        # discard registers before CTC head
        patch_out = tokens[:, 1 : 1 + N_patches, :]
        return self.ctc_head(patch_out)
```

**Training note:** The model must be *trained* with registers for the artefact-reduction effect to appear. Adding registers only at inference time will not reproduce the clean attention maps shown in Darcet et al.

---

## Writer Identification Connection

The character-level attention maps directly support the **VLAC (Vectors of Locally Aggregated Characters)** pipeline:

```
HTR attention map for char 'a'  →  spatial region mask
                                         │
                                  extract patch features
                                  within that mask
                                         │
                                  aggregate per character
                                  across all text lines
                                         │
                                  VLAC descriptor for 'a'
                                         │
                                  cosine distance between writers
```

Register tokens ensure the attention mask is clean (no background artefacts absorbed), so the extracted features are truly local allograph features rather than global document statistics.

---

## Extending the Script

### Adding GradCAM

```python
def gradcam_char(model, img_t, target_col):
    """GradCAM w.r.t. logit at CTC column target_col."""
    model.eval()
    img_t.requires_grad_(True)
    out   = model(img_t)
    score = out["logits"][0, target_col].max()
    score.backward()
    grads = img_t.grad[0, 0]           # (H, W)
    return grads.abs().numpy()
```

### Using multiple attention heads

Replace `average_attn_weights=True` with `average_attn_weights=False` in `AttentionBlock.forward` to get per-head maps of shape `(B, n_heads, T, T)`, then visualise head diversity as in Darcet et al. Fig. 18.

### Exporting maps for downstream models

```python
# dump all character maps as numpy arrays for VLAC aggregation
np.save("char_attn_maps.npy", np.stack(char_maps))  # (n_chars, n_h, n_w)
```

---

## References

1. **Darcet et al.** "Vision Transformers Need Registers." ICLR 2024. arXiv:2309.16588
2. **"Beyond Memorization: Training-Free Style Mixing for Variability in Handwritten Text Generation Using Writer Embedding Injection in Pretrained Diffusion Models."** ECCV workshop 2024.
3. **Retsinas et al.** "Best Practices for a Handwritten Text Recognition System." DAS 2022. [github.com/georgeretsi/HTR-best-practices](https://github.com/georgeretsi/HTR-best-practices)
4. **VLAC: "Interpretable Writer Recognition via Vectors of Locally Aggregated Characters."**

---

## File Structure

```
htr_attention_viz/
├── htr_register_attention_viz.py   # main script (this file)
├── README.md                       # this documentation
└── outputs/                        # generated figures (created at runtime)
    ├── fig1_register_impact_grid.pdf
    └── fig2_char_attention_grid.pdf
```

https://claude.ai/chat/9818c97d-ecf9-4401-8a04-92845597fb17