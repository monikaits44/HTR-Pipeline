"""
Visualize Register Token Impact on HTR Attention Patterns
==========================================================

Compares how different numbers of register tokens (0, 2, 4, 8, 16)
affect attention patterns in ViT-RGTS models for Handwritten Text Recognition.
Loads models from multiple run directories, hooks into transformer attention,
and produces side-by-side comparison visualizations.

Based on "Vision Transformers Need Registers" paper:
- Register tokens absorb high-norm artifacts in attention maps
- They provide "scratch space" for intermediate computations
- They improve feature quality and prevent outlier representations

Supported Architectures: vit_rgts only (requires register token configuration)
    Not applicable to: cnn_rnn, torchvision_vit, trocr
Input: Image file + multiple run directories with config.json & model.pt
Output: Multi-panel attention comparison figure

Outputs:
1. Predicted character sequence
2. Character-level attention visualization
3. Word-level attention visualization
4. Register comparison across all configurations
5. Per-head attention analysis

Usage:
    # Compare ViT-RGTS v2 runs (50-54) on a sample image
    python scripts/postprocessing/visualize_htr_register_attention.py \
        --image-path notebook/sample_images/a01-000u-00.png \
        --runs run_50 run_51 run_52 run_53 run_54 \
        --register-counts 0 2 4 8 16 \
        --output-dir visualizations/register_analysis

    # Compare just two configurations
    python scripts/postprocessing/visualize_htr_register_attention.py \
        --image-path notebook/sample_images/a01-000u-00.png \
        --runs run_50 run_54 \
        --register-counts 0 16 \
        --layer-type last

    # Use middle layer attention
    python scripts/postprocessing/visualize_htr_register_attention.py \
        --image-path notebook/sample_images/a01-000u-00.png \
        --runs run_53 --register-counts 8 \
        --layer-type middle
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from PIL import Image
import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models import HTRNet


class HTRViTWithAttention:
    """
    Wrapper to extract attention weights from HTR ViT models
    """
    def __init__(self, model, layer_type='last'):
        """
        Args:
            model: HTRNet model
            layer_type: 'last', 'middle', or specific layer index
        """
        self.model = model
        self.model.eval()
        self.layer_type = layer_type
        self.attention_weights = []
        self.register_attention = []
        self.num_registers = getattr(model.backbone, 'num_registers', 0)
        self.original_modules = []  # Store original modules for restoration
        
        # Register hooks to capture attention
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture attention weights"""
        backbone = self.model.backbone
        
        # Find transformer layers
        layers = None
        
        # Check for ViT-RGTS architecture (backbone.encoder.layers)
        if hasattr(backbone, 'encoder') and hasattr(backbone.encoder, 'layers'):
            layers = backbone.encoder.layers
        # Check for TorchVision ViT structure (backbone.vit.encoder or backbone.vit.transformer)
        elif hasattr(backbone, 'vit'):
            if hasattr(backbone.vit, 'encoder') and hasattr(backbone.vit.encoder, 'layers'):
                layers = backbone.vit.encoder.layers
            elif hasattr(backbone.vit, 'encoder') and hasattr(backbone.vit.encoder, 'blocks'):
                layers = backbone.vit.encoder.blocks
            elif hasattr(backbone.vit, 'transformer') and hasattr(backbone.vit.transformer, 'layers'):
                layers = backbone.vit.transformer.layers
        
        if layers is None:
            print(f"Warning: Could not find transformer layers in backbone")
            return
        
        num_layers = len(layers)
        
        # Determine which layer(s) to hook
        if self.layer_type == 'last':
            target_layers = [num_layers - 1]
        elif self.layer_type == 'middle':
            target_layers = [num_layers // 2]
        elif isinstance(self.layer_type, int):
            target_layers = [self.layer_type]
        else:
            raise ValueError(f"Invalid layer_type: {self.layer_type}")
        
        # Register hooks for target layers
        for layer_idx in target_layers:
            layer = layers[layer_idx]
            
            # For TransformerEncoderLayer, we need to wrap the self_attn module
            if hasattr(layer, 'self_attn'):
                # Create a wrapper class for MultiheadAttention that captures attention
                # Create wrapper - use explicit function to avoid closure issues
                def create_wrapper_class():
                    class MultiheadAttentionWrapper(torch.nn.MultiheadAttention):
                        def __init__(wrapper_self, original_attn, store_fn):
                            # Initialize with the same configuration as original
                            torch.nn.MultiheadAttention.__init__(
                                wrapper_self,
                                embed_dim=original_attn.embed_dim,
                                num_heads=original_attn.num_heads,
                                dropout=original_attn.dropout,
                                bias=original_attn.in_proj_bias is not None,
                                add_bias_kv=original_attn.bias_k is not None,
                                add_zero_attn=original_attn.add_zero_attn,
                                kdim=original_attn.kdim,
                                vdim=original_attn.vdim,
                                batch_first=original_attn.batch_first,
                                device=next(original_attn.parameters()).device,
                                dtype=next(original_attn.parameters()).dtype
                            )
                            # Copy the trained weights
                            wrapper_self.load_state_dict(original_attn.state_dict())
                            wrapper_self.store_fn = store_fn
                            
                        def forward(wrapper_self, query, key, value, key_padding_mask=None, 
                                   need_weights=True, attn_mask=None, average_attn_weights=True, **kwargs):
                            # FORCE need_weights=True and average_attn_weights=False regardless of what's passed
                            # This is because TransformerEncoderLayer._sa_block explicitly passes need_weights=False
                            output, attn_weights = torch.nn.MultiheadAttention.forward(
                                wrapper_self,
                                query, key, value,
                                key_padding_mask=key_padding_mask,
                                need_weights=True,  # Force True
                                attn_mask=attn_mask,
                                average_attn_weights=False,  # Force False to get per-head attention
                                **kwargs
                            )
                            # Store attention
                            if attn_weights is not None:
                                wrapper_self.store_fn(attn_weights)
                            # Return format expected by TransformerEncoderLayer (output, None)
                            return output, None
                    return MultiheadAttentionWrapper
                
                MultiheadAttentionWrapper = create_wrapper_class()
                
                # Store original and replace
                original_attn = layer.self_attn
                self.original_modules.append((layer, 'self_attn', original_attn))
                wrapper_instance = MultiheadAttentionWrapper(original_attn, self._store_attention)
                
                # Register a no-op hook to ensure forward hooks are properly set up
                # This appears to be necessary for PyTorch to correctly route through the wrapper's forward
                def noop_hook(module, input, output):
                    pass
                wrapper_instance.register_forward_hook(noop_hook)
                
                layer.self_attn = wrapper_instance
            
            # Fallback for other architectures
            elif hasattr(layer, 'attention') or hasattr(layer, 'attn'):
                attn_module = layer.attention if hasattr(layer, 'attention') else layer.attn
                attn_module.register_forward_hook(self._attention_hook)
    
    def _store_attention(self, attn_weights):
        """Store attention weights captured from forward wrapper"""
        # attn_weights shape: [batch, num_heads, seq_len, seq_len]
        self.attention_weights.append(attn_weights.detach().cpu())
    
    def _attention_hook(self, module, input, output):
        """Hook to capture attention weights"""
        # Attention weights are typically returned or stored in the module
        if hasattr(module, 'attention_weights'):
            attn = module.attention_weights
        elif isinstance(output, tuple) and len(output) > 1:
            # Some implementations return (output, attention_weights)
            attn = output[1]
        else:
            # Try to compute attention from query, key
            return
        
        # Store attention weights
        # Expected shape: [batch, heads, seq_len, seq_len]
        self.attention_weights.append(attn.detach().cpu())
    
    def forward(self, image):
        """Forward pass with attention capture"""
        self.attention_weights = []
        self.register_attention = []
        
        with torch.no_grad():
            output = self.model(image)
        
        return output
    
    def get_patch_attention(self):
        """Get attention to image patches (excluding CLS and register tokens)"""
        if not self.attention_weights:
            return None
        
        # Use last captured attention
        attn = self.attention_weights[-1]  # [batch, heads, seq, seq]
        
        # Extract CLS token attention to patches
        # Token order: [CLS, reg1, ..., regN, patch1, patch2, ...]
        num_regs = self.num_registers
        patch_attn = attn[:, :, 0, 1+num_regs:]  # [batch, heads, num_patches]
        
        return patch_attn
    
    def get_register_attention(self):
        """Get attention to register tokens"""
        if not self.attention_weights or self.num_registers == 0:
            return None
        
        attn = self.attention_weights[-1]
        # CLS token attention to register tokens
        reg_attn = attn[:, :, 0, 1:1+self.num_registers]  # [batch, heads, num_regs]
        
        return reg_attn
    
    def get_full_attention(self):
        """Get full attention matrix"""
        if not self.attention_weights:
            return None
        
        return self.attention_weights[-1]


def load_htr_model(run_path: str, device='cpu') -> Tuple[HTRNet, dict, int]:
    """
    Load HTR model from run directory
    
    Returns:
        model: Loaded HTRNet model
        config: Configuration dictionary
        num_registers: Number of register tokens
    """
    run_path = Path(run_path)
    
    # Load config
    config_path = run_path / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get number of registers
    num_registers = config['arch'].get('num_registers', 0)
    
    # Load character classes
    data_path = Path(config['data']['path'])
    classes_file = data_path / 'classes.npy'
    classes_array = np.load(classes_file, allow_pickle=True)
    
    # Create i2c and c2i mappings if classes is a simple array
    if isinstance(classes_array, np.ndarray) and classes_array.dtype.kind in ['U', 'S', 'O']:
        # It's an array of characters
        classes_list = classes_array.tolist()
        # Add 1 for CTC blank token (index 0)
        num_classes = len(classes_list) + 1
    else:
        # Assume it's a dict with i2c/c2i
        classes = classes_array.item()
        num_classes = len(classes['i2c'])
    
    # Create model
    from types import SimpleNamespace
    arch_cfg = SimpleNamespace(**config['arch'])
    
    model = HTRNet(arch_cfg, num_classes)
    
    # Load weights
    model_path = run_path / 'model.pt'
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Checkpoint is the state_dict directly
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, config, num_registers


def decode_prediction(output, classes, blank_idx=0):
    """Decode CTC output to text"""
    # output shape: [seq_len, batch, num_classes] or [batch, seq_len, num_classes]
    if output.dim() == 3:
        if output.shape[0] < output.shape[1]:
            # [seq_len, batch, classes]
            output = output.permute(1, 0, 2)
        # Now [batch, seq_len, classes]
    
    # Get predictions
    pred = output.argmax(dim=2)[0]  # [seq_len]
    
    # Remove duplicates and blanks (CTC decoding)
    decoded = []
    prev = None
    for idx in pred:
        idx = idx.item()
        if idx != blank_idx and idx != prev:
            decoded.append(classes['i2c'][idx])
        prev = idx
    
    return ''.join(decoded)


def visualize_attention_heads(
    model: HTRNet,
    image: torch.Tensor,
    text: str,
    num_registers: int,
    layer_type: str = 'last',
    save_path: Optional[str] = None
) -> plt.Figure:
    """Visualize per-head attention patterns with turbo colormap"""
    model_wrapper = HTRViTWithAttention(model, layer_type=layer_type)
    output = model_wrapper.forward(image)
    patch_attn = model_wrapper.get_patch_attention()
    
    if patch_attn is None:
        return None
    
    # Get per-head attention
    attention = patch_attn[0]  # [heads, num_patches]
    num_heads = attention.shape[0]
    
    # Reshape each head to 2D
    H, W = image.shape[2:]
    patch_h, patch_w = 16, 16
    grid_h, grid_w = H // patch_h, W // patch_w
    
    # Create figure with subplots for each head
    cols = 4
    rows = (num_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten() if num_heads > 1 else [axes]
    
    for head_idx in range(num_heads):
        head_attn = attention[head_idx]
        actual_seq_len = head_attn.shape[0]
        
        if actual_seq_len >= grid_h * grid_w:
            attn_map = head_attn[:grid_h * grid_w].reshape(grid_h, grid_w)
        else:
            padded = torch.nn.functional.pad(head_attn, (0, grid_h * grid_w - actual_seq_len))
            attn_map = padded.reshape(grid_h, grid_w)
        
        im = axes[head_idx].imshow(attn_map.cpu().numpy(), cmap='viridis', 
                                   aspect='auto', interpolation='bilinear')
        axes[head_idx].set_title(f'Head {head_idx}', fontsize=11, fontweight='bold')
        axes[head_idx].axis('off')
        plt.colorbar(im, ax=axes[head_idx], fraction=0.046)
    
    # Hide unused subplots
    for idx in range(num_heads, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'Attention Heads: {num_registers} Registers\n"{text[:40]}..."',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig





def visualize_register_comparison(
    models_dict: Dict[int, HTRNet],
    configs_dict: Dict[int, dict],
    image: torch.Tensor,
    classes: dict,
    layer_type: str = 'last',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare attention patterns across different register configurations
    
    Args:
        models_dict: {num_registers: model}
        configs_dict: {num_registers: config}
        image: Input image
        classes: Character classes
        layer_type: Layer to extract attention from
        save_path: Save path
    """
    register_counts = sorted(models_dict.keys())
    num_configs = len(register_counts)
    
    # Get predictions and attention for all models
    predictions = {}
    attention_maps = {}
    register_attentions = {}
    
    for num_regs in register_counts:
        model = models_dict[num_regs]
        wrapper = HTRViTWithAttention(model, layer_type=layer_type)
        
        # Forward pass
        output = wrapper.forward(image)
        
        # Decode prediction
        text = decode_prediction(output, classes)
        predictions[num_regs] = text
        
        # Get attention
        patch_attn = wrapper.get_patch_attention()
        register_attn = wrapper.get_register_attention()
        
        if patch_attn is not None:
            attention_maps[num_regs] = patch_attn[0].mean(dim=0)  # Average across heads
        
        if register_attn is not None:
            register_attentions[num_regs] = register_attn[0].mean(dim=0)
    
    # Create comparison figure matching "Vision Transformers Need Registers" paper style
    fig = plt.figure(figsize=(3.5 * (num_configs + 1), 8))
    gs = GridSpec(2, num_configs + 1, figure=fig, hspace=0.15, wspace=0.1)
    
    # Prepare image
    img_np = image[0].cpu().permute(1, 2, 0).numpy()
    if img_np.shape[2] == 1:
        img_np = img_np.squeeze(2)
    
    H, W = image.shape[2:]
    patch_h, patch_w = 16, 16
    grid_h, grid_w = H // patch_h, W // patch_w
    
    # Column 0: Input label and image
    ax_label = fig.add_subplot(gs[0, 0])
    ax_label.text(0.5, 0.5, 'Input', ha='center', va='center', 
                  fontsize=18, fontweight='bold')
    ax_label.axis('off')
    
    ax_orig = fig.add_subplot(gs[1, 0])
    if img_np.ndim == 2:
        ax_orig.imshow(img_np, cmap='gray', aspect='auto')
    else:
        ax_orig.imshow(img_np, aspect='auto')
    ax_orig.axis('off')
    
    # Process each configuration with turbo colormap (like paper)
    for col_idx, num_regs in enumerate(register_counts, start=1):
        # Row 0: Register count label
        ax_label = fig.add_subplot(gs[0, col_idx])
        ax_label.text(0.5, 0.5, f'{num_regs} [reg]', ha='center', va='center',
                     fontsize=16, fontweight='bold')
        ax_label.axis('off')
        
        # Row 1: Attention heatmap with turbo colormap (matches paper style)
        ax1 = fig.add_subplot(gs[1, col_idx])
        if num_regs in attention_maps:
            attn = attention_maps[num_regs]
            actual_seq_len = attn.shape[0]
            if actual_seq_len >= grid_h * grid_w:
                attn_map = attn[:grid_h * grid_w].reshape(grid_h, grid_w)
            else:
                padded = torch.nn.functional.pad(attn, (0, grid_h * grid_w - actual_seq_len))
                attn_map = padded.reshape(grid_h, grid_w)
            
            # Use viridis colormap: purple->cyan->green->yellow (matches reference)
            im1 = ax1.imshow(attn_map.cpu().numpy(), cmap='viridis', aspect='auto', 
                           interpolation='bilinear')
        else:
            blank = np.zeros((grid_h, grid_w))
            im1 = ax1.imshow(blank, cmap='viridis', aspect='auto')
        ax1.axis('off')
    
    # No title for clean paper-style visualization
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize register token impact on HTR attention')
    parser.add_argument('--image-path', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--runs', nargs='+', required=True,
                       help='Run directories (e.g., run_21 run_22 run_23 run_24 run_25)')
    parser.add_argument('--register-counts', nargs='+', type=int,
                       default=[0, 2, 4, 8, 16],
                       help='Register counts corresponding to runs')
    parser.add_argument('--output-dir', type=str, default='visualizations/register_analysis',
                       help='Output directory for visualizations')
    parser.add_argument('--layer-type', type=str, default='last',
                       choices=['last', 'middle'],
                       help='Which layer to extract attention from')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    register_dir = output_dir / 'register_comparison'
    heads_dir = output_dir / 'attention_heads'
    
    for d in [register_dir, heads_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("HTR Register Token Attention Visualization")
    print("="*70)
    print(f"Image: {args.image_path}")
    print(f"Runs: {args.runs}")
    print(f"Register counts: {args.register_counts}")
    print(f"Layer: {args.layer_type}")
    print(f"Output: {output_dir}")
    print("="*70)
    
    # Load image
    print("\nLoading image...")
    image_path = Path(args.image_path)
    image_pil = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Preprocess image
    image_np = np.array(image_pil).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # Resize to model input size (typically 128x1024)
    image_tensor = F.interpolate(image_tensor, size=(128, 1024), mode='bilinear', align_corners=False)
    image_tensor = image_tensor.to(args.device)
    
    # Load models
    print("\nLoading models...")
    models_dict = {}
    configs_dict = {}
    
    base_path = Path('saved_models/experiments')
    
    for run_name, num_regs in zip(args.runs, args.register_counts):
        run_path = base_path / run_name
        print(f"  Loading {run_name} ({num_regs} registers)...")
        
        try:
            model, config, actual_num_regs = load_htr_model(run_path, device=args.device)
            
            if actual_num_regs != num_regs:
                print(f"    Warning: Expected {num_regs} registers, found {actual_num_regs}")
            
            models_dict[num_regs] = model
            configs_dict[num_regs] = config
            print(f"    ✓ Loaded successfully")
            
        except Exception as e:
            print(f"    ✗ Error loading: {e}")
            continue
    
    if not models_dict:
        print("\n✗ No models loaded successfully!")
        return
    
    # Get character classes from first model
    first_config = list(configs_dict.values())[0]
    data_path = Path(first_config['data']['path'])
    
    # Load character classes
    classes_array = np.load(data_path / 'classes.npy', allow_pickle=True)
    
    # Create i2c and c2i mappings if classes is a simple array
    if isinstance(classes_array, np.ndarray) and classes_array.dtype.kind in ['U', 'S', 'O']:
        # It's an array of characters - create dict with mappings
        classes_list = classes_array.tolist()
        classes = {
            'i2c': {i+1: c for i, c in enumerate(classes_list)},  # +1 for CTC blank at 0
            'c2i': {c: i+1 for i, c in enumerate(classes_list)}
        }
        # Add blank token
        classes['i2c'][0] = '<blank>'
    else:
        # Assume it's already a dict
        classes = classes_array.item()
    
    print(f"\n✓ Loaded {len(models_dict)} models")
    
    image_name = image_path.stem
    
    # 1. Generate register comparison visualization (paper style)
    print("\n[1/2] Generating register comparison visualization (paper style)...")
    save_path = register_dir / f'{image_name}_register_comparison.png'
    fig = visualize_register_comparison(
        models_dict,
        configs_dict,
        image_tensor,
        classes,
        layer_type=args.layer_type,
        save_path=save_path
    )
    
    # 2. Generate attention heads visualization for each configuration
    print("\n[2/2] Generating attention heads visualizations...")
    for num_regs in sorted(models_dict.keys()):
        model = models_dict[num_regs]
        print(f"  Processing {num_regs} registers...")
        
        model_wrapper = HTRViTWithAttention(model, layer_type=args.layer_type)
        output = model_wrapper.forward(image_tensor)
        text = decode_prediction(output, classes)
        
        save_path = heads_dir / f'{image_name}_{num_regs}regs_heads.png'
        visualize_attention_heads(
            model,
            image_tensor,
            text,
            num_regs,
            layer_type=args.layer_type,
            save_path=save_path
        )
    
    plt.close('all')
    
    print("\n" + "="*70)
    print("✓ Visualization complete!")
    print(f"✓ Register comparison (paper style): {register_dir}")
    print(f"✓ Attention heads: {heads_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
