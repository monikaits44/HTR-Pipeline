#!/usr/bin/env python3
"""
Comprehensive demo script for explainable HTR system.

Demonstrates:
1. Loading different model architectures (ViT-RGTS, TorchVision ViT, TrOCR, CNN-RNN)
2. Making predictions on sample images
3. Extracting and visualizing attention maps
4. Analyzing register tokens (where applicable)
5. Comparing explainability across architectures

Usage:
    python scripts/postprocessing/demo_explainability.py \
        --config configs/config.yaml \
        --arch-config configs/baseline_vit_rgts.yaml \
        --resume saved_models/experiments/run_X/best_model.pt \
        --image-path data/IAM/processed_lines/test/c04-165-05.png \
        --output-dir demo_output

"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from omegaconf import OmegaConf

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import HTRNet
from utils.htr_dataset import HTRDataset
from utils.transforms import aug_transforms


def load_model(config, checkpoint_path, device='cuda'):
    """Load trained HTR model."""
    print(f"Loading model from {checkpoint_path}")
    
    # Load classes
    classes_path = os.path.join(config.data.path, 'classes.npy')
    classes = np.load(classes_path, allow_pickle=True)
    nclasses = len(classes) + 1  # +1 for CTC blank
    
    print(f"Number of classes: {nclasses}")
    print(f"Architecture type: {config.arch.type}")
    
    # Initialize model
    model = HTRNet(config.arch, nclasses)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    return model, classes


def load_and_preprocess_image(image_path, config):
    """Load and preprocess image for model input."""
    # Load image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Resize to model input size
    target_h = config.preproc.image_height
    target_w = config.preproc.image_width
    
    img = img.resize((target_w, target_h), Image.BILINEAR)
    
    # Convert to tensor
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    return img_tensor, img


def decode_prediction(logits, classes):
    """Decode CTC predictions."""
    # logits: [T, B, C]
    logits = logits.squeeze(1)  # [T, C]
    pred = torch.argmax(logits, dim=-1).cpu().numpy()
    
    # CTC decode - remove blanks and repeated characters
    decoded = []
    prev_char = None
    for idx in pred:
        if idx != 0 and idx != prev_char:  # 0 is CTC blank
            if idx <= len(classes):
                decoded.append(classes[idx - 1])
        prev_char = idx
    
    return ''.join(decoded)


def visualize_attention_maps(attn_maps, image, save_path=None):
    """
    Visualize multi-layer, multi-head attention maps.
    
    Args:
        attn_maps: List of attention tensors [B, H, S, S] per layer
        image: Original PIL image
        save_path: Where to save the visualization
    """
    num_layers = len(attn_maps)
    num_heads = attn_maps[0].shape[1]
    
    # Select middle layer for visualization (most semantic)
    mid_layer = num_layers // 2
    attn = attn_maps[mid_layer][0]  # [H, S, S]
    
    # Average across query tokens (or select CLS/first register)
    # For register tokens: first few positions
    # For patch tokens: rest
    
    # Visualize first 4 heads from middle layer
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Select heads to visualize
    heads_to_show = min(5, num_heads)
    for idx in range(heads_to_show):
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        
        # Get attention from this head
        head_attn = attn[idx].numpy()  # [S, S]
        
        # Visualize attention from CLS/first register token
        attn_from_first = head_attn[0, :]  # Attention FROM first token TO all tokens
        
        im = axes[row, col].imshow(attn_from_first.reshape(-1, 1), cmap='hot', aspect='auto')
        axes[row, col].set_title(f'Layer {mid_layer}, Head {idx}')
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Attention map saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_register_tokens(reg_tokens, save_path=None):
    """
    Visualize register token embeddings.
    
    Args:
        reg_tokens: [B, R, D] tensor of register embeddings
        save_path: Where to save the visualization
    """
    if reg_tokens is None:
        print("No register tokens available for this architecture")
        return
    
    reg_tokens = reg_tokens[0].cpu().numpy()  # [R, D]
    num_regs, dim = reg_tokens.shape
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Visualize as heatmap
    im0 = axes[0].imshow(reg_tokens, cmap='viridis', aspect='auto')
    axes[0].set_title(f'Register Token Embeddings ({num_regs} tokens, {dim} dims)')
    axes[0].set_xlabel('Embedding Dimension')
    axes[0].set_ylabel('Register Token Index')
    plt.colorbar(im0, ax=axes[0])
    
    # Visualize L2 norms
    norms = np.linalg.norm(reg_tokens, axis=1)
    axes[1].bar(range(num_regs), norms)
    axes[1].set_title('Register Token L2 Norms')
    axes[1].set_xlabel('Register Token Index')
    axes[1].set_ylabel('L2 Norm')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Register token visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_token_norms(token_norms, num_registers, save_path=None):
    """
    Visualize token norms across the sequence.
    
    Args:
        token_norms: [B, S] tensor of token L2 norms
        num_registers: Number of register tokens
        save_path: Where to save the visualization
    """
    token_norms = token_norms[0].numpy()  # [S]
    
    plt.figure(figsize=(14, 4))
    
    # Color register tokens differently
    colors = ['red' if i < num_registers else 'blue' for i in range(len(token_norms))]
    
    plt.bar(range(len(token_norms)), token_norms, color=colors, alpha=0.6)
    plt.axvline(x=num_registers - 0.5, color='black', linestyle='--', 
                label=f'Register/Patch boundary ({num_registers} registers)')
    plt.title('Token L2 Norms (Red=Register, Blue=Patch)')
    plt.xlabel('Token Index')
    plt.ylabel('L2 Norm')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Token norms saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def run_demo(config, checkpoint_path, image_path, output_dir, device='cuda'):
    """Run comprehensive demo."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("Explainable HTR Demo")
    print("="*80)
    
    # Load model
    model, classes = load_model(config, checkpoint_path, device)
    
    # Load and preprocess image
    print(f"\nLoading image: {image_path}")
    img_tensor, img_pil = load_and_preprocess_image(image_path, config)
    img_tensor = img_tensor.to(device)
    
    # Save preprocessed image
    img_pil.save(os.path.join(output_dir, 'input_image.png'))
    
    # Make prediction
    print("\n" + "="*80)
    print("Standard Prediction")
    print("="*80)
    
    with torch.no_grad():
        logits = model(img_tensor)  # [T, B, C]
    
    predicted_text = decode_prediction(logits, classes)
    print(f"Predicted text: {predicted_text}")
    
    # Check if model supports explainability
    arch_type = config.arch.type
    print(f"\nArchitecture: {arch_type}")
    
    if arch_type in ['vit_rgts', 'torchvision_vit', 'trocr']:
        print("\n" + "="*80)
        print("Explainability Analysis")
        print("="*80)
        
        # Get explainability outputs
        result = model.forward_explain(img_tensor)
        logits_explain, reg_tokens, attn_maps, token_norms, grid = result
        
        print(f"Number of attention layers: {len(attn_maps)}")
        print(f"Attention shape per layer: {attn_maps[0].shape}")  # [B, H, S, S]
        print(f"Grid size (H x W): {grid}")
        
        # Visualize attention maps
        print("\nGenerating attention visualizations...")
        visualize_attention_maps(
            attn_maps, 
            img_pil, 
            save_path=os.path.join(output_dir, 'attention_maps.png')
        )
        
        # Visualize register tokens (if available)
        if reg_tokens is not None:
            num_regs = reg_tokens.shape[1]
            print(f"\nNumber of register tokens: {num_regs}")
            visualize_register_tokens(
                reg_tokens,
                save_path=os.path.join(output_dir, 'register_tokens.png')
            )
            
            # Visualize token norms
            visualize_token_norms(
                token_norms,
                num_regs,
                save_path=os.path.join(output_dir, 'token_norms.png')
            )
        else:
            print("\nNo register tokens for this architecture")
            # Still visualize token norms
            visualize_token_norms(
                token_norms,
                0,
                save_path=os.path.join(output_dir, 'token_norms.png')
            )
        
        # Save features for downstream tasks
        np.save(os.path.join(output_dir, 'attention_maps.npy'), 
                [a.numpy() for a in attn_maps])
        if reg_tokens is not None:
            np.save(os.path.join(output_dir, 'register_tokens.npy'), 
                    reg_tokens.cpu().numpy())
        np.save(os.path.join(output_dir, 'token_norms.npy'), 
                token_norms.numpy())
        
        print("\n✓ All visualizations and features saved to:", output_dir)
        
    else:
        print(f"\nNote: Architecture '{arch_type}' does not support explainability features")
        print("Try using: vit_rgts, torchvision_vit, or trocr")
    
    # Summary
    print("\n" + "="*80)
    print("Demo Summary")
    print("="*80)
    print(f"Image: {image_path}")
    print(f"Predicted: {predicted_text}")
    print(f"Architecture: {arch_type}")
    print(f"Output directory: {output_dir}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Explainable HTR Demo")
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Base config file (e.g., configs/config.yaml)'
    )
    parser.add_argument(
        '--arch-config',
        type=str,
        required=True,
        help='Architecture config file (e.g., configs/baseline_vit_rgts.yaml)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--image-path',
        type=str,
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='demo_output',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    
    args = parser.parse_args()
    
    # Load and merge configs
    config = OmegaConf.load(args.config)
    arch_config = OmegaConf.load(args.arch_config)
    config = OmegaConf.merge(config, arch_config)
    
    # Run demo
    run_demo(
        config=config,
        checkpoint_path=args.resume,
        image_path=args.image_path,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == '__main__':
    main()
