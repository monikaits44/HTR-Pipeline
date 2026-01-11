#!/usr/bin/env python3
"""
Character-Level Attention Visualization for HTR

This script:
1. Loads line images from train/val/test sets
2. Extracts attention maps from ViT models
3. Maps attention to individual character patches
4. Visualizes character-level attention localization

DESIGN DECISIONS:
- Uses PADDING (not resize) to preserve character shapes
- Maps patch tokens to character positions
- Shows attention flow: character → patches
- Useful for explainability and writer identification

Usage:
    python scripts/postprocessing/visualize_character_attention.py \
        --config configs/config.yaml \
        --arch-config configs/baseline_vit_rgts.yaml \
        --resume saved_models/experiments/run_X/best_model.pt \
        --image-path data/IAM/processed_lines/test/sample.png \
        --gt-text "ground truth text" \
        --output-dir char_attention_output
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import torch
import cv2
from omegaconf import OmegaConf

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import HTRNet
from utils.htr_dataset import HTRDataset


def pad_patch(patch, target_size=(64, 32)):
    """
    Pad character patch to fixed size WITHOUT distorting aspect ratio.
    
    WHY PADDING vs RESIZE:
    - Preserves character shape (crucial for attention localization)
    - Maintains aspect ratio
    - Padding is "neutral" (white space) vs distortion
    - Better for explainability and downstream tasks
    
    Args:
        patch: numpy array [H, W] (grayscale)
        target_size: (height, width) tuple
    
    Returns:
        padded patch [target_h, target_w]
    """
    h, w = patch.shape
    target_h, target_w = target_size
    
    # Calculate padding needed
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    
    # Center the patch
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # Pad with white (255 for grayscale)
    padded = np.pad(
        patch,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=255
    )
    
    # Crop if patch is larger than target
    if h > target_h or w > target_w:
        start_h = max(0, (h - target_h) // 2)
        start_w = max(0, (w - target_w) // 2)
        padded = patch[start_h:start_h+target_h, start_w:start_w+target_w]
    
    return padded


def extract_character_patches(image, gt_text, patch_height=32, min_width=8):
    """
    Extract character patches from line image.
    
    Strategy:
    1. Use connected components to find character regions
    2. Sort left-to-right to match text order
    3. Associate with ground truth characters
    4. Return patches with metadata
    
    Args:
        image: PIL Image or numpy array [H, W]
        gt_text: ground truth text string
        patch_height: target height for patches
        min_width: minimum width to consider as character
    
    Returns:
        List of dicts with:
        - 'patch': numpy array
        - 'bbox': (x, y, w, h)
        - 'char': character
        - 'idx': character index
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image.convert('L'))
    else:
        img_array = image
    
    # Binarize image
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Extract character bounding boxes
    char_bboxes = []
    for i in range(1, num_labels):  # Skip background (0)
        x, y, w, h, area = stats[i]
        
        # Filter noise (too small)
        if w < min_width or h < 5 or area < 20:
            continue
        
        char_bboxes.append({
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'centroid': centroids[i]
        })
    
    # Sort left-to-right
    char_bboxes.sort(key=lambda b: b['x'])
    
    # Extract patches and associate with characters
    patches = []
    gt_chars = list(gt_text)
    
    for idx, bbox in enumerate(char_bboxes):
        if idx >= len(gt_chars):
            break  # More components than characters
        
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
        
        # Extract patch from original image
        patch = img_array[y:y+h, x:x+w]
        
        # Pad to consistent size
        padded_patch = pad_patch(patch, target_size=(patch_height, 32))
        
        patches.append({
            'patch': padded_patch,
            'bbox': (x, y, w, h),
            'char': gt_chars[idx],
            'idx': idx,
            'original_patch': patch  # Keep original for reference
        })
    
    return patches


def map_characters_to_patches(char_bboxes, grid_size, image_size):
    """
    Map character bounding boxes to ViT patch token positions.
    
    Args:
        char_bboxes: List of character bbox dicts
        grid_size: (Hp, Wp) - patch grid dimensions
        image_size: (H, W) - image dimensions
    
    Returns:
        List of patch indices for each character
    """
    Hp, Wp = grid_size
    H, W = image_size
    
    patch_h = H / Hp
    patch_w = W / Wp
    
    char_to_patches = []
    
    for char_info in char_bboxes:
        x, y, w, h = char_info['bbox']
        
        # Find overlapping patches
        start_row = int(y / patch_h)
        end_row = min(int((y + h) / patch_h) + 1, Hp)
        start_col = int(x / patch_w)
        end_col = min(int((x + w) / patch_w) + 1, Wp)
        
        # Get patch indices
        patch_indices = []
        for r in range(start_row, end_row):
            for c in range(start_col, end_col):
                patch_idx = r * Wp + c
                patch_indices.append(patch_idx)
        
        char_to_patches.append(patch_indices)
    
    return char_to_patches


def visualize_character_attention(
    image, 
    char_patches, 
    char_to_patches_map,
    attention_maps,
    grid_size,
    layer_idx=-2,
    save_path=None
):
    """
    Visualize attention at character level.
    
    Args:
        image: Original line image
        char_patches: List of character patch dicts
        char_to_patches_map: Mapping from chars to patch indices
        attention_maps: List of attention tensors [B, H, S, S]
        grid_size: (Hp, Wp)
        layer_idx: Which layer to visualize (-2 = second to last)
        save_path: Where to save visualization
    """
    if len(attention_maps) == 0:
        print("No attention maps available")
        return
    
    # Select layer
    attn = attention_maps[layer_idx][0]  # [H, S, S]
    num_heads = attn.shape[0]
    
    # Average across heads
    attn_avg = attn.mean(dim=0).numpy()  # [S, S]
    
    Hp, Wp = grid_size
    num_patches = Hp * Wp
    
    # Create figure
    num_chars = len(char_patches)
    fig = plt.figure(figsize=(20, 4 * ((num_chars + 2) // 3)))
    
    # Original image at top
    ax_img = plt.subplot(((num_chars + 2) // 3) + 1, 3, 1)
    ax_img.imshow(image, cmap='gray')
    ax_img.set_title('Original Line Image', fontsize=12, fontweight='bold')
    ax_img.axis('off')
    
    # Draw character bounding boxes
    for char_info in char_patches:
        x, y, w, h = char_info['bbox']
        rect = Rectangle((x, y), w, h, linewidth=2, 
                        edgecolor='red', facecolor='none')
        ax_img.add_patch(rect)
        # Add character label
        ax_img.text(x + w/2, y - 5, char_info['char'], 
                   color='red', fontsize=10, ha='center', fontweight='bold')
    
    # Character-level attention visualization
    for idx, char_info in enumerate(char_patches):
        ax = plt.subplot(((num_chars + 2) // 3) + 1, 3, idx + 4)
        
        # Get patch indices for this character
        if idx < len(char_to_patches_map):
            patch_indices = char_to_patches_map[idx]
            
            # Extract attention from character patches to all patches
            if len(patch_indices) > 0:
                # Average attention from all patches of this character
                char_attn = attn_avg[patch_indices, :].mean(axis=0)
                
                # Focus on patch tokens only (skip CLS/register if present)
                # Assuming: [CLS, REG0, REG1, ..., PATCH0, PATCH1, ...]
                # We need to know num_registers
                num_registers = attn.shape[1] - num_patches - 1  # Assuming CLS + registers + patches
                num_registers = max(0, num_registers)
                
                patch_start = 1 + num_registers
                patch_attn = char_attn[patch_start:patch_start + num_patches]
                
                # Reshape to grid
                attn_grid = patch_attn.reshape(Hp, Wp)
                
                # Overlay on image
                img_resized = cv2.resize(np.array(image), (Wp * 16, Hp * 16))
                attn_resized = cv2.resize(attn_grid, (Wp * 16, Hp * 16))
                
                # Show attention heatmap
                ax.imshow(img_resized, cmap='gray', alpha=0.6)
                im = ax.imshow(attn_resized, cmap='hot', alpha=0.4)
                plt.colorbar(im, ax=ax, fraction=0.046)
                
                # Highlight character region
                x, y, w, h = char_info['bbox']
                rect = Rectangle((x, y), w, h, linewidth=3, 
                                edgecolor='lime', facecolor='none')
                ax.add_patch(rect)
        
        ax.set_title(f"'{char_info['char']}' (idx={char_info['idx']})", 
                    fontsize=10, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved character attention to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def load_model(config, checkpoint_path, device='cuda'):
    """Load trained HTR model."""
    classes_path = os.path.join(config.data.path, 'classes.npy')
    classes = np.load(classes_path, allow_pickle=True)
    nclasses = len(classes) + 1
    
    model = HTRNet(config.arch, nclasses)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, classes


def preprocess_image(image_path, config):
    """Load and preprocess image."""
    img = Image.open(image_path).convert('L')
    
    target_h = config.preproc.image_height
    target_w = config.preproc.image_width
    
    img_resized = img.resize((target_w, target_h), Image.BILINEAR)
    
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    
    return img_tensor, img, img_resized


def main():
    parser = argparse.ArgumentParser(description="Character-Level Attention Visualization")
    
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--arch-config', type=str, required=True)
    parser.add_argument('--resume', type=str, required=True)
    parser.add_argument('--image-path', type=str, required=True)
    parser.add_argument('--gt-text', type=str, required=True, 
                       help="Ground truth text for the image")
    parser.add_argument('--output-dir', type=str, default='char_attention_output')
    parser.add_argument('--layer-idx', type=int, default=-2,
                       help="Which attention layer to visualize (negative = from end)")
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Load configs
    config = OmegaConf.load(args.config)
    arch_config = OmegaConf.load(args.arch_config)
    config = OmegaConf.merge(config, arch_config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("Character-Level Attention Visualization")
    print("="*80)
    
    # Load model
    print(f"\nLoading model from: {args.resume}")
    model, classes = load_model(config, args.resume, args.device)
    
    # Load and preprocess image
    print(f"Loading image: {args.image_path}")
    img_tensor, img_original, img_resized = preprocess_image(args.image_path, config)
    img_tensor = img_tensor.to(args.device)
    
    # Extract character patches
    print(f"\nExtracting character patches for: '{args.gt_text}'")
    char_patches = extract_character_patches(img_original, args.gt_text)
    print(f"Found {len(char_patches)} character patches")
    
    # Get attention maps
    print("\nExtracting attention maps...")
    arch_type = config.arch.type
    
    if arch_type in ['vit_rgts', 'torchvision_vit', 'trocr']:
        result = model.forward_explain(img_tensor)
        logits, reg_tokens, attn_maps, token_norms, grid = result
        
        print(f"Grid size: {grid}")
        print(f"Number of layers: {len(attn_maps)}")
        
        # Map characters to patch positions
        img_size = (config.preproc.image_height, config.preproc.image_width)
        char_to_patches = map_characters_to_patches(char_patches, grid, img_size)
        
        # Visualize
        print("\nGenerating character-level attention visualization...")
        save_path = os.path.join(args.output_dir, 'character_attention.png')
        visualize_character_attention(
            img_resized,
            char_patches,
            char_to_patches,
            attn_maps,
            grid,
            layer_idx=args.layer_idx,
            save_path=save_path
        )
        
        # Save individual character patches
        patches_dir = os.path.join(args.output_dir, 'character_patches')
        os.makedirs(patches_dir, exist_ok=True)
        
        for char_info in char_patches:
            patch_path = os.path.join(
                patches_dir, 
                f"char_{char_info['idx']:02d}_{char_info['char']}.png"
            )
            Image.fromarray(char_info['patch']).save(patch_path)
        
        print(f"\n✓ Visualization saved to: {save_path}")
        print(f"✓ Character patches saved to: {patches_dir}")
        
    else:
        print(f"Error: Architecture '{arch_type}' does not support attention extraction")
        print("Please use: vit_rgts, torchvision_vit, or trocr")
        return
    
    print("\n" + "="*80)
    print("Character-Level Attention Analysis Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
