#!/usr/bin/env python3
"""
Quick script to verify pretrained model sizes and cache them locally.

This will download models once and cache them in pretrained_models/ directory.
Run this before training to avoid downloads during training.
"""

import os
import sys
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_torchvision_vit(model_name='vit_b_16'):
    """Download and cache TorchVision ViT model."""
    from torchvision.models import get_model
    
    cache_dir = PROJECT_ROOT / 'pretrained_models' / 'torchvision'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    local_path = cache_dir / f'{model_name}.pth'
    
    if local_path.exists():
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"✓ {model_name} already cached ({size_mb:.2f} MB)")
        return local_path, size_mb
    
    print(f"Downloading {model_name}...")
    model = get_model(model_name, weights="DEFAULT")
    
    print(f"Saving to {local_path}...")
    torch.save(model.state_dict(), local_path)
    
    size_mb = local_path.stat().st_size / (1024 * 1024)
    print(f"✓ {model_name} cached successfully ({size_mb:.2f} MB)")
    
    return local_path, size_mb


def download_trocr(model_name='microsoft/trocr-base-handwritten'):
    """Download and cache TrOCR model."""
    from transformers import VisionEncoderDecoderModel, TrOCRProcessor
    
    cache_dir = PROJECT_ROOT / 'pretrained_models' / 'transformers'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {model_name}...")
    print(f"Cache directory: {cache_dir}")
    
    model = VisionEncoderDecoderModel.from_pretrained(
        model_name,
        cache_dir=str(cache_dir)
    )
    processor = TrOCRProcessor.from_pretrained(
        model_name,
        cache_dir=str(cache_dir)
    )
    
    # Calculate cache size
    total_size = 0
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))
    
    size_mb = total_size / (1024 * 1024)
    print(f"✓ {model_name} cached successfully")
    print(f"  Total cache size: {size_mb:.2f} MB")
    
    return cache_dir, size_mb


def main():
    print("="*70)
    print("Pretrained Models Download & Cache Verification")
    print("="*70)
    print()
    
    results = {}
    
    # TorchVision ViT models
    print("TorchVision ViT Models:")
    print("-" * 70)
    
    for model_name in ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32']:
        try:
            path, size = download_torchvision_vit(model_name)
            results[model_name] = size
        except Exception as e:
            print(f"✗ Error with {model_name}: {e}")
        print()
    
    # TrOCR models
    print("\nHuggingFace TrOCR Models:")
    print("-" * 70)
    
    for model_name in ['microsoft/trocr-base-handwritten']:
        try:
            path, size = download_trocr(model_name)
            results[model_name] = size
        except Exception as e:
            print(f"✗ Error with {model_name}: {e}")
        print()
    
    # Summary
    print("\n" + "="*70)
    print("Summary - Cached Model Sizes:")
    print("="*70)
    
    for name, size in results.items():
        print(f"  {name:40s} : {size:>8.2f} MB")
    
    print("\n✓ All models cached successfully!")
    print(f"Cache location: {PROJECT_ROOT / 'pretrained_models'}")
    print()


if __name__ == '__main__':
    main()
