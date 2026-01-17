#!/usr/bin/env python3
"""
Architecture Verification Script
Tests both CNN-RNN and ViT-RGTS architectures to ensure correct forward passes
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from omegaconf import OmegaConf
from models import HTRNet

def test_architecture(config_path, arch_name):
    """Test a specific architecture configuration"""
    print(f"\n{'='*60}")
    print(f"Testing {arch_name}")
    print(f"{'='*60}")
    
    config = OmegaConf.load(config_path)
    
    # Create model
    nclasses = 80  # 79 characters + 1 blank for CTC
    model = HTRNet(config.arch, nclasses)
    model.eval()
    
    print(f"Architecture type: {model.arch_type}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test input: batch_size=2, channels=1, height=128, width=1024
    batch_size = 2
    test_input = torch.randn(batch_size, 1, 128, 1024)
    
    print(f"\nInput shape: {test_input.shape}")
    
    # Forward pass
    try:
        with torch.no_grad():
            output = model(test_input)
            
            if isinstance(output, tuple):
                # For 'both' head type
                main_output, aux_output = output
                print(f"✓ Forward pass successful!")
                print(f"  Main output shape: {main_output.shape}")
                print(f"  Aux output shape: {aux_output.shape}")
                print(f"  Expected: [T, B, C] = [T, {batch_size}, {nclasses}]")
                
                # Verify shapes
                assert main_output.shape[1] == batch_size, f"Batch dimension mismatch!"
                assert main_output.shape[2] == nclasses, f"Class dimension mismatch!"
                
            else:
                print(f"✓ Forward pass successful!")
                print(f"  Output shape: {output.shape}")
                print(f"  Expected: [T, B, C] = [T, {batch_size}, {nclasses}]")
                
                # Verify shapes
                assert output.shape[1] == batch_size, f"Batch dimension mismatch!"
                assert output.shape[2] == nclasses, f"Class dimension mismatch!"
            
            # Check for reasonable output values (not all zeros/ones)
            output_tensor = output[0] if isinstance(output, tuple) else output
            output_mean = output_tensor.mean().item()
            output_std = output_tensor.std().item()
            output_min = output_tensor.min().item()
            output_max = output_tensor.max().item()
            
            print(f"\n  Output statistics:")
            print(f"    Mean: {output_mean:.4f}")
            print(f"    Std:  {output_std:.4f}")
            print(f"    Min:  {output_min:.4f}")
            print(f"    Max:  {output_max:.4f}")
            
            # Check if output has reasonable variance
            if output_std < 0.01:
                print(f"  ⚠️  WARNING: Very low output variance! Model may not be learning.")
            else:
                print(f"  ✓ Output has reasonable variance")
                
        # Test explainability for ViT models
        if model.arch_type == "vit_rgts":
            print(f"\n  Testing explainability (forward_explain)...")
            try:
                with torch.no_grad():
                    logits, reg_tokens, attn_maps, token_norms, grid = model.forward_explain(test_input)
                
                print(f"  ✓ Explainability pass successful!")
                print(f"    Logits shape: {logits.shape}")
                print(f"    Register tokens shape: {reg_tokens.shape}")
                print(f"    Attention maps: {len(attn_maps)} layers")
                if len(attn_maps) > 0:
                    print(f"    First attn map shape: {attn_maps[0].shape} [B, H, S, S]")
                print(f"    Token norms shape: {token_norms.shape}")
                print(f"    Grid size: {grid}")
                
            except Exception as e:
                print(f"  ✗ Explainability failed: {e}")
                import traceback
                traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all architectures"""
    
    print("\n" + "="*60)
    print("HTR ARCHITECTURE VERIFICATION")
    print("="*60)
    
    # Test configurations
    configs = [
        ("configs/config.yaml configs/baseline.yaml", "CNN-RNN Baseline"),
        ("configs/config.yaml configs/baseline_vit_rgts.yaml", "ViT-RGTS with Registers"),
    ]
    
    # For proper loading, we need to merge configs
    results = {}
    
    # Test CNN-RNN
    print("\n" + "="*60)
    print("1. CNN-RNN BASELINE")
    print("="*60)
    base_config = OmegaConf.load("configs/config.yaml")
    arch_config = OmegaConf.load("configs/baseline.yaml")
    config = OmegaConf.merge(base_config, arch_config)
    
    nclasses = 80
    try:
        model = HTRNet(config.arch, nclasses)
        model.eval()
        
        print(f"Architecture type: {model.arch_type}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        test_input = torch.randn(2, 1, 128, 1024)
        with torch.no_grad():
            output = model(test_input)
            if isinstance(output, tuple):
                output = output[0]
            print(f"✓ Forward pass successful!")
            print(f"  Output shape: {output.shape} (expected: [T, 2, 80])")
        results["CNN-RNN"] = True
    except Exception as e:
        print(f"✗ Failed: {e}")
        results["CNN-RNN"] = False
    
    # Test ViT-RGTS
    print("\n" + "="*60)
    print("2. ViT-RGTS WITH REGISTERS")
    print("="*60)
    base_config = OmegaConf.load("configs/config.yaml")
    arch_config = OmegaConf.load("configs/baseline_vit_rgts.yaml")
    config = OmegaConf.merge(base_config, arch_config)
    
    try:
        model = HTRNet(config.arch, nclasses)
        model.eval()
        
        print(f"Architecture type: {model.arch_type}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Register tokens: {model.backbone.num_registers}")
        
        test_input = torch.randn(2, 1, 128, 1024)
        with torch.no_grad():
            output = model(test_input)
            if isinstance(output, tuple):
                output = output[0]
            print(f"✓ Forward pass successful!")
            print(f"  Output shape: {output.shape} (expected: [T, 2, 80])")
            
            # Test explainability
            print(f"\n  Testing attention extraction...")
            logits, reg_tokens, attn_maps, token_norms, grid = model.forward_explain(test_input)
            print(f"  ✓ Attention extraction successful!")
            print(f"    Register tokens: {reg_tokens.shape}")
            print(f"    Attention layers: {len(attn_maps)}")
            print(f"    Grid size: {grid}")
            
        results["ViT-RGTS"] = True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        results["ViT-RGTS"] = False
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{name:30s} {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All architectures verified successfully!")
        return 0
    else:
        print("\n✗ Some architectures failed verification!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
