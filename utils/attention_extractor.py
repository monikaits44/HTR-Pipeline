"""
Enhanced Explainability Module for HTR
Extracts attention maps from transformer blocks for analysis
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


class AttentionExtractor:
    """
    Extract and analyze attention maps from ViT transformer blocks.
    Supports multi-head attention analysis and layer-wise attention flow.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize attention extractor.
        
        Args:
            model: HTRNet model with vit_rgts architecture
        """
        self.model = model
        self.hooks = []
        self.attentions = {}
        
        if not hasattr(model, 'arch_type') or model.arch_type != 'vit_rgts':
            raise ValueError("AttentionExtractor only works with vit_rgts architecture")
    
    def extract_from_forward(self, images: torch.Tensor) -> Dict:
        """
        Extract attention using the built-in forward_explain method.
        
        Args:
            images: Input images [B, 1, H, W]
            
        Returns:
            Dictionary with:
                - logits: [T, B, C]
                - register_tokens: [B, R, D]
                - attention_maps: List of [B, H, S, S] per layer
                - token_norms: [B, S]
                - grid_size: (Hp, Wp)
        """
        self.model.eval()
        
        with torch.no_grad():
            logits, reg_tokens, attn_maps, token_norms, grid = self.model.forward_explain(images)
        
        return {
            'logits': logits,
            'register_tokens': reg_tokens,
            'attention_maps': attn_maps,
            'token_norms': token_norms,
            'grid_size': grid,
            'num_layers': len(attn_maps),
            'num_heads': attn_maps[0].shape[1] if len(attn_maps) > 0 else 0
        }
    
    def analyze_register_attention(
        self, 
        attention_maps: List[torch.Tensor],
        num_registers: int
    ) -> Dict:
        """
        Analyze how register tokens interact with patch tokens.
        
        Args:
            attention_maps: List of [B, H, S, S] attention maps per layer
            num_registers: Number of register tokens
            
        Returns:
            Dictionary with register attention statistics
        """
        results = {}
        
        for layer_idx, attn in enumerate(attention_maps):
            # attn: [B, H, S, S]
            B, H, S, _ = attn.shape
            
            # Extract attention FROM registers TO patches
            # Registers are first R tokens, patches are rest
            reg_to_patch = attn[:, :, :num_registers, num_registers:]  # [B, H, R, P]
            
            # Extract attention FROM patches TO registers
            patch_to_reg = attn[:, :, num_registers:, :num_registers]  # [B, H, P, R]
            
            # Average across batch and heads
            reg_to_patch_mean = reg_to_patch.mean(dim=(0, 1))  # [R, P]
            patch_to_reg_mean = patch_to_reg.mean(dim=(0, 1))  # [P, R]
            
            results[f'layer_{layer_idx}'] = {
                'register_to_patch': reg_to_patch_mean.cpu().numpy(),
                'patch_to_register': patch_to_reg_mean.cpu().numpy(),
                'register_attention_entropy': self._compute_entropy(reg_to_patch),
                'patch_attention_entropy': self._compute_entropy(patch_to_reg),
            }
        
        return results
    
    def get_layer_attention_maps(
        self,
        images: torch.Tensor,
        layer_indices: Optional[List[int]] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Extract attention maps from specific transformer layers.
        
        Args:
            images: Input images [B, 1, H, W]
            layer_indices: Which layers to extract (None = all)
            
        Returns:
            Dictionary mapping layer_idx -> attention map [B, H, S, S]
        """
        result = self.extract_from_forward(images)
        attn_maps = result['attention_maps']
        
        if layer_indices is None:
            layer_indices = list(range(len(attn_maps)))
        
        return {idx: attn_maps[idx] for idx in layer_indices if idx < len(attn_maps)}
    
    def visualize_attention_flow(
        self,
        images: torch.Tensor,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize how attention evolves through transformer layers.
        
        Args:
            images: Input images [B, 1, H, W]
            save_path: Optional path to save visualization
            
        Returns:
            Attention flow visualization as numpy array
        """
        result = self.extract_from_forward(images)
        attn_maps = result['attention_maps']
        num_registers = result['register_tokens'].shape[1]
        
        # Analyze register attention across layers
        register_stats = self.analyze_register_attention(attn_maps, num_registers)
        
        # Create visualization (placeholder for now)
        print(f"Extracted attention from {len(attn_maps)} transformer layers")
        for layer_idx, stats in register_stats.items():
            print(f"  {layer_idx}: entropy = {stats['register_attention_entropy']:.4f}")
        
        return None  # Would return actual visualization
    
    def _compute_entropy(self, attn: torch.Tensor) -> float:
        """
        Compute Shannon entropy of attention distribution.
        Higher entropy = more distributed attention
        Lower entropy = more focused attention
        
        Args:
            attn: Attention tensor [B, H, S1, S2]
            
        Returns:
            Mean entropy across batch and heads
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        attn = attn + eps
        
        # Compute entropy: -sum(p * log(p))
        entropy = -(attn * torch.log(attn)).sum(dim=-1)  # [B, H, S1]
        
        return entropy.mean().item()
    
    def extract_patch_attention(
        self,
        images: torch.Tensor,
        layer_idx: int = -1
    ) -> np.ndarray:
        """
        Extract attention map for patches (excluding registers).
        Useful for visualizing which image regions the model focuses on.
        
        Args:
            images: Input images [B, 1, H, W]
            layer_idx: Which layer (-1 = last layer)
            
        Returns:
            Patch attention map [B, Hp, Wp]
        """
        result = self.extract_from_forward(images)
        attn_maps = result['attention_maps']
        num_registers = result['register_tokens'].shape[1]
        Hp, Wp = result['grid_size']
        
        if layer_idx < 0:
            layer_idx = len(attn_maps) + layer_idx
        
        # Get attention for this layer
        attn = attn_maps[layer_idx]  # [B, H, S, S]
        
        # Extract patch-to-patch attention (exclude registers)
        patch_attn = attn[:, :, num_registers:, num_registers:]  # [B, H, P, P]
        
        # Average across heads and source tokens
        patch_attn_mean = patch_attn.mean(dim=(1, 2))  # [B, P]
        
        # Reshape to spatial grid
        B = patch_attn_mean.shape[0]
        patch_attn_spatial = patch_attn_mean.view(B, Hp, Wp)  # [B, Hp, Wp]
        
        return patch_attn_spatial.cpu().numpy()


def extract_mid_level_features(
    model: nn.Module,
    images: torch.Tensor,
    layer_range: Tuple[int, int] = (2, 4)
) -> Dict[str, torch.Tensor]:
    """
    Extract features and attention from mid-level transformer blocks.
    
    This is analogous to extracting from "mid-level diffusion blocks" 
    but for transformer architecture.
    
    Args:
        model: HTRNet with vit_rgts architecture
        images: Input images [B, 1, H, W]
        layer_range: Range of layers to extract (start, end)
        
    Returns:
        Dictionary with mid-level features and attention
    """
    extractor = AttentionExtractor(model)
    result = extractor.extract_from_forward(images)
    
    start_layer, end_layer = layer_range
    attn_maps = result['attention_maps']
    
    # Extract mid-level attention
    mid_level_attention = attn_maps[start_layer:end_layer]
    
    return {
        'mid_level_attention': mid_level_attention,
        'layer_range': (start_layer, end_layer),
        'num_layers': len(mid_level_attention),
        'shape_info': f"[B, H, S, S] = {mid_level_attention[0].shape if len(mid_level_attention) > 0 else 'N/A'}"
    }
