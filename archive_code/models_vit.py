"""
Vision Transformer (ViT) based HTR Model
Implements ViT-B/16 architecture for Handwritten Text Recognition

Architecture:
    Input Image → Patch Embedding → Transformer Encoder → Sequence Decoder → CTC Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PatchEmbedding(nn.Module):
    """
    Convert image to sequence of patch embeddings.
    ViT-B/16: 16x16 patches
    """
    
    def __init__(self, img_height=128, img_width=1024, patch_size=16, in_channels=1, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches_h = img_height // patch_size
        self.num_patches_w = img_width // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Patch embedding via convolution
        self.projection = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Learnable position embeddings
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim)
        )
        
        # CLS token (optional for HTR, but included for compatibility)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - input image
        Returns:
            [B, N, D] - sequence of patch embeddings
        """
        B = x.shape[0]
        
        # Project patches: [B, C, H, W] -> [B, D, H/P, W/P]
        x = self.projection(x)
        
        # Flatten spatial dimensions: [B, D, H/P, W/P] -> [B, D, N]
        x = x.flatten(2)
        
        # Transpose: [B, D, N] -> [B, N, D]
        x = x.transpose(1, 2)
        
        # Add position embeddings
        x = x + self.position_embeddings
        
        # Optionally add CLS token
        # cls_tokens = self.cls_token.expand(B, -1, -1)
        # x = torch.cat([cls_tokens, x], dim=1)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    Stores attention weights for visualization.
    """
    
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
        # Store attention weights for visualization
        self.attention_weights = None
        
    def forward(self, x):
        """
        Args:
            x: [B, N, D] - input sequence
        Returns:
            [B, N, D] - output sequence
        """
        B, N, D = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D/H]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention: Q @ K^T / sqrt(d)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        # Store attention weights for visualization
        self.attention_weights = attn.detach()
        
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Transformer Encoder Block with Multi-Head Self-Attention and MLP.
    """
    
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, N, D]
        Returns:
            [B, N, D]
        """
        # Attention with residual
        x = x + self.attn(self.norm1(x))
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder.
    Stack of Transformer blocks.
    """
    
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super(ViTEncoder, self).__init__()
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: [B, N, D] - patch embeddings
        Returns:
            [B, N, D] - encoded features
        """
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        return x


class SequenceDecoder(nn.Module):
    """
    Sequence decoder for HTR task.
    Converts ViT output to character sequence predictions.
    """
    
    def __init__(self, embed_dim=768, hidden_dim=512, num_classes=80, decoder_type='rnn'):
        super(SequenceDecoder, self).__init__()
        
        self.decoder_type = decoder_type
        
        if decoder_type == 'rnn':
            # RNN-based decoder
            self.rnn = nn.LSTM(
                embed_dim, 
                hidden_dim, 
                num_layers=2, 
                bidirectional=True, 
                dropout=0.2,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_dim * 2, num_classes)
            
        elif decoder_type == 'linear':
            # Simple linear decoder
            self.fc = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")
    
    def forward(self, x):
        """
        Args:
            x: [B, N, D] - encoded patch features
        Returns:
            [T, B, C] - sequence predictions for CTC
        """
        if self.decoder_type == 'rnn':
            # RNN decoder
            x, _ = self.rnn(x)  # [B, N, H*2]
            x = self.fc(x)      # [B, N, C]
            x = x.permute(1, 0, 2)  # [N, B, C] - CTC expects [T, B, C]
        else:
            # Linear decoder
            x = self.fc(x)      # [B, N, C]
            x = x.permute(1, 0, 2)  # [N, B, C]
        
        return x


class HTRViT(nn.Module):
    """
    Vision Transformer for Handwritten Text Recognition.
    Based on ViT-B/16 architecture.
    
    Args:
        img_height: Input image height (default: 128)
        img_width: Input image width (default: 1024)
        patch_size: Size of image patches (default: 16 for ViT-B/16)
        in_channels: Number of input channels (default: 1 for grayscale)
        num_classes: Number of output classes (characters + blank)
        embed_dim: Embedding dimension (default: 768 for ViT-B)
        depth: Number of transformer blocks (default: 12 for ViT-B)
        num_heads: Number of attention heads (default: 12 for ViT-B)
        mlp_ratio: MLP hidden dim ratio (default: 4.0)
        dropout: Dropout rate (default: 0.1)
        decoder_type: Type of sequence decoder ('rnn' or 'linear')
    """
    
    def __init__(
        self,
        img_height=128,
        img_width=1024,
        patch_size=16,
        in_channels=1,
        num_classes=80,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        decoder_type='rnn'
    ):
        super(HTRViT, self).__init__()
        
        self.model_type = 'vit'  # Identifier for visualization
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_height, img_width, patch_size, in_channels, embed_dim
        )
        
        # Transformer encoder
        self.encoder = ViTEncoder(embed_dim, depth, num_heads, mlp_ratio, dropout)
        
        # Sequence decoder
        self.decoder = SequenceDecoder(embed_dim, embed_dim // 2, num_classes, decoder_type)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - input image
        Returns:
            [T, B, Classes] - sequence predictions for CTC
        """
        # Patch embedding
        x = self.patch_embed(x)  # [B, N, D]
        
        # Transformer encoding
        x = self.encoder(x)      # [B, N, D]
        
        # Sequence decoding
        x = self.decoder(x)      # [T, B, C]
        
        return x
    
    def get_attention_weights(self):
        """
        Extract attention weights from all transformer blocks.
        Returns list of attention weight tensors for visualization.
        """
        attention_weights = []
        for block in self.encoder.blocks:
            if hasattr(block.attn, 'attention_weights') and block.attn.attention_weights is not None:
                attention_weights.append(block.attn.attention_weights)
        return attention_weights


def create_vit_htr_model(config, num_classes):
    """
    Factory function to create ViT-HTR model from config.
    
    Args:
        config: OmegaConf config object with arch_vit settings
        num_classes: Number of output classes
    
    Returns:
        HTRViT model instance
    """
    arch_config = config.arch_vit
    
    model = HTRViT(
        img_height=config.preproc.image_height,
        img_width=config.preproc.image_width,
        patch_size=arch_config.get('patch_size', 16),
        in_channels=1,
        num_classes=num_classes,
        embed_dim=arch_config.get('embed_dim', 768),
        depth=arch_config.get('depth', 12),
        num_heads=arch_config.get('num_heads', 12),
        mlp_ratio=arch_config.get('mlp_ratio', 4.0),
        dropout=arch_config.get('dropout', 0.1),
        decoder_type=arch_config.get('decoder_type', 'rnn')
    )
    
    return model


if __name__ == '__main__':
    # Test the model
    print("Testing HTRViT model...")
    
    # Create model (ViT-B/16 configuration)
    model = HTRViT(
        img_height=128,
        img_width=1024,
        patch_size=16,
        num_classes=80,
        embed_dim=768,
        depth=12,
        num_heads=12
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {n_params:,}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 1, 128, 1024)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")  # [T, B, C]
        
        # Test attention extraction
        attn_weights = model.get_attention_weights()
        print(f"Number of attention layers: {len(attn_weights)}")
        if len(attn_weights) > 0:
            print(f"Attention shape (layer 0): {attn_weights[0].shape}")  # [B, H, N, N]
    
    print("\n✓ HTRViT model test passed!")
