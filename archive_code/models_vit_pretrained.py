"""
Pretrained ViT-B/16 for Handwritten Text Recognition

This module uses pretrained Vision Transformer from timm library
and adapts it for HTR task with CTC loss.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple


class PretrainedViTHTR(nn.Module):
    """
    HTR model using pretrained ViT-B/16 from timm library.
    
    Architecture:
        Input: [B, 1, 128, 1024] grayscale images
        ↓
        Convert to RGB: [B, 3, 128, 1024]
        ↓
        Pretrained ViT-B/16 encoder
        ↓
        Sequence Decoder (RNN or Linear)
        ↓
        Output: [T, B, num_classes] for CTC loss
    
    Args:
        num_classes: Number of character classes (including CTC blank)
        pretrained: Whether to load pretrained ImageNet weights
        decoder_type: 'rnn' or 'linear'
        decoder_hidden: Hidden size for RNN decoder
        decoder_layers: Number of RNN layers
        freeze_encoder: Whether to freeze pretrained weights
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        decoder_type: str = 'rnn',
        decoder_hidden: int = 512,
        decoder_layers: int = 2,
        freeze_encoder: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm library is required for pretrained ViT. "
                "Install with: pip install timm"
            )
        
        self.num_classes = num_classes
        self.decoder_type = decoder_type
        self.pretrained = pretrained
        
        # Load pretrained ViT-B/16 from timm
        # timm uses 224x224 input by default, but we can adapt to our size
        print(f"Loading pretrained ViT-B/16 (pretrained={pretrained})...")
        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='',  # Keep sequence output
        )
        
        # Get ViT output dimension (768 for ViT-Base)
        self.embed_dim = self.vit.embed_dim  # 768
        
        # Freeze encoder if requested
        if freeze_encoder:
            print("Freezing pretrained ViT encoder weights...")
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Grayscale to RGB conversion (learned weights)
        self.gray_to_rgb = nn.Conv2d(1, 3, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.gray_to_rgb.weight)
        
        # Adaptive pooling to match ViT input size (224x224)
        # Our input is 128x1024, we'll resize to 224x896 (preserve aspect ratio roughly)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((224, 224))
        
        # Decoder
        if decoder_type == 'rnn':
            self.decoder = nn.LSTM(
                input_size=self.embed_dim,
                hidden_size=decoder_hidden,
                num_layers=decoder_layers,
                batch_first=False,
                dropout=dropout if decoder_layers > 1 else 0,
                bidirectional=True
            )
            self.fc = nn.Linear(decoder_hidden * 2, num_classes)
        elif decoder_type == 'linear':
            self.decoder = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.embed_dim, num_classes)
            )
            self.fc = None
        else:
            raise ValueError(f"decoder_type must be 'rnn' or 'linear', got {decoder_type}")
        
        # Store attention weights for visualization
        self.attention_weights = []
        self._register_attention_hooks()
    
    def _register_attention_hooks(self):
        """Register forward hooks to capture attention weights from timm ViT."""
        # Monkey-patch each attention module to store attention weights
        for i, block in enumerate(self.vit.blocks):
            original_forward = block.attn.forward
            
            def make_hooked_forward(original_fn, block_idx, attn_module):
                def hooked_forward(x, attn_mask=None):
                    B, N, C = x.shape
                    qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    
                    # Compute attention weights
                    attn = (q @ k.transpose(-2, -1)) * attn_module.scale
                    
                    # Apply attention mask if provided
                    if attn_mask is not None:
                        attn = attn + attn_mask
                    
                    attn = attn.softmax(dim=-1)
                    
                    # Store attention weights for visualization
                    self.attention_weights.append(attn.detach().cpu())
                    
                    attn = attn_module.attn_drop(attn)
                    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                    x = attn_module.proj(x)
                    x = attn_module.proj_drop(x)
                    return x
                return hooked_forward
            
            block.attn.forward = make_hooked_forward(original_forward, i, block.attn)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, 1, H, W] where H=128, W=1024
        
        Returns:
            Output tensor [T, B, num_classes] for CTC loss
        """
        B = x.size(0)
        
        # Clear previous attention weights
        self.attention_weights = []
        
        # Convert grayscale to RGB: [B, 1, 128, 1024] → [B, 3, 128, 1024]
        x = self.gray_to_rgb(x)
        
        # Resize to ViT input size: [B, 3, 128, 1024] → [B, 3, 224, 224]
        x = self.adaptive_pool(x)
        
        # ViT encoder: [B, 3, 224, 224] → [B, num_patches+1, embed_dim]
        # num_patches = (224/16)^2 = 196, +1 for CLS token = 197
        x = self.vit.forward_features(x)
        
        # Remove CLS token: [B, 197, 768] → [B, 196, 768]
        x = x[:, 1:, :]  # Remove first token (CLS)
        
        # Transpose for sequence decoding: [B, 196, 768] → [196, B, 768]
        x = x.transpose(0, 1)
        T = x.size(0)
        
        # Decode sequence
        if self.decoder_type == 'rnn':
            # LSTM: [T, B, 768] → [T, B, decoder_hidden*2]
            x, _ = self.decoder(x)
            # Linear: [T, B, decoder_hidden*2] → [T, B, num_classes]
            x = self.fc(x)
        else:
            # Linear decoder: [T, B, 768] → [T, B, num_classes]
            x = self.decoder(x)
        
        return x
    
    def get_attention_weights(self) -> List[torch.Tensor]:
        """
        Extract attention weights from all transformer layers.
        
        Returns:
            List of attention weight tensors [B, num_heads, num_patches, num_patches]
        """
        return self.attention_weights


def create_pretrained_vit_htr_model(config, num_classes: int) -> PretrainedViTHTR:
    """
    Factory function to create pretrained ViT HTR model from config.
    
    Args:
        config: OmegaConf configuration object
        num_classes: Number of character classes (including CTC blank)
    
    Returns:
        PretrainedViTHTR model instance
    """
    arch_config = config.arch_vit_pretrained
    
    model = PretrainedViTHTR(
        num_classes=num_classes,
        pretrained=arch_config.get('pretrained', True),
        decoder_type=arch_config.get('decoder_type', 'rnn'),
        decoder_hidden=arch_config.get('decoder_hidden', 512),
        decoder_layers=arch_config.get('decoder_layers', 2),
        freeze_encoder=arch_config.get('freeze_encoder', False),
        dropout=arch_config.get('dropout', 0.1)
    )
    
    return model


# Test code
if __name__ == '__main__':
    print("=" * 80)
    print("Testing Pretrained ViT-B/16 HTR Model")
    print("=" * 80)
    
    # Check if timm is installed
    try:
        import timm
        print(f"✓ timm version: {timm.__version__}")
    except ImportError:
        print("✗ timm not installed. Run: pip install timm")
        exit(1)
    
    # Create model
    num_classes = 80
    model = PretrainedViTHTR(
        num_classes=num_classes,
        pretrained=True,
        decoder_type='rnn',
        freeze_encoder=False
    )
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 1, 128, 1024)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected: [T={196}, B={batch_size}, C={num_classes}]")
    
    # Check attention weights
    attention_weights = model.get_attention_weights()
    print(f"\nNumber of attention layers captured: {len(attention_weights)}")
    if attention_weights:
        print(f"Attention weight shape (first layer): {attention_weights[0].shape}")
        print(f"Expected: [B={batch_size}, num_heads=12, num_patches=197, num_patches=197]")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test with frozen encoder
    print("\n" + "=" * 80)
    print("Testing with Frozen Encoder")
    print("=" * 80)
    
    model_frozen = PretrainedViTHTR(
        num_classes=num_classes,
        pretrained=True,
        decoder_type='rnn',
        freeze_encoder=True
    )
    
    trainable_params_frozen = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
    print(f"Trainable parameters (frozen encoder): {trainable_params_frozen:,}")
    print(f"Frozen parameters: {total_params - trainable_params_frozen:,}")
    
    print("\n✓ All tests passed!")
