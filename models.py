import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
from einops import rearrange, repeat
from typing import Optional, Tuple, List

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CNN(nn.Module):
    def __init__(self, cnn_cfg, flattening='maxpool'):
        super(CNN, self).__init__()

        self.k = 1
        self.flattening = flattening

        self.features = nn.ModuleList([nn.Conv2d(1, 32, 7, [4, 2], 3), nn.ReLU()])
        in_channels = 32
        cntm = 0
        cnt = 1

        for m in cnn_cfg:
            if m == 'M':
                self.features.add_module('mxp' + str(cntm), nn.MaxPool2d(kernel_size=2, stride=2))
                cntm += 1
            else:
                for i in range(int(m[0])):
                    x = int(m[1])
                    self.features.add_module('cnv' + str(cnt), BasicBlock(in_channels, x,))
                    in_channels = x
                    cnt += 1

    def forward(self, x, reduce='max'):

        y = x
        for i, nn_module in enumerate(self.features):
            y = nn_module(y)

        if self.flattening=='maxpool':
            y = F.max_pool2d(y, [y.size(2), self.k], stride=[y.size(2), 1], padding=[0, self.k//2])
        elif self.flattening=='concat':
            y = y.view(y.size(0), -1, 1, y.size(3))

        return y

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)


class CTCtopC(nn.Module):
    def __init__(self, input_size, nclasses, dropout=0.0):
        super(CTCtopC, self).__init__()

        # Add LayerNorm for better training stability
        self.norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)
        self.cnn_top = nn.Conv2d(input_size, nclasses, kernel_size=(1, 3), stride=1, padding=(0, 1))
        
        # Initialize output layer with smaller weights for stable CTC training
        nn.init.xavier_uniform_(self.cnn_top.weight, gain=0.1)
        if self.cnn_top.bias is not None:
            nn.init.zeros_(self.cnn_top.bias)

    def forward(self, x):
        # x: [B, C, H, W] where H=1
        # Apply LayerNorm on feature dimension
        B, C, H, W = x.shape
        x_norm = x.squeeze(2).permute(2, 0, 1)  # [W, B, C]
        x_norm = self.norm(x_norm)  # LayerNorm on last dim
        x = x_norm.permute(1, 2, 0).unsqueeze(2)  # Back to [B, C, 1, W]
        
        x = self.dropout(x)
        y = self.cnn_top(x)  # [B, nclasses, H, W]
        y = y.squeeze(2).permute(2, 0, 1)  # [T, B, nclasses]
        return y


class CTCtopLinear(nn.Module):
    """
    Linear CTC head for Mamba or other sequence models.
    Takes [T, B, D] or [B, D, 1, T] and outputs [T, B, nclasses]
    """
    def __init__(self, input_size, nclasses, dropout=0.0):
        super(CTCtopLinear, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_size, nclasses)
    
    def forward(self, x):
        # Handle both formats
        if x.dim() == 4:  # [B, C, H, W] where H=1
            x = x.squeeze(2).permute(2, 0, 1)  # [T, B, C]
        # x is now [T, B, D]
        
        x = self.dropout(x)
        y = self.linear(x)  # [T, B, nclasses]
        return y


class CTCtopR(nn.Module):
    def __init__(self, input_size, rnn_cfg, nclasses, rnn_type='gru'):
        super(CTCtopR, self).__init__()

        hidden, num_layers = rnn_cfg

        # Add LayerNorm before RNN for better gradient flow
        self.norm = nn.LayerNorm(input_size)
        
        if rnn_type == 'gru':
            self.rec = nn.GRU(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=.2)
        elif rnn_type == 'lstm':
            self.rec = nn.LSTM(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=.2)
        else:
            print('problem! - no such rnn type is defined')
            exit()
        
        # Add residual connection support
        self.use_residual = (input_size == 2 * hidden)
        if self.use_residual:
            self.residual_proj = nn.Identity()
        
        self.fnl = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Dropout(.3), 
            nn.Linear(2 * hidden, nclasses)
        )
        
        # Better initialization for output layer
        nn.init.xavier_uniform_(self.fnl[2].weight, gain=0.1)
        if self.fnl[2].bias is not None:
            nn.init.zeros_(self.fnl[2].bias)

    def forward(self, x):
        # x: [B, C, H, W] where H should be 1 for sequence data
        # Need to reshape to [T, B, C] for RNN
        
        # Remove singleton spatial dimension and transpose
        # [B, C, 1, W] -> [B, C, W] -> [W, B, C]
        y = x.squeeze(2).permute(2, 0, 1)  # [T, B, C]
        
        # Apply LayerNorm
        y = self.norm(y)  # [T, B, C]
        
        y = self.rec(y)[0]  # [T, B, 2*hidden]
        y = self.fnl(y)     # [T, B, nclasses]

        return y

class CTCtopB(nn.Module):
    def __init__(self, input_size, rnn_cfg, nclasses, rnn_type='gru'):
        super(CTCtopB, self).__init__()

        hidden, num_layers = rnn_cfg

        if rnn_type == 'gru':
            self.rec = nn.GRU(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=.2)
        elif rnn_type == 'lstm':
            self.rec = nn.LSTM(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=.2)
        else:
            print('problem! - no such rnn type is defined')
            exit()
        
        self.fnl = nn.Sequential(nn.Dropout(.5), nn.Linear(2 * hidden, nclasses))

        self.cnn = nn.Sequential(nn.Dropout(.5), 
                                 nn.Conv2d(input_size, nclasses, kernel_size=(1, 3), stride=1, padding=(0, 1))
        )

    def forward(self, x):
        # RNN path: [B, C, H, W] -> [T, B, C] -> [T, B, nclasses]
        y = x.squeeze(2).permute(2, 0, 1)  # [T, B, C]
        y = self.rec(y)[0]  # [T, B, 2*hidden]
        y = self.fnl(y)     # [T, B, nclasses]

        # CNN shortcut path: [B, C, H, W] -> [T, B, nclasses]
        cnn_out = self.cnn(x).squeeze(2).permute(2, 0, 1)  # [T, B, nclasses]

        if self.training:
            return y, cnn_out
        else:
            return y, cnn_out


class ViTRGTSBackbone(nn.Module):
    """
    ViT-style backbone with register tokens, inspired by:
    - 'Vision Transformers Need Registers'
    - kyegomez/Vit-RGTS

    Input:  x : [B, 1, H, W]  (grayscale line image)
    Output: seq_tokens : [T, B, D]  (patch tokens for CTC)
            reg_tokens : [B, R, D]  (register tokens for analysis later)
            grid_size  : (Hp, Wp)   (patch grid, for spatial mapping of tokens)
    """

    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 16,
        patch_height: int = None,
        patch_width: int = None,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_registers: int = 4,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()

        # Support both square and rectangular patches
        if patch_height is None and patch_width is None:
            self.patch_height = patch_size
            self.patch_width = patch_size
        else:
            self.patch_height = patch_height if patch_height is not None else patch_size
            self.patch_width = patch_width if patch_width is not None else patch_size
        
        self.embed_dim = embed_dim
        self.num_registers = num_registers
        self.max_seq_len = max_seq_len

        # Patch embedding: from [B, 1, H, W] -> [B, D, Hp, Wp]
        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=(self.patch_height, self.patch_width),
            stride=(self.patch_height, self.patch_width),
            padding=0,
        )

        # Register tokens (learned, store global info)
        self.register_tokens = nn.Parameter(
            torch.zeros(1, num_registers, embed_dim)
        )


        # Learned positional embeddings for (registers + patches)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_seq_len, embed_dim)
        )

        self.emb_dropout = nn.Dropout(emb_dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # [B, S, D]
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.register_tokens, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        # patch_embed uses Kaiming by default; you can tweak if you want stronger init


    def forward(self, x):
        """
        x: [B, 1, H, W]
        returns:
            seq_tokens: [T, B, D]   (for CTC head)
            reg_tokens: [B, R, D]   (for analysis)
            grid_size : (Hp, Wp)
        """
        B, C, H, W = x.shape
        assert C == 1, f"Expected grayscale input (C=1), got C={C}"

        # Patchify
        x = self.patch_embed(x)               # [B, D, Hp, Wp]
        B, D, Hp, Wp = x.shape
        num_patches = Hp * Wp

        patch_tokens = x.flatten(2).transpose(1, 2)  # [B, Np, D]


        # Prepare register tokens
        reg_tokens = self.register_tokens.expand(B, -1, -1)  # [B, R, D]

        # Concatenate registers + patches -> [B, S, D]
        tokens = torch.cat([reg_tokens, patch_tokens], dim=1)  # S = R + Np

        S = tokens.size(1)
        if S > self.max_seq_len:
            raise ValueError(
                f"Sequence length {S} exceeds max_seq_len={self.max_seq_len}. "
                "Increase vit_max_seq_len in your config."
            )

        # Add positional embeddings and dropout
        # CRITICAL: When num_registers=0, use pos_embed starting from index 0
        # When num_registers>0, pos_embed[:R] are for registers, pos_embed[R:] for patches
        pos = self.pos_embed[:, :S, :]  # [1, S, D]
        tokens = tokens + pos
        tokens = self.emb_dropout(tokens)

        # Transformer encoder
        encoded = self.encoder(tokens)         # [B, S, D]

        # Split back into registers and patch tokens

        reg_out = encoded[:, :self.num_registers, :]      # [B, R, D]
        patch_out = encoded[:, self.num_registers:, :]    # [B, Np, D]

        # For CTC, we want [T, B, D] (time-major)
        seq_tokens = patch_out.transpose(0, 1)            # [Np, B, D]

        return seq_tokens, reg_out, (Hp, Wp)
    
    @torch.no_grad()
    def forward_explain(self, x):
        """
        Explainability forward pass.
        Returns:
            seq_tokens:   [T, B, D]           - patch token sequence (CTC input)
            reg_tokens:   [B, R, D]           - register embeddings
            attn_maps:    List[L] of [B, H, S, S]  - per-layer, per-head attention
            token_norms:  [B, S]              - L2 norm per token at final layer
            grid_size:    (Hp, Wp)
        """
        B, C, H, W = x.shape

        # === PATCH EMBEDDING ===================================================
        x = self.patch_embed(x)               # [B, D, Hp, Wp]
        B, D, Hp, Wp = x.shape
        patch_tokens = x.flatten(2).transpose(1, 2)  # [B, Np, D]
        num_patches = Hp * Wp

        reg_tokens = self.register_tokens.expand(B, -1, -1)  # [B, R, D]

        tokens = torch.cat([reg_tokens, patch_tokens], dim=1)   # [B, S, D]
        
        S = tokens.size(1)

        pos = self.pos_embed[:, :S, :]
        tokens = tokens + pos
        tokens = self.emb_dropout(tokens)

        # === INITIALIZE EXPLAINABILITY BUFFERS =================================
        attn_maps = []   # list of per-layer attention
        token_norms = None

        # === MANUAL TRANSFORMER ENCODER STACK ==================================
        # We need to extract attention weights from each layer
        # PyTorch's MultiheadAttention doesn't expose q,k,v projections directly
        # So we'll use a hook-based approach or compute manually
        
        x_tokens = tokens
        for layer_idx, layer in enumerate(self.encoder.layers):
            # Get the attention module
            attn_module = layer.self_attn
            
            # Manually compute attention for explainability
            # Extract weights from the attention module
            embed_dim = attn_module.embed_dim
            num_heads = attn_module.num_heads
            head_dim = embed_dim // num_heads
            
            # Apply input projection (in_proj contains Q, K, V weights)
            if attn_module._qkv_same_embed_dim:
                # Single weight matrix for Q, K, V
                q, k, v = torch.nn.functional.linear(
                    x_tokens, attn_module.in_proj_weight, attn_module.in_proj_bias
                ).chunk(3, dim=-1)
            else:
                # Separate Q, K, V projections
                q = torch.nn.functional.linear(x_tokens, attn_module.q_proj_weight, attn_module.in_proj_bias[:embed_dim])
                k = torch.nn.functional.linear(x_tokens, attn_module.k_proj_weight, attn_module.in_proj_bias[embed_dim:2*embed_dim])
                v = torch.nn.functional.linear(x_tokens, attn_module.v_proj_weight, attn_module.in_proj_bias[2*embed_dim:])
            
            B_, S_, E_ = q.shape
            
            # Reshape for multi-head attention: [B, S, E] -> [B, H, S, D_h]
            q = q.view(B_, S_, num_heads, head_dim).transpose(1, 2)
            k = k.view(B_, S_, num_heads, head_dim).transpose(1, 2)
            v = v.view(B_, S_, num_heads, head_dim).transpose(1, 2)
            
            # Compute attention scores
            attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
            attn_weights = torch.softmax(attn_weights, dim=-1)  # [B, H, S, S]
            
            attn_maps.append(attn_weights.cpu())
            
            # Apply attention to values
            attn_output = torch.matmul(attn_weights, v)  # [B, H, S, D_h]
            attn_output = attn_output.transpose(1, 2).contiguous().view(B_, S_, E_)
            
            # Apply output projection
            attn_output = torch.nn.functional.linear(
                attn_output, attn_module.out_proj.weight, attn_module.out_proj.bias
            )
            
            # Apply residual connection and layer norm (norm-first)
            x_tokens = layer.norm1(x_tokens)
            x_tokens = x_tokens + attn_output
            
            # Feed-forward network
            x_tokens = layer.norm2(x_tokens)
            ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x_tokens))))
            x_tokens = x_tokens + ff_output


        reg_out = x_tokens[:, :self.num_registers, :]            # [B, R, D]
        patch_out = x_tokens[:, self.num_registers:, :]          # [B, Np, D]

        # === TOKEN NORMS (FINAL LAYER) ==========================================
        token_norms = x_tokens.norm(dim=-1).cpu()                # [B, S]

        # === FINAL SHAPE FOR CTC ===============================================
        seq_tokens = patch_out.transpose(0, 1)                   # [T, B, D]

        return seq_tokens, reg_out, attn_maps, token_norms, (Hp, Wp)


class TorchVisionViTBackbone(nn.Module):
    """
    Wrapper for TorchVision's pretrained ViT models (e.g., vit_b_16).
    
    Adapts the standard ImageNet-pretrained ViT for HTR by:
    - Converting grayscale to RGB (channel expansion)
    - Extracting patch tokens (excluding CLS token)
    - Optionally adding register tokens for better attention maps
    
    Input:  x : [B, 1, H, W]  (grayscale line image)
    Output: seq_tokens : [T, B, D]  (patch tokens for CTC)
            cls_token  : [B, D]     (CLS token for analysis)
            grid_size  : (Hp, Wp)   (patch grid)
    """
    
    def __init__(
        self,
        model_name: str = "vit_b_16",
        pretrained: bool = True,
        num_registers: int = 0,  # Optional register tokens
        freeze_backbone: bool = False,
        image_height: int = 128,
        image_width: int = 1024,
    ):
        super().__init__()
        
        try:
            from torchvision.models import get_model
            from torchvision.models.vision_transformer import VisionTransformer
        except ImportError:
            raise ImportError(
                "TorchVision not found. Install with: pip install torchvision"
            )
        
        # Set up local cache directory for pretrained models
        import os
        cache_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            'pretrained_models', 
            'torchvision'
        ))
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load pretrained model with local caching
        if pretrained:
            # Check if model exists locally
            local_model_path = os.path.join(cache_dir, f"{model_name}.pth")
            
            if os.path.exists(local_model_path):
                print(f"Loading {model_name} from local cache: {local_model_path}")
                self.vit = get_model(model_name, weights=None)
                state_dict = torch.load(local_model_path, map_location='cpu')
                self.vit.load_state_dict(state_dict)
            else:
                print(f"Downloading {model_name} and caching to: {local_model_path}")
                self.vit = get_model(model_name, weights="DEFAULT")
                # Save to local cache
                torch.save(self.vit.state_dict(), local_model_path)
                print(f"Model cached successfully")
        else:
            self.vit = get_model(model_name, weights=None)
        
        if not isinstance(self.vit, VisionTransformer):
            raise ValueError(f"{model_name} is not a Vision Transformer model")
        
        # Get model dimensions
        self.embed_dim = self.vit.hidden_dim
        self.patch_size = self.vit.patch_size
        self.num_registers = num_registers
        self.image_height = image_height
        self.image_width = image_width
        
        # Calculate grid size
        self.grid_h = image_height // self.patch_size
        self.grid_w = image_width // self.patch_size
        
        # Grayscale to RGB conversion (ViT expects 3 channels)
        self.gray_to_rgb = nn.Conv2d(1, 3, 1, bias=False)
        # Initialize to replicate grayscale across channels
        with torch.no_grad():
            self.gray_to_rgb.weight.fill_(1.0)
        
        # Optional: Register tokens (inspired by ViT-RGTS)
        if num_registers > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, num_registers, self.embed_dim)
            )
            nn.init.normal_(self.register_tokens, std=0.02)
        
        # Optionally freeze the backbone
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        x: [B, 1, H, W]
        returns:
            seq_tokens: [T, B, D]   (for CTC head)
            cls_token:  [B, D]      (for analysis)
            grid_size:  (Hp, Wp)
        """
        B, C, H, W = x.shape
        assert C == 1, f"Expected grayscale input (C=1), got C={C}"
        
        # Convert grayscale to RGB
        x_rgb = self.gray_to_rgb(x)  # [B, 3, H, W]
        
        # Resize if needed
        if H != self.image_height or W != self.image_width:
            x_rgb = F.interpolate(
                x_rgb, size=(self.image_height, self.image_width),
                mode='bilinear', align_corners=False
            )
        
        # Extract features from ViT encoder
        # Forward through patch embedding
        x_patch = self.vit.conv_proj(x_rgb)  # [B, D, Hp, Wp]
        B, D, Hp, Wp = x_patch.shape
        
        # Flatten patches
        patch_tokens = x_patch.flatten(2).transpose(1, 2)  # [B, Np, D]
        
        # Add class token
        cls_tokens = self.vit.class_token.expand(B, -1, -1)  # [B, 1, D]
        
        # Optionally add register tokens
        if self.num_registers > 0:
            reg_tokens = self.register_tokens.expand(B, -1, -1)  # [B, R, D]
            tokens = torch.cat([cls_tokens, reg_tokens, patch_tokens], dim=1)
        else:
            tokens = torch.cat([cls_tokens, patch_tokens], dim=1)
        
        # Handle positional encoding - resize if needed
        S = tokens.size(1)
        pos_embed_size = self.vit.encoder.pos_embedding.size(1)
        
        if S != pos_embed_size:
            # Need to interpolate positional embeddings
            # pos_embedding shape: [1, N_original, D]
            pos_embed = self.vit.encoder.pos_embedding
            
            # Extract CLS token positional embedding
            cls_pos = pos_embed[:, 0:1, :]  # [1, 1, D]
            
            # Get patch positional embeddings (skip CLS)
            patch_pos = pos_embed[:, 1:, :]  # [1, N_original-1, D]
            
            # Calculate original grid size from pretrained ViT
            N_original_patches = pos_embed_size - 1
            orig_size = int(N_original_patches ** 0.5)
            
            # Reshape to 2D grid for interpolation
            patch_pos = patch_pos.reshape(1, orig_size, orig_size, self.embed_dim)
            patch_pos = patch_pos.permute(0, 3, 1, 2)  # [1, D, H_orig, W_orig]
            
            # Interpolate to new grid size
            patch_pos = F.interpolate(
                patch_pos, 
                size=(Hp, Wp), 
                mode='bicubic', 
                align_corners=False
            )
            
            # Reshape back to sequence
            patch_pos = patch_pos.permute(0, 2, 3, 1)  # [1, Hp, Wp, D]
            patch_pos = patch_pos.reshape(1, Hp * Wp, self.embed_dim)  # [1, Np, D]
            
            # Reconstruct positional embedding with registers if needed
            if self.num_registers > 0:
                # Create zero positional embeddings for register tokens
                reg_pos = torch.zeros(1, self.num_registers, self.embed_dim, device=patch_pos.device)
                pos_embedding = torch.cat([cls_pos, reg_pos, patch_pos], dim=1)
            else:
                pos_embedding = torch.cat([cls_pos, patch_pos], dim=1)
        else:
            pos_embedding = self.vit.encoder.pos_embedding
        
        tokens = tokens + pos_embedding
        tokens = self.vit.encoder.dropout(tokens)
        
        # Forward through transformer encoder
        encoded = self.vit.encoder.layers(tokens)  # [B, S, D]
        encoded = self.vit.encoder.ln(encoded)
        
        # Split tokens
        cls_out = encoded[:, 0, :]  # [B, D]
        
        if self.num_registers > 0:
            # Skip CLS + registers to get patch tokens
            patch_out = encoded[:, 1 + self.num_registers:, :]  # [B, Np, D]
        else:
            patch_out = encoded[:, 1:, :]  # [B, Np, D]
        
        # Convert to [T, B, D] for CTC
        seq_tokens = patch_out.transpose(0, 1)  # [Np, B, D]
        
        return seq_tokens, cls_out, (Hp, Wp)
    
    @torch.no_grad()
    def forward_explain(self, x):
        """
        Extended forward pass with attention maps for explainability.
        """
        B, C, H, W = x.shape
        
        # Convert and resize
        x_rgb = self.gray_to_rgb(x)
        if H != self.image_height or W != self.image_width:
            x_rgb = F.interpolate(
                x_rgb, size=(self.image_height, self.image_width),
                mode='bilinear', align_corners=False
            )
        
        # Patch embedding
        x_patch = self.vit.conv_proj(x_rgb)
        B, D, Hp, Wp = x_patch.shape
        patch_tokens = x_patch.flatten(2).transpose(1, 2)
        
        # Prepare tokens
        cls_tokens = self.vit.class_token.expand(B, -1, -1)
        if self.num_registers > 0:
            reg_tokens = self.register_tokens.expand(B, -1, -1)
            tokens = torch.cat([cls_tokens, reg_tokens, patch_tokens], dim=1)
        else:
            tokens = torch.cat([cls_tokens, patch_tokens], dim=1)
        
        # Handle positional encoding - resize if needed (same as forward method)
        S = tokens.size(1)
        pos_embed_size = self.vit.encoder.pos_embedding.size(1)
        
        if S != pos_embed_size:
            pos_embed = self.vit.encoder.pos_embedding
            cls_pos = pos_embed[:, 0:1, :]
            patch_pos = pos_embed[:, 1:, :]
            
            N_original_patches = pos_embed_size - 1
            orig_size = int(N_original_patches ** 0.5)
            
            patch_pos = patch_pos.reshape(1, orig_size, orig_size, self.embed_dim)
            patch_pos = patch_pos.permute(0, 3, 1, 2)
            
            patch_pos = F.interpolate(
                patch_pos, size=(Hp, Wp), mode='bicubic', align_corners=False
            )
            
            patch_pos = patch_pos.permute(0, 2, 3, 1)
            patch_pos = patch_pos.reshape(1, Hp * Wp, self.embed_dim)
            
            if self.num_registers > 0:
                reg_pos = torch.zeros(1, self.num_registers, self.embed_dim, device=patch_pos.device)
                pos_embedding = torch.cat([cls_pos, reg_pos, patch_pos], dim=1)
            else:
                pos_embedding = torch.cat([cls_pos, patch_pos], dim=1)
        else:
            pos_embedding = self.vit.encoder.pos_embedding
        
        tokens = tokens + pos_embedding
        tokens = self.vit.encoder.dropout(tokens)
        
        # Collect attention maps from each layer
        attn_maps = []
        x_tokens = tokens
        
        for layer in self.vit.encoder.layers.layers:
            # Extract attention weights manually
            attn_output, attn_weights = layer.self_attention(
                layer.ln_1(x_tokens), 
                need_weights=True, 
                average_attn_weights=False
            )
            attn_maps.append(attn_weights.cpu())
            
            # Complete the transformer block
            x_tokens = x_tokens + attn_output
            x_tokens = x_tokens + layer.mlp(layer.ln_2(x_tokens))
        
        encoded = self.vit.encoder.ln(x_tokens)
        

        # Split tokens
        cls_out = encoded[:, 0, :]
        if self.num_registers > 0:
            reg_out = encoded[:, 1:1+self.num_registers, :]
            patch_out = encoded[:, 1+self.num_registers:, :]
        else:
            reg_out = None
            patch_out = encoded[:, 1:, :]
        
        seq_tokens = patch_out.transpose(0, 1)
        token_norms = encoded.norm(dim=-1).cpu()
        
        return seq_tokens, cls_out, reg_out, attn_maps, token_norms, (Hp, Wp)


class TrOCREncoderBackbone(nn.Module):
    """
    Wrapper for TrOCR encoder from HuggingFace transformers.
    
    TrOCR uses a ViT encoder + Transformer decoder architecture.
    For HTR with CTC, we only use the encoder part.
    
    Input:  x : [B, 1, H, W]  (grayscale line image)
    Output: seq_tokens : [T, B, D]  (patch tokens for CTC)
            grid_size  : (Hp, Wp)   (patch grid)
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-handwritten",
        freeze_encoder: bool = False,
        image_height: int = 384,
        image_width: int = 384,
    ):
        super().__init__()
        
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError:
            raise ImportError(
                "Transformers not found. Install with: pip install transformers"
            )
        
        # Set up local cache directory for pretrained models
        import os
        cache_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            'asset',
            'pretrained_models', 
            'trocr'
        ))
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load pretrained TrOCR model with local caching
        print(f"Loading TrOCR model: {model_name}")
        print(f"Cache directory: {cache_dir}")
        
        # HuggingFace transformers automatically caches, but we set explicit cache dir
        self.model = VisionEncoderDecoderModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        self.processor = TrOCRProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        print(f"TrOCR model loaded successfully")
        
        # Extract encoder only
        self.encoder = self.model.encoder
        self.embed_dim = self.encoder.config.hidden_size
        
        # Image preprocessing parameters
        self.image_height = image_height
        self.image_width = image_width
        
        # Grayscale to RGB conversion
        self.gray_to_rgb = nn.Conv2d(1, 3, 1, bias=False)
        with torch.no_grad():
            self.gray_to_rgb.weight.fill_(1.0)
        
        # Calculate patch grid (TrOCR uses 16x16 patches)
        self.patch_size = 16
        self.grid_h = image_height // self.patch_size
        self.grid_w = image_width // self.patch_size
        
        # Optionally freeze encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        x: [B, 1, H, W]
        returns:
            seq_tokens: [T, B, D]   (for CTC head)
            grid_size:  (Hp, Wp)
        """
        B, C, H, W = x.shape
        assert C == 1, f"Expected grayscale input (C=1), got C={C}"
        
        # Convert to RGB
        x_rgb = self.gray_to_rgb(x)  # [B, 3, H, W]
        
        # Resize to TrOCR's expected input size
        if H != self.image_height or W != self.image_width:
            x_rgb = F.interpolate(
                x_rgb, size=(self.image_height, self.image_width),
                mode='bilinear', align_corners=False
            )
        
        # Normalize (TrOCR expects normalized inputs)
        mean = torch.tensor([0.5, 0.5, 0.5], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=x.device).view(1, 3, 1, 1)
        x_rgb = (x_rgb - mean) / std
        
        # Forward through encoder
        encoder_outputs = self.encoder(pixel_values=x_rgb)
        hidden_states = encoder_outputs.last_hidden_state  # [B, S, D]
        
        # TrOCR encoder includes CLS token, remove it
        # Shape: [B, 1 + Hp*Wp, D] -> [B, Hp*Wp, D]
        seq_tokens = hidden_states[:, 1:, :]  # Skip CLS token
        
        # Convert to [T, B, D] for CTC
        seq_tokens = seq_tokens.transpose(0, 1)  # [T, B, D]
        
        return seq_tokens, (self.grid_h, self.grid_w)
    
    @torch.no_grad()
    def forward_explain(self, x):
        """
        Extended forward pass with attention from all layers.
        """
        B, C, H, W = x.shape
        
        # Preprocess
        x_rgb = self.gray_to_rgb(x)
        if H != self.image_height or W != self.image_width:
            x_rgb = F.interpolate(
                x_rgb, size=(self.image_height, self.image_width),
                mode='bilinear', align_corners=False
            )
        
        mean = torch.tensor([0.5, 0.5, 0.5], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=x.device).view(1, 3, 1, 1)
        x_rgb = (x_rgb - mean) / std
        
        # Forward with attention output
        encoder_outputs = self.encoder(
            pixel_values=x_rgb,
            output_attentions=True,
            return_dict=True
        )
        
        hidden_states = encoder_outputs.last_hidden_state
        attn_maps = [attn.cpu() for attn in encoder_outputs.attentions]
        
        # Extract tokens
        cls_token = hidden_states[:, 0, :]  # [B, D]
        seq_tokens = hidden_states[:, 1:, :].transpose(0, 1)  # [T, B, D]
        
        # Token norms
        token_norms = hidden_states.norm(dim=-1).cpu()  # [B, S]
        
        return seq_tokens, cls_token, attn_maps, token_norms, (self.grid_h, self.grid_w)


class HTRNet(nn.Module):
    """
    Unified entry point for:
      - CNN+RNN (original Best Practices model):   arch_cfg.type == 'cnn_rnn'
      - ViT+Registers backbone (this work):        arch_cfg.type == 'vit_rgts'
      - TorchVision ViT backbone:                  arch_cfg.type == 'torchvision_vit'
      - TrOCR encoder backbone:                    arch_cfg.type == 'trocr'
      - CNN+Mamba (state space model):             arch_cfg.type == 'cnn_mamba'

    Forward always returns:
        - logits: [T, B, nclasses]   (CTC-ready)
    """

    def __init__(self, arch_cfg, nclasses):
        super(HTRNet, self).__init__()

        self.arch_type = getattr(arch_cfg, "type", "cnn_rnn")

        # (Optional) STN hook – still not implemented
        if getattr(arch_cfg, "stn", False):
            raise NotImplementedError(
                "Spatial Transformer Networks not implemented - "
                "you can plug your own STN here if needed."
            )
        self.stn = None

        # ------------------------------------------------------------------
        # 1) CNN + RNN path  (original Best Practices HTRNet)
        # ------------------------------------------------------------------
        if self.arch_type == "cnn_rnn":
            cnn_cfg = arch_cfg.cnn_cfg
            self.features = CNN(cnn_cfg, flattening=arch_cfg.flattening)

            # Hidden size after CNN flattening
            if arch_cfg.flattening in ["maxpool", "avgpool"]:
                hidden = cnn_cfg[-1][-1]          # last channel
            elif arch_cfg.flattening == "concat":
                # In original code, k is the pooling height in the last layer
                # If you changed that, expose it in arch_cfg and use it here.
                k = getattr(arch_cfg, "k", 4)
                hidden = cnn_cfg[-1][-1] * k
            else:
                raise ValueError(
                    f"Unknown flattening mode: {arch_cfg.flattening}"
                )

            head = arch_cfg.head_type
            if head == "cnn":
                self.top = CTCtopC(hidden, nclasses)
            elif head == "rnn":
                self.top = CTCtopR(
                    hidden,
                    (arch_cfg.rnn_hidden_size, arch_cfg.rnn_layers),
                    nclasses,
                    rnn_type=arch_cfg.rnn_type,
                )
            elif head == "both":
                self.top = CTCtopB(
                    hidden,
                    (arch_cfg.rnn_hidden_size, arch_cfg.rnn_layers),
                    nclasses,
                    rnn_type=arch_cfg.rnn_type,
                )
            else:
                raise ValueError(f"Unknown head_type: {head}")

               # ------------------------------------------------------------------
        # 2) ViT + Registers path  (Vision Transformers Need Registers)
        # ------------------------------------------------------------------
        elif self.arch_type == "vit_rgts":
            # Read ViT hyper-parameters from baseline_vit_rgts.yaml
            # and provide sensible fallbacks.

            image_height = getattr(arch_cfg, "image_height", 128)
            image_width  = getattr(arch_cfg, "image_width", 1024)

            # Allow either (patch_height/patch_width) or a single patch_size
            patch_height = getattr(arch_cfg, "patch_height", getattr(arch_cfg, "patch_size", 16))
            patch_width  = getattr(arch_cfg, "patch_width", patch_height)

            dim    = getattr(arch_cfg, "dim", getattr(arch_cfg, "vit_embed_dim", 256))
            depth  = getattr(arch_cfg, "depth", getattr(arch_cfg, "vit_depth", 6))
            heads  = getattr(arch_cfg, "heads", getattr(arch_cfg, "vit_num_heads", 8))
            mlp_dim = getattr(arch_cfg, "mlp_dim", int(dim * 4))

            # Convert mlp_dim into a ratio if needed
            mlp_ratio = getattr(arch_cfg, "vit_mlp_ratio", float(mlp_dim) / float(dim))

            num_registers = getattr(arch_cfg, "num_registers", 4)

            dropout = getattr(arch_cfg, "dropout", getattr(arch_cfg, "vit_dropout", 0.1))
            emb_dropout = getattr(arch_cfg, "emb_dropout", getattr(arch_cfg, "vit_emb_dropout", 0.1))

            # Compute a safe max_seq_len if not specified:
            #   R registers + Hp*Wp patches + small margin
            Hp = image_height // patch_height
            Wp = image_width  // patch_width
            seq_len_est = num_registers + Hp * Wp + 8

            max_seq_len = getattr(arch_cfg, "vit_max_seq_len", seq_len_est)

            self.backbone = ViTRGTSBackbone(
                image_size=image_height,
                patch_height=patch_height,        # Pass asymmetric patch sizes
                patch_width=patch_width,
                embed_dim=dim,
                depth=depth,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                num_registers=num_registers,
                dropout=dropout,
                emb_dropout=emb_dropout,
                max_seq_len=max_seq_len,
            )

            hidden = self.backbone.embed_dim

            # You can still choose to stack an RNN on top of the ViT tokens,
            # or go pure-transformer with a linear CTC head.
            head = getattr(arch_cfg, "head_type", "rnn")

            if head == "cnn":
                # 'cnn' in this context = linear over ViT sequence (no RNN)
                self.top = CTCtopC(hidden, nclasses)
            elif head == "both":
                self.top = CTCtopB(
                    hidden,
                    (arch_cfg.rnn_hidden_size, arch_cfg.rnn_layers),
                    nclasses,
                    rnn_type=arch_cfg.rnn_type,
                )
            else:
                # default: ViT -> BiRNN -> CTC
                self.top = CTCtopR(
                    hidden,
                    (arch_cfg.rnn_hidden_size, arch_cfg.rnn_layers),
                    nclasses,
                    rnn_type=arch_cfg.rnn_type,
                )
        
        # ------------------------------------------------------------------
        # 3) TorchVision ViT path  (pretrained ViT from torchvision)
        # ------------------------------------------------------------------
        elif self.arch_type == "torchvision_vit":
            image_height = getattr(arch_cfg, "image_height", 128)
            image_width = getattr(arch_cfg, "image_width", 1024)
            model_name = getattr(arch_cfg, "model_name", "vit_b_16")
            pretrained = getattr(arch_cfg, "pretrained", True)
            num_registers = getattr(arch_cfg, "num_registers", 0)
            freeze_backbone = getattr(arch_cfg, "freeze_backbone", False)
            
            self.backbone = TorchVisionViTBackbone(
                model_name=model_name,
                pretrained=pretrained,
                num_registers=num_registers,
                freeze_backbone=freeze_backbone,
                image_height=image_height,
                image_width=image_width,
            )
            
            hidden = self.backbone.embed_dim
            head = getattr(arch_cfg, "head_type", "rnn")
            
            if head == "cnn":
                self.top = CTCtopC(hidden, nclasses)
            elif head == "both":
                self.top = CTCtopB(
                    hidden,
                    (arch_cfg.rnn_hidden_size, arch_cfg.rnn_layers),
                    nclasses,
                    rnn_type=arch_cfg.rnn_type,
                )
            else:
                self.top = CTCtopR(
                    hidden,
                    (arch_cfg.rnn_hidden_size, arch_cfg.rnn_layers),
                    nclasses,
                    rnn_type=arch_cfg.rnn_type,
                )
        
        # ------------------------------------------------------------------
        # 4) TrOCR encoder path  (TrOCR from HuggingFace)
        # ------------------------------------------------------------------
        elif self.arch_type == "trocr":
            model_name = getattr(arch_cfg, "model_name", "microsoft/trocr-base-handwritten")
            freeze_encoder = getattr(arch_cfg, "freeze_encoder", False)
            image_height = getattr(arch_cfg, "image_height", 384)
            image_width = getattr(arch_cfg, "image_width", 384)
            
            self.backbone = TrOCREncoderBackbone(
                model_name=model_name,
                freeze_encoder=freeze_encoder,
                image_height=image_height,
                image_width=image_width,
            )
            
            hidden = self.backbone.embed_dim
            head = getattr(arch_cfg, "head_type", "rnn")
            
            if head == "cnn":
                self.top = CTCtopC(hidden, nclasses)
            elif head == "both":
                self.top = CTCtopB(
                    hidden,
                    (arch_cfg.rnn_hidden_size, arch_cfg.rnn_layers),
                    nclasses,
                    rnn_type=arch_cfg.rnn_type,
                )
            else:
                self.top = CTCtopR(
                    hidden,
                    (arch_cfg.rnn_hidden_size, arch_cfg.rnn_layers),
                    nclasses,
                    rnn_type=arch_cfg.rnn_type,
                )
        else:
            raise ValueError(f"Unknown architecture type: {self.arch_type}")

    def forward(self, x):
        """
        x: [B, 1, H, W]
        returns:
            logits_ctc: [T, B, nclasses]
        """

        # Optional spatial transformer
        if self.stn is not None:
            x = self.stn(x)

        if self.arch_type == "cnn_rnn":
            # CNN features already return [T, B, C]
            seq = self.features(x)      # [B, C, H=1, W]
            logits = self.top(seq)      # [T, B, nclasses]
            return logits

        elif self.arch_type == "vit_rgts":
            # ViT backbone returns [T, B, D]
            seq_tokens, reg_tokens, grid_size = self.backbone(x)   # [T, B, D]

            # Adapt to the 4D format expected by CTCtop{C,R,B}:
            # original heads expect [B, C, H, W] with H=1, W=T
            # ViT output is [T, B, D] -> [B, D, 1, T]
            seq_4d = seq_tokens.permute(1, 2, 0).unsqueeze(2)      # [B, D, 1, T]

            logits = self.top(seq_4d)                              # [T, B, nclasses]
            return logits
        
        elif self.arch_type == "torchvision_vit":
            # TorchVision ViT backbone returns [T, B, D]
            seq_tokens, cls_token, grid_size = self.backbone(x)    # [T, B, D]
            
            # Adapt to 4D format for CTC head
            seq_4d = seq_tokens.permute(1, 2, 0).unsqueeze(2)      # [B, D, 1, T]
            
            logits = self.top(seq_4d)                              # [T, B, nclasses]
            return logits
        
        elif self.arch_type == "trocr":
            # TrOCR encoder returns [T, B, D]
            seq_tokens, grid_size = self.backbone(x)               # [T, B, D]
            
            # Adapt to 4D format for CTC head
            seq_4d = seq_tokens.permute(1, 2, 0).unsqueeze(2)      # [B, D, 1, T]
            
            logits = self.top(seq_4d)                              # [T, B, nclasses]
            return logits

        elif self.arch_type == "cnn_mamba":
            # CNN feature extraction
            cnn_out = self.features(x)      # [B, C, H=1, W]
            
            # Reshape to sequence: [B, C, 1, W] -> [B, C, W] -> [W, B, C]
            seq = cnn_out.squeeze(2).permute(2, 0, 1)  # [T, B, C]
            
            # Project to Mamba dimension if needed
            if hasattr(self, 'cnn_to_mamba'):
                # [T, B, C] -> [T, B, D]
                seq = self.cnn_to_mamba(seq)
            
            # Mamba sequence modeling: [T, B, D] -> [T, B, D]
            seq = self.mamba(seq)
            
            # CTC head: [T, B, D] -> [T, B, nclasses]
            logits = self.top(seq)
            return logits

    @torch.no_grad()
    def forward_explain(self, x):
        """
        Returns logits + attention + registers + norms for analysis
        Supports: vit_rgts, torchvision_vit, trocr
        """
        if self.arch_type == "vit_rgts":
            seq_tokens, reg_tokens, attn_maps, token_norms, grid = self.backbone.forward_explain(x)

            # reshape seq_tokens to 4D for the top head
            seq_4d = seq_tokens.permute(1, 2, 0).unsqueeze(2)  # [B, D, 1, T]
            logits = self.top(seq_4d)  # [T, B, C]

            return logits, reg_tokens, attn_maps, token_norms, grid
        
        elif self.arch_type == "torchvision_vit":
            seq_tokens, cls_token, reg_tokens, attn_maps, token_norms, grid = self.backbone.forward_explain(x)
            
            # reshape seq_tokens to 4D for the top head
            seq_4d = seq_tokens.permute(1, 2, 0).unsqueeze(2)  # [B, D, 1, T]
            logits = self.top(seq_4d)  # [T, B, C]
            
            return logits, reg_tokens, attn_maps, token_norms, grid
        
        elif self.arch_type == "trocr":
            seq_tokens, cls_token, attn_maps, token_norms, grid = self.backbone.forward_explain(x)
            
            # reshape seq_tokens to 4D for the top head
            seq_4d = seq_tokens.permute(1, 2, 0).unsqueeze(2)  # [B, D, 1, T]
            logits = self.top(seq_4d)  # [T, B, C]
            
            # TrOCR doesn't have register tokens, return None for compatibility
            return logits, None, attn_maps, token_norms, grid
        
        else:
            raise ValueError(f"forward_explain not supported for architecture: {self.arch_type}")
