"""
Fine-tuning utilities for pretrained models (TorchVision ViT-B, TrOCR Base).

Implements:
  1. Layer-wise Learning Rate Decay (LLRD)
  2. Gradual unfreezing schedule
  3. Parameter group builders for differential LR/WD
  4. Freeze/unfreeze utilities

References:
  - Howard & Ruder, "Universal Language Model Fine-tuning" (2018) — gradual unfreezing
  - Clark et al., "ELECTRA" (2020) — layer-wise LR decay
  - He et al., "MAE" (2022) — ViT fine-tuning with LLRD

Usage:
    from utils.finetuning import build_finetune_optimizer, GradualUnfreezer

    optimizer = build_finetune_optimizer(net, arch_type='torchvision_vit', cfg=config)
    unfreezer = GradualUnfreezer(net, arch_type='torchvision_vit', cfg=config)
"""

import torch
from torch import nn


# --------------------------------------------------------------------------- #
# Layer-wise Learning Rate Decay (LLRD)
# --------------------------------------------------------------------------- #

def get_vit_layer_groups(net, arch_type):
    """
    Partition model parameters into layer groups for LLRD.

    Returns list of (group_name, params_list) ordered from deepest (head)
    to shallowest (embedding/patch projection).

    TorchVision ViT structure:
        vit.conv_proj         → patch embedding (layer 0)
        vit.encoder.layers.0  → transformer block 0 (shallowest)
        ...
        vit.encoder.layers.11 → transformer block 11 (deepest)
        vit.encoder.ln        → final layer norm
        top.*                  → CTC head (learns fastest)

    TrOCR structure:
        encoder.embeddings            → patch + position embedding (layer 0)
        encoder.encoder.layers.0      → transformer block 0
        ...
        encoder.encoder.layers.11     → transformer block 11
        encoder.layernorm             → final layer norm
        top.*                         → CTC head
    """
    groups = []

    if arch_type == 'torchvision_vit':
        # Group 0: Patch embedding + positional encoding
        embed_params = []
        for n, p in net.named_parameters():
            if 'backbone.vit.conv_proj' in n or 'backbone.vit.class_token' in n:
                embed_params.append(p)
            elif 'backbone.vit.encoder.pos_embedding' in n:
                embed_params.append(p)
        if embed_params:
            groups.append(('embedding', embed_params))

        # Groups 1-12: Transformer blocks
        num_layers = 12  # ViT-B has 12 layers
        for layer_idx in range(num_layers):
            layer_params = [p for n, p in net.named_parameters()
                           if f'backbone.vit.encoder.layers.encoder_layer_{layer_idx}.' in n]
            if layer_params:
                groups.append((f'block_{layer_idx}', layer_params))

        # Group 13: Final layer norm
        ln_params = [p for n, p in net.named_parameters()
                     if 'backbone.vit.encoder.ln' in n]
        if ln_params:
            groups.append(('final_ln', ln_params))

        # Group 14: Adapter layers (gray_to_rgb, register tokens)
        adapter_params = [p for n, p in net.named_parameters()
                         if 'backbone.gray_to_rgb' in n or 'backbone.register_tokens' in n]
        if adapter_params:
            groups.append(('adapters', adapter_params))

        # Group 15: CTC head (highest LR)
        head_params = [p for n, p in net.named_parameters()
                       if n.startswith('top.')]
        if head_params:
            groups.append(('head', head_params))

    elif arch_type == 'trocr':
        # Group 0: Embeddings
        embed_params = [p for n, p in net.named_parameters()
                       if 'backbone.encoder.embeddings' in n]
        if embed_params:
            groups.append(('embedding', embed_params))

        # Groups 1-12: Encoder layers
        num_layers = 12  # TrOCR-base has 12 layers
        for layer_idx in range(num_layers):
            layer_params = [p for n, p in net.named_parameters()
                           if f'backbone.encoder.encoder.layer.{layer_idx}.' in n]
            if layer_params:
                groups.append((f'block_{layer_idx}', layer_params))

        # Group 13: Final layer norm
        ln_params = [p for n, p in net.named_parameters()
                     if 'backbone.encoder.layernorm' in n]
        if ln_params:
            groups.append(('final_ln', ln_params))

        # Group 14: Adapter (gray_to_rgb)
        adapter_params = [p for n, p in net.named_parameters()
                         if 'backbone.gray_to_rgb' in n]
        if adapter_params:
            groups.append(('adapters', adapter_params))

        # Group 15: CTC head
        head_params = [p for n, p in net.named_parameters()
                       if n.startswith('top.')]
        if head_params:
            groups.append(('head', head_params))

    return groups


def build_finetune_param_groups(net, arch_type, base_lr, head_lr, lr_decay_rate=0.9,
                                 weight_decay=0.01, head_wd=0.0, no_decay_keywords=None):
    """
    Build optimizer parameter groups with layer-wise LR decay.

    LR schedule (from head to embedding):
        head_lr             → head, adapters
        base_lr * decay^0   → final_ln (deepest backbone layer)
        base_lr * decay^1   → block_11
        base_lr * decay^2   → block_10
        ...
        base_lr * decay^12  → block_0
        base_lr * decay^13  → embedding (lowest LR)

    Parameters
    ----------
    net : HTRNet
    arch_type : str
    base_lr : float — LR for the deepest backbone layer (e.g., 2e-5)
    head_lr : float — LR for head/adapters (e.g., 5e-4)
    lr_decay_rate : float — multiplicative decay per layer (e.g., 0.9)
    weight_decay : float — default WD for backbone
    head_wd : float — WD for head (typically 0)
    no_decay_keywords : list[str] — param names that skip WD (bias, norm)

    Returns
    -------
    param_groups : list[dict] — ready for torch.optim.AdamW
    """
    if no_decay_keywords is None:
        no_decay_keywords = ['bias', 'LayerNorm', 'layernorm', 'layer_norm', 'ln']

    groups = get_vit_layer_groups(net, arch_type)

    # Reverse so index 0 = head (highest LR), last = embedding (lowest)
    # groups are already ordered shallow→deep, so we reverse for LR assignment
    # Actually groups are: [embedding, block_0, ..., block_11, final_ln, adapters, head]
    # LR assignment: head/adapters get head_lr, backbone layers get decayed base_lr
    num_backbone_groups = len(groups) - 2  # exclude adapters and head

    param_groups = []
    for group_idx, (name, params) in enumerate(groups):
        if name in ('head', 'adapters'):
            lr = head_lr
            wd = head_wd
        else:
            # Distance from top of backbone (final_ln is distance 0)
            # final_ln idx = num_backbone_groups - 1 in the backbone portion
            backbone_idx = group_idx  # 0=embedding, ..., N-1=final_ln
            distance_from_top = (num_backbone_groups - 1) - backbone_idx
            lr = base_lr * (lr_decay_rate ** distance_from_top)
            wd = weight_decay

        # Split into decay / no-decay sub-groups
        decay_params = []
        no_decay_params = []
        for p in params:
            if not p.requires_grad:
                continue
            # Check if param name matches no-decay keywords
            # We use a heuristic: 1-D params are typically bias/norm
            if p.dim() == 1:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        if decay_params:
            param_groups.append({
                'params': decay_params,
                'lr': lr,
                'weight_decay': wd,
                'group_name': f'{name}_decay',
            })
        if no_decay_params:
            param_groups.append({
                'params': no_decay_params,
                'lr': lr,
                'weight_decay': 0.0,
                'group_name': f'{name}_no_decay',
            })

    return param_groups


# --------------------------------------------------------------------------- #
# Gradual Unfreezing
# --------------------------------------------------------------------------- #

class GradualUnfreezer:
    """
    Gradually unfreeze transformer layers during training.

    Schedule: Start with only head + adapters trainable, then unfreeze
    one block per `unfreeze_every` epochs (from deepest to shallowest).

    Example (12-layer ViT, unfreeze_every=3, warmup_frozen=5):
        Epoch 1-5:   Only head + adapters trainable
        Epoch 6-8:   + block_11 unfrozen
        Epoch 9-11:  + block_10 unfrozen
        ...
        Epoch 39+:   All layers unfrozen
    """

    def __init__(self, net, arch_type, warmup_frozen=5, unfreeze_every=3):
        """
        Parameters
        ----------
        net : HTRNet
        arch_type : str — 'torchvision_vit' or 'trocr'
        warmup_frozen : int — epochs to keep backbone fully frozen
        unfreeze_every : int — epochs between unfreezing next layer
        """
        self.net = net
        self.arch_type = arch_type
        self.warmup_frozen = warmup_frozen
        self.unfreeze_every = unfreeze_every

        # Get layer groups (ordered: embedding, block_0, ..., block_11, final_ln, adapters, head)
        self.groups = get_vit_layer_groups(net, arch_type)

        # Identify backbone groups (to freeze/unfreeze)
        self.backbone_group_names = [name for name, _ in self.groups
                                     if name not in ('head', 'adapters')]

        # Initially freeze all backbone parameters
        self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze all backbone layers."""
        for name, params in self.groups:
            if name in ('head', 'adapters'):
                for p in params:
                    p.requires_grad = True
            else:
                for p in params:
                    p.requires_grad = False

    def step(self, epoch):
        """
        Call at the start of each epoch. Returns number of unfrozen layers.

        Parameters
        ----------
        epoch : int (1-indexed)

        Returns
        -------
        n_unfrozen : int — number of backbone layers currently unfrozen
        """
        if epoch <= self.warmup_frozen:
            return 0

        # How many layers to unfreeze (from deepest first)
        elapsed = epoch - self.warmup_frozen
        n_to_unfreeze = min(elapsed // self.unfreeze_every + 1,
                            len(self.backbone_group_names))

        # Unfreeze from deepest (final_ln, block_11, block_10, ...) to shallowest
        reversed_names = list(reversed(self.backbone_group_names))
        for i, name in enumerate(reversed_names):
            group_params = next(params for gname, params in self.groups if gname == name)
            if i < n_to_unfreeze:
                for p in group_params:
                    p.requires_grad = True
            else:
                for p in group_params:
                    p.requires_grad = False

        return n_to_unfreeze


# --------------------------------------------------------------------------- #
# Optimizer Builder (high-level API)
# --------------------------------------------------------------------------- #

def build_finetune_optimizer(net, arch_type, cfg):
    """
    Build AdamW optimizer with LLRD for fine-tuning pretrained models.

    Reads hyperparameters from cfg.finetune namespace:
        cfg.finetune.base_lr        — backbone LR (default: 2e-5)
        cfg.finetune.head_lr        — head LR (default: 5e-4)
        cfg.finetune.lr_decay_rate  — LLRD decay factor (default: 0.9)
        cfg.finetune.weight_decay   — backbone WD (default: 0.01)
        cfg.finetune.head_wd        — head WD (default: 0.0)

    Parameters
    ----------
    net : HTRNet
    arch_type : str
    cfg : OmegaConf config

    Returns
    -------
    optimizer : torch.optim.AdamW
    """
    ft = cfg.finetune if hasattr(cfg, 'finetune') else cfg

    base_lr = getattr(ft, 'base_lr', 2e-5)
    head_lr = getattr(ft, 'head_lr', 5e-4)
    lr_decay_rate = getattr(ft, 'lr_decay_rate', 0.9)
    weight_decay = getattr(ft, 'weight_decay', 0.01)
    head_wd = getattr(ft, 'head_wd', 0.0)

    param_groups = build_finetune_param_groups(
        net, arch_type,
        base_lr=base_lr,
        head_lr=head_lr,
        lr_decay_rate=lr_decay_rate,
        weight_decay=weight_decay,
        head_wd=head_wd,
    )

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    return optimizer


def build_finetune_scheduler(optimizer, cfg, max_epochs):
    """
    Build warmup + cosine annealing scheduler for fine-tuning.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    cfg : OmegaConf config
    max_epochs : int

    Returns
    -------
    scheduler : torch.optim.lr_scheduler.SequentialLR
    """
    ft = cfg.finetune if hasattr(cfg, 'finetune') else cfg
    warmup_epochs = getattr(ft, 'warmup_epochs', 5)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs - warmup_epochs, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )
    return scheduler
