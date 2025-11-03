"""
Attention Map Visualizer for HTR Models
Modular visualization utilities for generating and saving attention maps
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch.nn.functional as F


class AttentionVisualizer:
    """
    Handles attention map visualization for HTR models.
    Captures intermediate feature maps and generates heatmaps overlaid on input images.
    """
    
    def __init__(self, save_dir: str = './attention_maps', enabled: bool = True):
        """
        Initialize the attention visualizer.
        
        Args:
            save_dir: Directory to save attention maps
            enabled: Whether visualization is enabled (allows zero-overhead when disabled)
        """
        self.save_dir = Path(save_dir)
        self.enabled = enabled
        self.activations = {}
        self.hooks = []
        
        if self.enabled:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def register_hooks(self, model: torch.nn.Module, layer_names: Optional[List[str]] = None):
        """
        Register forward hooks to capture intermediate activations.
        
        Args:
            model: The HTR model
            layer_names: Specific layer names to hook (if None, hooks CNN features)
        """
        if not self.enabled:
            return
        
        self.clear_hooks()
        
        # Hook into CNN features by default
        if layer_names is None:
            # Hook the CNN output (before RNN)
            if hasattr(model, 'features'):
                hook = model.features.register_forward_hook(
                    self._get_activation_hook('cnn_features')
                )
                self.hooks.append(hook)
            
            # Hook RNN if available
            if hasattr(model, 'top') and hasattr(model.top, 'rec'):
                hook = model.top.rec.register_forward_hook(
                    self._get_activation_hook('rnn_features')
                )
                self.hooks.append(hook)
        else:
            # Hook specific named layers
            for name, module in model.named_modules():
                if name in layer_names:
                    hook = module.register_forward_hook(
                        self._get_activation_hook(name)
                    )
                    self.hooks.append(hook)
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
    
    def _get_activation_hook(self, name: str):
        """Create a hook function that saves activations."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                # For RNN outputs (output, hidden)
                self.activations[name] = output[0].detach()
            else:
                self.activations[name] = output.detach()
        return hook
    
    def generate_attention_maps(
        self,
        input_image: torch.Tensor,
        activations: Optional[Dict[str, torch.Tensor]] = None,
        timestep: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate attention heatmaps from activations.
        
        Args:
            input_image: Original input image tensor [1, C, H, W]
            activations: Dictionary of layer activations (uses self.activations if None)
            timestep: Specific timestep to visualize for sequence features
            
        Returns:
            Dictionary mapping layer names to attention heatmaps
        """
        if not self.enabled:
            return {}
        
        if activations is None:
            activations = self.activations
        
        attention_maps = {}
        
        for name, activation in activations.items():
            # Handle different activation shapes
            if len(activation.shape) == 4:  # [B, C, H, W] - CNN features
                # Average across channels to get spatial attention
                attn_map = activation[0].mean(dim=0).cpu().numpy()
            elif len(activation.shape) == 3:  # [T, B, C] - RNN features
                # Use specific timestep or average across time
                if timestep is not None and timestep < activation.shape[0]:
                    attn_map = activation[timestep, 0, :].abs().cpu().numpy()
                else:
                    # Average across sequence for global attention
                    attn_map = activation[:, 0, :].abs().mean(dim=0).cpu().numpy()
                # Expand to 2D for visualization (time dimension)
                attn_map = attn_map.reshape(1, -1)
            else:
                continue
            
            # Normalize to [0, 1]
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            attention_maps[name] = attn_map
        
        return attention_maps
    
    def visualize_and_save(
        self,
        input_image: torch.Tensor,
        original_image: Optional[np.ndarray] = None,
        transcription: str = '',
        prediction: str = '',
        sample_id: str = 'sample',
        save_individual: bool = True
    ):
        """
        Create and save attention visualizations.
        
        Args:
            input_image: Preprocessed input tensor [1, C, H, W]
            original_image: Original image as numpy array (if None, uses input_image)
            transcription: Ground truth text
            prediction: Predicted text
            sample_id: Identifier for the sample
            save_individual: Save individual layer attention maps
        """
        if not self.enabled:
            return
        
        # Generate attention maps
        attention_maps = self.generate_attention_maps(input_image)
        
        if not attention_maps:
            return
        
        # Prepare original image for overlay
        if original_image is None:
            # Convert tensor to numpy
            img_np = input_image[0].permute(1, 2, 0).cpu().numpy()
            if img_np.shape[2] == 1:
                img_np = np.repeat(img_np, 3, axis=2)
        else:
            img_np = original_image
            if len(img_np.shape) == 2:
                img_np = np.stack([img_np] * 3, axis=-1)
        
        # Normalize image to [0, 1]
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        # Create figure with subplots
        n_maps = len(attention_maps)
        fig, axes = plt.subplots(n_maps + 1, 1, figsize=(15, 3 * (n_maps + 1)))
        
        if n_maps == 0:
            axes = [axes]
        
        # Plot original image
        axes[0].imshow(img_np)
        axes[0].set_title(f'Original Image\nGT: {transcription}\nPred: {prediction}', fontsize=10)
        axes[0].axis('off')
        
        # Plot attention maps
        for idx, (name, attn_map) in enumerate(attention_maps.items(), start=1):
            # Resize attention map to match input image size
            attn_resized = cv2.resize(attn_map, (img_np.shape[1], img_np.shape[0]))
            
            # Create heatmap
            heatmap = plt.cm.jet(attn_resized)[:, :, :3]
            
            # Overlay on original image
            overlay = 0.4 * img_np + 0.6 * heatmap
            
            axes[idx].imshow(overlay)
            axes[idx].set_title(f'Attention Map: {name}', fontsize=10)
            axes[idx].axis('off')
            
            # Save individual attention map if requested
            if save_individual:
                individual_fig, individual_ax = plt.subplots(1, 1, figsize=(15, 3))
                individual_ax.imshow(overlay)
                individual_ax.set_title(f'{name} - GT: {transcription} | Pred: {prediction}', fontsize=10)
                individual_ax.axis('off')
                individual_path = self.save_dir / f'{sample_id}_{name}.png'
                individual_fig.savefig(individual_path, bbox_inches='tight', dpi=150)
                plt.close(individual_fig)
        
        # Save combined figure
        combined_path = self.save_dir / f'{sample_id}_combined.png'
        fig.tight_layout()
        fig.savefig(combined_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        
        print(f'Attention maps saved to {self.save_dir}')
    
    def visualize_sequence_attention(
        self,
        input_image: torch.Tensor,
        rnn_activations: torch.Tensor,
        decoded_sequence: str,
        ground_truth: str = '',
        sample_id: str = 'sequence'
    ):
        """
        Visualize attention across sequence timesteps.
        
        Args:
            input_image: Input image tensor [1, C, H, W]
            rnn_activations: RNN output activations [T, B, C]
            decoded_sequence: Decoded character sequence
            ground_truth: Ground truth transcription
            sample_id: Sample identifier
        """
        if not self.enabled:
            return
        
        seq_len = rnn_activations.shape[0]
        
        # Compute attention weights (magnitude of activations)
        attention_weights = rnn_activations[:, 0, :].abs().mean(dim=1).cpu().numpy()
        attention_weights = attention_weights / (attention_weights.max() + 1e-8)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6))
        
        # Plot input image
        img_np = input_image[0].permute(1, 2, 0).cpu().numpy().squeeze()
        ax1.imshow(img_np, cmap='gray')
        ax1.set_title(f'Input Image\nGT: {ground_truth} | Pred: {decoded_sequence}')
        ax1.axis('off')
        
        # Plot sequence attention
        timesteps = np.arange(seq_len)
        ax2.bar(timesteps, attention_weights, color='steelblue', alpha=0.7)
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Attention Weight')
        ax2.set_title('Sequence Attention Weights')
        ax2.grid(True, alpha=0.3)
        
        # Annotate with decoded characters if available
        if len(decoded_sequence) > 0:
            for i, char in enumerate(decoded_sequence[:min(len(decoded_sequence), seq_len)]):
                ax2.text(i, attention_weights[min(i, seq_len-1)], char, 
                        ha='center', va='bottom', fontsize=8)
        
        save_path = self.save_dir / f'{sample_id}_sequence_attention.png'
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
    
    def visualize_transformer_attention(
        self,
        input_image: torch.Tensor,
        attention_weights: List[torch.Tensor],
        transcription: str = '',
        prediction: str = '',
        sample_id: str = 'transformer',
        layers_to_visualize: Optional[List[int]] = None
    ):
        """
        Visualize self-attention maps from Vision Transformer.
        
        Args:
            input_image: Input image tensor [1, C, H, W]
            attention_weights: List of attention tensors [B, H, N, N] for each layer
            transcription: Ground truth text
            prediction: Predicted text
            sample_id: Sample identifier
            layers_to_visualize: Which layers to visualize (default: [0, mid, last])
        """
        if not self.enabled or not attention_weights:
            return
        
        num_layers = len(attention_weights)
        
        # Select layers to visualize
        if layers_to_visualize is None:
            if num_layers >= 3:
                layers_to_visualize = [0, num_layers // 2, num_layers - 1]
            else:
                layers_to_visualize = list(range(num_layers))
        
        # Prepare original image
        img_np = input_image[0].permute(1, 2, 0).cpu().numpy().squeeze()
        if len(img_np.shape) == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        # Create figure with subplots
        n_layers = len(layers_to_visualize)
        fig, axes = plt.subplots(n_layers + 1, 1, figsize=(15, 3 * (n_layers + 1)))
        
        if n_layers == 0:
            axes = [axes]
        
        # Plot original image
        axes[0].imshow(img_np)
        axes[0].set_title(f'Original Image\nGT: {transcription}\nPred: {prediction}', fontsize=10)
        axes[0].axis('off')
        
        # Plot attention maps for selected layers
        for idx, layer_idx in enumerate(layers_to_visualize, start=1):
            attn = attention_weights[layer_idx]  # [B, H, N, N]
            
            # Average across heads and batch
            attn_map = attn[0].mean(dim=0).cpu().numpy()  # [N, N]
            
            # Average attention from all patches (mean over source dimension)
            attn_avg = attn_map.mean(axis=0)  # [N] - attention received by each patch
            
            # Reshape to 2D grid (approximate original image layout)
            num_patches = attn_avg.shape[0]
            # Calculate grid dimensions (assuming square-ish patches)
            grid_h = int(np.sqrt(num_patches))
            grid_w = num_patches // grid_h
            
            if grid_h * grid_w < num_patches:
                grid_w += 1
            
            # Pad if necessary
            attn_2d = np.zeros(grid_h * grid_w)
            attn_2d[:num_patches] = attn_avg
            attn_2d = attn_2d.reshape(grid_h, grid_w)
            
            # Normalize
            attn_2d = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)
            
            # Resize to match input image size
            attn_resized = cv2.resize(attn_2d, (img_np.shape[1], img_np.shape[0]))
            
            # Create heatmap
            heatmap = plt.cm.jet(attn_resized)[:, :, :3]
            
            # Overlay on original image
            overlay = 0.4 * img_np + 0.6 * heatmap
            
            axes[idx].imshow(overlay)
            axes[idx].set_title(f'Transformer Layer {layer_idx} Self-Attention', fontsize=10)
            axes[idx].axis('off')
        
        # Save combined figure
        combined_path = self.save_dir / f'{sample_id}_transformer_attention.png'
        fig.tight_layout()
        fig.savefig(combined_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        
        print(f'Transformer attention maps saved to {self.save_dir}')
    
    def visualize_attention_rollout(
        self,
        input_image: torch.Tensor,
        attention_weights: List[torch.Tensor],
        transcription: str = '',
        prediction: str = '',
        sample_id: str = 'rollout'
    ):
        """
        Visualize attention rollout - accumulated attention across all layers.
        Shows the complete attention flow from input to output.
        
        Args:
            input_image: Input image tensor [1, C, H, W]
            attention_weights: List of attention tensors [B, H, N, N]
            transcription: Ground truth text
            prediction: Predicted text
            sample_id: Sample identifier
        """
        if not self.enabled or not attention_weights:
            return
        
        # Compute attention rollout
        # Start with identity matrix
        num_patches = attention_weights[0].shape[-1]
        rollout = torch.eye(num_patches).to(attention_weights[0].device)
        
        for attn in attention_weights:
            # Average across heads
            attn_avg = attn[0].mean(dim=0)  # [N, N]
            
            # Add identity (residual connection)
            attn_avg = attn_avg + torch.eye(num_patches).to(attn_avg.device)
            
            # Normalize
            attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
            
            # Multiply with previous rollout
            rollout = torch.matmul(attn_avg, rollout)
        
        # Extract attention to first patch (or CLS token)
        rollout_attention = rollout[0, :].cpu().numpy()
        
        # Reshape to 2D grid
        num_patches_total = rollout_attention.shape[0]
        grid_h = int(np.sqrt(num_patches_total))
        grid_w = num_patches_total // grid_h
        
        if grid_h * grid_w < num_patches_total:
            grid_w += 1
        
        attn_2d = np.zeros(grid_h * grid_w)
        attn_2d[:num_patches_total] = rollout_attention
        attn_2d = attn_2d.reshape(grid_h, grid_w)
        
        # Normalize
        attn_2d = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)
        
        # Prepare original image
        img_np = input_image[0].permute(1, 2, 0).cpu().numpy().squeeze()
        if len(img_np.shape) == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        # Resize attention map
        attn_resized = cv2.resize(attn_2d, (img_np.shape[1], img_np.shape[0]))
        
        # Create heatmap
        heatmap = plt.cm.jet(attn_resized)[:, :, :3]
        overlay = 0.4 * img_np + 0.6 * heatmap
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Original image
        ax1.imshow(img_np)
        ax1.set_title(f'Original Image\nGT: {transcription}\nPred: {prediction}')
        ax1.axis('off')
        
        # Attention rollout
        ax2.imshow(overlay)
        ax2.set_title('Attention Rollout (All Layers Combined)')
        ax2.axis('off')
        
        save_path = self.save_dir / f'{sample_id}_attention_rollout.png'
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self.clear_hooks()


class AttentionHook:
    """
    Context manager for temporary attention visualization during inference.
    """
    
    def __init__(self, model: torch.nn.Module, visualizer: AttentionVisualizer):
        self.model = model
        self.visualizer = visualizer
    
    def __enter__(self):
        self.visualizer.register_hooks(self.model)
        return self.visualizer
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.visualizer.clear_hooks()
