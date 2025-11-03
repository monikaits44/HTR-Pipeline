"""
Standalone Attention Visualization Script
Generates attention maps for HTR model predictions without modifying training code.

Usage:
    python visualize_attention.py --model saved_models/experiments/run_1/model.pt \
                                   --config config.yaml \
                                   --num_samples 10 \
                                   --output_dir ./attention_visualizations
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models import HTRNet
from models_vit import HTRViT, create_vit_htr_model
from models_vit_pretrained import PretrainedViTHTR, create_pretrained_vit_htr_model
from utils.htr_dataset import HTRDataset
from utils.visualizer import AttentionVisualizer, AttentionHook
from torch.utils.data import DataLoader


def detect_model_type(model):
    """
    Detect whether the model is CNN-RNN, ViT, or Pretrained ViT based.
    
    Args:
        model: PyTorch model instance
    
    Returns:
        str: 'vit_pretrained', 'vit', or 'cnn-rnn'
    """
    if hasattr(model, 'model_type'):
        return model.model_type
    elif isinstance(model, PretrainedViTHTR):
        return 'vit_pretrained'
    elif isinstance(model, HTRViT):
        return 'vit'
    elif hasattr(model, 'vit'):  # PretrainedViTHTR has .vit attribute
        return 'vit_pretrained'
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'blocks'):
        return 'vit'
    else:
        return 'cnn-rnn'


def decode_prediction(output, classes_dict, blank_id=0):
    """Decode CTC output to text."""
    # output shape: [T, B, C] or [T, C]
    if len(output.shape) == 3:
        output = output[:, 0, :]  # Take first batch element
    
    tdec = output.argmax(1).cpu().numpy()
    # Remove duplicates
    tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
    # Remove blanks and convert to text
    dec_transcr = ''.join([classes_dict[t] for t in tt if t != blank_id])
    return dec_transcr


def visualize_samples(
    model_path: str,
    config_path: str,
    num_samples: int = 10,
    output_dir: str = './attention_visualizations',
    dataset_split: str = 'test',
    device: str = 'cuda:0'
):
    """
    Generate attention visualizations for sample images.
    
    Args:
        model_path: Path to trained model checkpoint
        config_path: Path to config file
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualizations
        dataset_split: Dataset split to use ('train', 'val', or 'test')
        device: Device to run inference on
    """
    
    # Load configuration
    config = OmegaConf.load(config_path)
    
    # Load character classes
    classes = np.load(Path(config.data.path) / 'classes.npy', allow_pickle=True)
    cdict = {c: (i + 1) for i, c in enumerate(classes)}
    icdict = {(i + 1): c for i, c in enumerate(classes)}
    
    # Load model - try Pretrained ViT, then ViT, fallback to CNN-RNN
    print(f'Loading model from {model_path}...')
    try:
        if hasattr(config, 'arch_vit_pretrained'):
            model = create_pretrained_vit_htr_model(config, len(classes) + 1)
            model_type = 'vit_pretrained'
        elif hasattr(config, 'arch_vit'):
            model = create_vit_htr_model(config, len(classes) + 1)
            model_type = 'vit'
        else:
            model = HTRNet(config.arch, len(classes) + 1)
            model_type = 'cnn-rnn'
    except Exception as e:
        print(f"Warning: Error loading model with config type: {e}")
        # Fallback to CNN-RNN
        model = HTRNet(config.arch, len(classes) + 1)
        model_type = 'cnn-rnn'
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Detect model type
    model_type = detect_model_type(model)
    print(f'Model type detected: {model_type.upper()}')
    
    # Load dataset
    print(f'Loading {dataset_split} dataset...')
    fixed_size = (config.preproc.image_height, config.preproc.image_width)
    dataset = HTRDataset(config.data.path, dataset_split, fixed_size=fixed_size, transforms=None)
    
    # Initialize visualizer
    visualizer = AttentionVisualizer(save_dir=output_dir, enabled=True)
    
    print(f'Generating attention maps for {num_samples} samples...')
    
    # Process samples
    for idx in range(min(num_samples, len(dataset))):
        img, transcription = dataset[idx]
        
        # Add batch dimension
        img_batch = img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            if model_type in ['vit', 'vit_pretrained']:
                # ViT-specific visualization (works for both custom and pretrained)
                output = model(img_batch)
                
                # Handle both single output and tuple
                if isinstance(output, tuple):
                    output = output[0]
                
                # Decode prediction
                prediction = decode_prediction(output, icdict)
                
                # Get attention weights from transformer
                attention_weights = model.get_attention_weights()
                
                # Visualize transformer attention
                if attention_weights:
                    visualizer.visualize_transformer_attention(
                        input_image=img_batch,
                        attention_weights=attention_weights,
                        transcription=transcription.strip(),
                        prediction=prediction.strip(),
                        sample_id=f'sample_{idx:04d}',
                        layers_to_visualize=[0, len(attention_weights)//2, len(attention_weights)-1]
                    )
                    
                    # Visualize attention rollout
                    visualizer.visualize_attention_rollout(
                        input_image=img_batch,
                        attention_weights=attention_weights,
                        transcription=transcription.strip(),
                        prediction=prediction.strip(),
                        sample_id=f'sample_{idx:04d}'
                    )
            else:
                # CNN-RNN visualization (original code)
                with AttentionHook(model, visualizer) as viz:
                    output = model(img_batch)
                    
                    # Handle both single output and tuple (for 'both' head type)
                    if isinstance(output, tuple):
                        output = output[0]
                    
                    # Decode prediction
                    prediction = decode_prediction(output, icdict)
                    
                    # Generate and save attention maps
                    viz.visualize_and_save(
                        input_image=img_batch,
                        transcription=transcription.strip(),
                        prediction=prediction.strip(),
                        sample_id=f'sample_{idx:04d}',
                        save_individual=True
                    )
                    
                    # Also create sequence attention visualization if RNN is used
                    if 'rnn_features' in viz.activations:
                        viz.visualize_sequence_attention(
                            input_image=img_batch,
                            rnn_activations=viz.activations['rnn_features'],
                            decoded_sequence=prediction.strip(),
                            ground_truth=transcription.strip(),
                            sample_id=f'sample_{idx:04d}'
                        )
        
        print(f'  [{idx+1}/{num_samples}] GT: "{transcription.strip()}" | Pred: "{prediction.strip()}"')
    
    print(f'\n✓ Attention visualizations saved to: {output_dir}')


def visualize_specific_sample(
    model_path: str,
    config_path: str,
    image_path: str,
    output_dir: str = './attention_visualizations',
    device: str = 'cuda:0'
):
    """
    Visualize attention for a specific image file.
    
    Args:
        model_path: Path to trained model checkpoint
        config_path: Path to config file
        image_path: Path to input image
        output_dir: Directory to save visualizations
        device: Device to run inference on
    """
    import cv2
    
    # Load configuration
    config = OmegaConf.load(config_path)
    
    # Load character classes
    classes = np.load(Path(config.data.path) / 'classes.npy', allow_pickle=True)
    icdict = {(i + 1): c for i, c in enumerate(classes)}
    
    # Load model - try Pretrained ViT, then ViT, fallback to CNN-RNN
    print(f'Loading model from {model_path}...')
    try:
        if hasattr(config, 'arch_vit_pretrained'):
            model = create_pretrained_vit_htr_model(config, len(classes) + 1)
            model_type = 'vit_pretrained'
        elif hasattr(config, 'arch_vit'):
            model = create_vit_htr_model(config, len(classes) + 1)
            model_type = 'vit'
        else:
            model = HTRNet(config.arch, len(classes) + 1)
            model_type = 'cnn-rnn'
    except Exception as e:
        print(f"Warning: Error loading model with config type: {e}")
        # Fallback to CNN-RNN
        model = HTRNet(config.arch, len(classes) + 1)
        model_type = 'cnn-rnn'
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Detect model type
    model_type = detect_model_type(model)
    print(f'Model type detected: {model_type.upper()}')
    
    # Load and preprocess image
    print(f'Loading image from {image_path}...')
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Resize to model input size
    fixed_size = (config.preproc.image_height, config.preproc.image_width)
    img_resized = cv2.resize(img, (fixed_size[1], fixed_size[0]))
    
    # Normalize
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Convert to tensor [1, 1, H, W]
    img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0).to(device)
    
    # Initialize visualizer
    visualizer = AttentionVisualizer(save_dir=output_dir, enabled=True)
    
    with torch.no_grad():
        if model_type in ['vit', 'vit_pretrained']:
            # ViT-specific visualization (works for both custom and pretrained)
            output = model(img_tensor)
            
            if isinstance(output, tuple):
                output = output[0]
            
            prediction = decode_prediction(output, icdict)
            
            # Get attention weights from transformer
            attention_weights = model.get_attention_weights()
            
            # Visualize transformer attention
            if attention_weights:
                visualizer.visualize_transformer_attention(
                    input_image=img_tensor,
                    attention_weights=attention_weights,
                    transcription='',
                    prediction=prediction.strip(),
                    sample_id=Path(image_path).stem,
                    layers_to_visualize=[0, len(attention_weights)//2, len(attention_weights)-1]
                )
                
                # Visualize attention rollout
                visualizer.visualize_attention_rollout(
                    input_image=img_tensor,
                    attention_weights=attention_weights,
                    transcription='',
                    prediction=prediction.strip(),
                    sample_id=Path(image_path).stem
                )
        else:
            # CNN-RNN visualization (original code)
            with AttentionHook(model, visualizer) as viz:
                output = model(img_tensor)
                
                if isinstance(output, tuple):
                    output = output[0]
                
                prediction = decode_prediction(output, icdict)
                
                viz.visualize_and_save(
                    input_image=img_tensor,
                    transcription='',
                    prediction=prediction.strip(),
                    sample_id=Path(image_path).stem,
                    save_individual=True
                )
                
                if 'rnn_features' in viz.activations:
                    viz.visualize_sequence_attention(
                        input_image=img_tensor,
                        rnn_activations=viz.activations['rnn_features'],
                        decoded_sequence=prediction.strip(),
                        ground_truth='',
                        sample_id=Path(image_path).stem
                    )
    
    print(f'\nPrediction: "{prediction.strip()}"')
    print(f'✓ Attention visualizations saved to: {output_dir}')


def main():
    parser = argparse.ArgumentParser(description='Generate attention visualizations for HTR model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint (.pt file)')
    parser.add_argument('--config', type=str, required=True, help='Path to config file (.yaml)')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='./attention_visualizations', 
                       help='Directory to save visualizations')
    parser.add_argument('--dataset', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run on')
    parser.add_argument('--image', type=str, default=None, 
                       help='Specific image path to visualize (overrides --num_samples)')
    
    args = parser.parse_args()
    
    if args.image:
        # Visualize specific image
        visualize_specific_sample(
            model_path=args.model,
            config_path=args.config,
            image_path=args.image,
            output_dir=args.output_dir,
            device=args.device
        )
    else:
        # Visualize samples from dataset
        visualize_samples(
            model_path=args.model,
            config_path=args.config,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            dataset_split=args.dataset,
            device=args.device
        )


if __name__ == '__main__':
    main()
