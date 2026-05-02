#!/usr/bin/env python3
"""
Run HTR Inference on Handwritten Text Images

Loads a trained model and predicts text from one or more handwriting images.
Supports single image, multiple images, or an entire directory.
Saves predictions to a 'predicted/' folder inside the model's run directory.

Supported Architectures: ALL (cnn_rnn, vit_rgts, torchvision_vit, trocr)
Input: config YAML(s) + model checkpoint + image path(s) or directory
Output: predictions.txt in <run_folder>/predicted/

Usage:
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline && python scripts/postprocessing/demo.py configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml arch.num_registers=16 resume=saved_models/experiments/run_54/model.pt -- /home/hpc/iwi5/iwi5369h/HTR-Pipeline/notebook/sample_images/a01-096u-10.png



    # Single image
    python scripts/postprocessing/demo.py configs/config.yaml configs/baseline.yaml \
        configs/baseline_vit_rgts_v2.yaml arch.num_registers=16 \
        resume=saved_models/experiments/run_54/model.pt \
        -- path/to/image.png

    # Multiple images
    python scripts/postprocessing/demo.py configs/config.yaml configs/baseline.yaml \
        resume=saved_models/experiments/run_32/model.pt \
        -- img1.png img2.png img3.png

    # Directory of images
    python scripts/postprocessing/demo.py configs/config.yaml configs/baseline.yaml \
        configs/baseline_vit_rgts_v2.yaml arch.num_registers=16 \
        resume=saved_models/experiments/run_54/model.pt \
        -- path/to/image_folder/
"""

from omegaconf import OmegaConf
from datetime import datetime

import sys
import os
import glob

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import tqdm
import numpy as np
import torch
import torch.nn as nn

from models import HTRNet
from utils.preprocessing import load_image, preprocess

# Supported image extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


class HTREval(nn.Module):
    def __init__(self, config):
        super(HTREval, self).__init__()
        self.config = config

        self.prepare_dataloaders()
        self.prepare_net()

    def prepare_dataloaders(self):

        config = self.config

        # prepare datset loader
        dataset_folder = config.data.path
        # load classes from the training set saved in the data folder
        classes = np.load(os.path.join(dataset_folder, 'classes.npy'))

        # create dictionaries for character to index and index to character 
        # 0 index is reserved for CTC blank
        cdict = {c:(i+1) for i,c in enumerate(classes)}
        icdict = {(i+1):c for i,c in enumerate(classes)}

        self.classes = {
            'classes': classes,
            'c2i': cdict,
            'i2c': icdict
        }

    def prepare_net(self):

        config = self.config

        device = config.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
            self.config.device = "cpu"

        print('Preparing Net - Architectural elements:')
        print(config.arch)

        classes = self.classes['classes']

        net = HTRNet(config.arch, len(classes) + 1)
        
        if config.resume is not None:
            print(f'resuming from checkpoint: {config.resume}')
            load_dict = torch.load(config.resume, map_location=device)
            load_status = net.load_state_dict(load_dict, strict=True)
            print(load_status)
        net.to(device)

        # print number of parameters
        n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f'Number of parameters: {n_params}')

        self.net = net

    def decode(self, tdec, tdict, blank_id=0):
        
        tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
        dec_transcr = ''.join([tdict[t] for t in tt if t != blank_id])
        
        return dec_transcr
                
    def predict(self, img_path):
        """Run inference on a single image and return predicted text."""
        img = load_image(img_path)
        img = preprocess(img, (self.config.preproc.image_height, self.config.preproc.image_width))
        img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)

        img = img.to(self.config.device)

        self.net.eval()
        with torch.no_grad():
            tst_o = self.net(img)
            # In eval mode, head_type='both' already returns single tensor
            # Only unpack tuple if model is in training mode (shouldn't happen here)
            if isinstance(tst_o, tuple):
                tst_o = tst_o[0]

        tdec = tst_o.argmax(2).permute(1, 0).cpu().numpy().squeeze()
        dec_transcr = self.decode(tdec, self.classes['i2c'])

        return dec_transcr.strip()

    def predict_batch(self, img_paths, output_dir):
        """Run inference on multiple images and save results to output_dir."""
        os.makedirs(output_dir, exist_ok=True)

        results = []
        for img_path in tqdm.tqdm(img_paths, desc='Predicting'):
            try:
                prediction = self.predict(img_path)
                results.append((os.path.basename(img_path), prediction))
                print(f'  {os.path.basename(img_path):>40s}  ->  {prediction}')
            except Exception as e:
                results.append((os.path.basename(img_path), f'[ERROR: {e}]'))
                print(f'  {os.path.basename(img_path):>40s}  ->  [ERROR: {e}]')

        # Save predictions.txt
        out_path = os.path.join(output_dir, 'predictions.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f'# HTR Predictions  —  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'# Model: {self.config.resume}\n')
            f.write(f'# Images: {len(results)}\n')
            f.write('#' + '-' * 59 + '\n')
            for fname, pred in results:
                f.write(f'{fname}\t{pred}\n')

        print(f'\n{len(results)} predictions saved to: {out_path}')
        return results


def collect_image_paths(paths):
    """Resolve a list of paths into individual image file paths.
    
    Each entry can be a single image file or a directory (scanned non-recursively).
    """
    image_paths = []
    for p in paths:
        if os.path.isdir(p):
            for fname in sorted(os.listdir(p)):
                if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                    image_paths.append(os.path.join(p, fname))
        elif os.path.isfile(p):
            image_paths.append(p)
        else:
            print(f'Warning: skipping {p} (not a file or directory)')
    return image_paths


def parse_args():
    """Parse config YAML(s), key=value overrides, and image paths separated by --."""
    # Split argv at '--' separator
    argv = sys.argv[1:]
    if '--' in argv:
        sep_idx = argv.index('--')
        config_args = argv[:sep_idx]
        image_args = argv[sep_idx + 1:]
    else:
        # Legacy mode: last arg is the image path
        config_args = argv[:-1]
        image_args = [argv[-1]] if argv else []

    yaml_files = [a for a in config_args if a.endswith('.yaml')]
    overrides  = [a for a in config_args if not a.endswith('.yaml')]

    if not yaml_files:
        print('Error: at least one .yaml config file is required.')
        sys.exit(1)

    conf = OmegaConf.load(yaml_files[0])
    for yf in yaml_files[1:]:
        conf = OmegaConf.merge(conf, OmegaConf.load(yf))

    OmegaConf.set_struct(conf, False)
    cli_conf = OmegaConf.from_dotlist(overrides)
    conf = OmegaConf.merge(conf, cli_conf)

    return conf, image_args


if __name__ == '__main__':
    # ----------------------- initialize configuration ----------------------- #
    config, image_args = parse_args()

    if not image_args:
        print('Error: no image path(s) provided. Use -- followed by image paths.')
        sys.exit(1)

    # Resolve image paths (files and/or directories)
    image_paths = collect_image_paths(image_args)
    if not image_paths:
        print('Error: no valid image files found in the given paths.')
        sys.exit(1)

    # Derive output directory: <run_folder>/predicted/
    if hasattr(config, 'resume') and config.resume:
        run_dir = os.path.dirname(os.path.abspath(config.resume))
    else:
        run_dir = 'output'
    output_dir = os.path.join(run_dir, 'predicted')

    print(f'Images found : {len(image_paths)}')
    print(f'Output dir   : {output_dir}')
    print()

    htr_eval = HTREval(config)
    htr_eval.predict_batch(image_paths, output_dir)


    