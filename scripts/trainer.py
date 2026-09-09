import argparse
from omegaconf import OmegaConf

import sys
import os
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.htr_dataset import HTRDataset
from utils.finetuning import (
    build_finetune_optimizer,
    build_finetune_scheduler,
    GradualUnfreezer,
)

from models import HTRNet
from utils.transforms import aug_transforms_cnn, aug_transforms_vit

import torch.nn.functional as F

from utils.metrics import CER, WER
import json
import shutil
from datetime import datetime
import csv
import time
import fcntl


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_next_run_number(experiments_dir):
    """Get the next run number by checking existing run directories."""
    if not os.path.exists(experiments_dir):
        return 1

    existing_runs = [d for d in os.listdir(experiments_dir) if d.startswith('run_')]
    if not existing_runs:
        return 1

    run_numbers = []
    for run_dir in existing_runs:
        try:
            num = int(run_dir.split('_')[1])
            run_numbers.append(num)
        except (IndexError, ValueError):
            continue

    return max(run_numbers) + 1 if run_numbers else 1


def setup_experiment_dir(config):
    """
    Create experiment directory structure: saved_models/experiments/run_<n>/
    Returns the experiment directory path and logging file handle.

    Uses an exclusive file lock (.run_lock) to prevent concurrent SLURM array
    tasks from claiming the same run number simultaneously.
    """
    base_dir = './saved_models'
    experiments_dir = os.path.join(base_dir, 'experiments')
    os.makedirs(experiments_dir, exist_ok=True)

    lock_path = os.path.join(experiments_dir, '.run_lock')
    with open(lock_path, 'w') as lock_fh:
        # Acquire exclusive lock — blocks until no other process holds it
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            run_number = get_next_run_number(experiments_dir)
            run_dir = os.path.join(experiments_dir, f'run_{run_number}')
            # Create directory while still holding the lock so no other
            # process can scan the same max and pick the same number.
            os.makedirs(run_dir, exist_ok=False)
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
    
    # Save config as JSON
    config_dict = OmegaConf.to_container(config, resolve=True)
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    # Create log file
    log_path = os.path.join(run_dir, 'training.log')
    log_file = open(log_path, 'w')
    
    # Log initial information
    log_file.write(f"Experiment: run_{run_number}\n")
    log_file.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Configuration saved to: {config_path}\n")
    log_file.write("="*80 + "\n\n")
    log_file.flush()
    
    # Create CSV results file with header
    csv_path = os.path.join(run_dir, 'results.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'epoch', 'lr', 'train/ctc_loss', 
        'val/cer', 'val/wer', 
        'test/cer', 'test/wer',
        'data/train_lines', 'data/val_lines', 'data/test_lines',
        'data/train_charset_size', 'data/val_charset_size', 'data/test_charset_size',
        'model/params', 'time/epoch(s)', 'device', 'seed'
    ])
    csv_file.flush()
    
    return run_dir, log_file, run_number, csv_file, csv_writer


class TqdmToLogFile:
    def __init__(self, log_file):
        self.log_file = log_file

    def write(self, buf):
        if buf.strip():  # Only write non-empty lines
            self.log_file.write(buf)
            self.log_file.flush()
            
    def flush(self):
        if self.log_file:
            self.log_file.flush()

class HTRTrainer(nn.Module):
    def __init__(self, config, experiment_dir=None, log_file=None, run_number=None, csv_file=None, csv_writer=None):
        super(HTRTrainer, self).__init__()
        self.config = config
        self.experiment_dir = experiment_dir
        self.log_file = log_file
        self.run_number = run_number
        self.csv_file = csv_file
        self.csv_writer = csv_writer
        
        # Fine-tuning: gradual unfreezer (set in prepare_optimizers if enabled)
        self.unfreezer = None
        
        # Create tqdm logger
        self.tqdm_logger = TqdmToLogFile(log_file) if log_file else None
        
        # Tracking variables for CSV
        self.train_losses = []
        self.epoch_start_time = None
        self.num_params = 0
        
        # Setup attention extraction directory for ViT models
        self.attention_dir = None
        if experiment_dir is not None:
            self.attention_dir = os.path.join(experiment_dir, 'attention_weights')
            os.makedirs(self.attention_dir, exist_ok=True)

        # SWA state (populated by prepare_optimizers when swa.enabled=true)
        self.swa_model = None       # AveragedModel — holds the running weight average
        self.swa_scheduler = None   # SWALR — drives LR during the SWA phase
        self.swa_enabled = False    # mirror of config.swa.enabled for fast lookup
        
        # Setup detailed evaluation CSV file
        if experiment_dir is not None:
            self.eval_csv_path = os.path.join(experiment_dir, 'evaluation_details.csv')
            self.eval_csv_file = open(self.eval_csv_path, 'w', newline='')
            self.eval_csv_writer = csv.writer(self.eval_csv_file)
            self.eval_csv_writer.writerow([
                'epoch', 'dataset', 'sample_idx', 'ground_truth', 'prediction', 
                'sample_cer', 'sample_wer', 'gt_length', 'pred_length'
            ])
            self.eval_csv_file.flush()
        else:
            self.eval_csv_file = None
            self.eval_csv_writer = None

        self.prepare_dataloaders()
        self.prepare_net()
        self.prepare_losses()
        self.prepare_optimizers()
    
    def log(self, message):
        """Log message to both console and log file."""
        print(message)
        if self.log_file:
            self.log_file.write(message + '\n')
            self.log_file.flush()

    def _read_character_classes(self, basefolder, subset):
        """Read gt.txt and return set of unique characters.
        
        Used to compute unified character classes when training on synthetic data
        but evaluating on IAM — ensures the model vocabulary covers all characters
        from both domains.
        """
        chars = set()
        gt_path = os.path.join(basefolder, subset, 'gt.txt')
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) >= 2:
                    transcr = ' '.join(parts[1:])
                    chars.update(list(transcr))
        return chars

    def prepare_dataloaders(self):

        config = self.config

        # prepare datset loader
        dataset_folder = config.data.path
        fixed_size = (config.preproc.image_height, config.preproc.image_width)

        # ========================================================================
        # AUTOMATIC AUGMENTATION SELECTION BASED ON ARCHITECTURE
        # ========================================================================
        # ViT architectures need stronger augmentation due to low inductive bias
        arch_type = getattr(config.arch, 'type', 'cnn_rnn')
        
        # Check if user explicitly specified augmentation strategy
        aug_strategy = getattr(config.train, 'augmentation', 'auto')
        
        if aug_strategy == 'auto':
            # Architecture-aware augmentation selection:
            # - CNN-RNN: Moderate augmentation (proven effective, strong inductive bias)
            # - ViT from scratch: Strong (needs invariance training, but not TOO strong
            #   or it destroys signal before the model can learn basics)
            # - Pretrained ViT/TrOCR: Moderate (pretrained features already robust;
            #   over-augmenting destroys the distribution the pretrained weights expect)
            if arch_type == 'cnn_rnn':
                selected_transforms = aug_transforms_cnn
                aug_name = 'cnn (moderate)'
            elif arch_type == 'vit_rgts':
                selected_transforms = aug_transforms_vit
                aug_name = 'vit (strong)'
            elif arch_type == 'htrvt':
                selected_transforms = aug_transforms_cnn
                aug_name = 'cnn (moderate — HTR-VT has strong CNN stem + span-mask)'
            elif arch_type in ['torchvision_vit', 'trocr']:
                selected_transforms = aug_transforms_cnn
                aug_name = 'cnn (moderate — pretrained model)'
            else:
                selected_transforms = aug_transforms_vit
                aug_name = 'vit (strong — fallback)'
            self.log(f"🔄 Auto-selected {aug_name} augmentation for {arch_type}")
        else:
            # User explicitly specified augmentation strategy (override)
            if aug_strategy == 'cnn':
                selected_transforms = aug_transforms_cnn
                aug_name = 'cnn'
            elif aug_strategy == 'vit':
                selected_transforms = aug_transforms_vit
                aug_name = 'vit'
            elif aug_strategy == 'vit_strong':  # removed — use 'vit' instead
                selected_transforms = aug_transforms_vit
                aug_name = 'vit (vit_strong alias)'
            else:
                raise ValueError(f"Unknown augmentation strategy: {aug_strategy}")
            self.log(f"🎯 User-specified augmentation: {aug_name}")
        
        self.log(f"📊 Training with {aug_name} augmentation strategy")

        # ====================================================================
        # DATA MODE: 'iam' (Aachen splits) or 'synthetic' (synth train + IAM test)
        # ====================================================================
        data_mode = getattr(config.data, 'mode', 'iam')

        if data_mode == 'synthetic':
            # ----------------------------------------------------------------
            # SYNTHETIC MODE
            # Train/Val: synthetic processed_lines (extracted from LMDB)
            # Test:      IAM Aachen test split (always — the real benchmark)
            # ----------------------------------------------------------------
            synthetic_folder = getattr(config.data, 'synthetic_path', None)
            if synthetic_folder is None:
                raise ValueError(
                    "data.synthetic_path must be set when data.mode='synthetic'. "
                    "Add to config or CLI: data.synthetic_path=/path/to/synthetic_processed_lines"
                )

            self.log(f'📦 Data mode: SYNTHETIC PRETRAINING')
            self.log(f'   Train/Val source: {synthetic_folder}')
            self.log(f'   Test source (IAM Aachen): {dataset_folder}')

            # Compute unified character classes: synthetic train ∪ IAM test
            # Critical for CTC: model vocabulary must cover all characters it will
            # encounter during both training AND evaluation.
            synth_chars = self._read_character_classes(synthetic_folder, 'train')
            iam_test_chars = self._read_character_classes(dataset_folder, 'test')
            unified_classes = sorted(list(synth_chars | iam_test_chars))
            self.log(f'   Synthetic train chars: {len(synth_chars)}')
            self.log(f'   IAM test chars:        {len(iam_test_chars)}')
            self.log(f'   Unified charset:       {len(unified_classes)} characters')

            # Check for IAM-only characters (model will see them only at test time)
            iam_only = iam_test_chars - synth_chars
            if iam_only:
                self.log(f'   ⚠️  Characters in IAM test but NOT in synthetic train: {sorted(iam_only)}')
                self.log(f'      The model may struggle with these {len(iam_only)} unseen characters.')

            train_set = HTRDataset(synthetic_folder, 'train', fixed_size=fixed_size,
                                   transforms=selected_transforms, character_classes=unified_classes)
            val_set = HTRDataset(synthetic_folder, 'val', fixed_size=fixed_size,
                                 transforms=None, character_classes=unified_classes)
            test_set = HTRDataset(dataset_folder, 'test', fixed_size=fixed_size,
                                  transforms=None, character_classes=unified_classes)
            classes = unified_classes
            classes_save_dir = synthetic_folder

            # Memory advisory for large synthetic datasets
            est_mem_mb = len(train_set) * 200 / (1024 * 1024)  # ~200 bytes per data list entry
            self.log(f'   ℹ️  Dataset index memory: ~{est_mem_mb:.0f} MB '
                     f'(×{config.train.num_workers} DataLoader workers via fork)')
            if len(train_set) > 500000 and config.train.num_workers > 4:
                self.log(f'   ⚠️  Consider reducing num_workers to 4 for memory-constrained nodes')
            iters_per_epoch = len(train_set) // config.train.batch_size
            self.log(f'   ℹ️  ~{iters_per_epoch:,} iterations/epoch '
                     f'(vs ~{6482 // config.train.batch_size} for IAM)')
            self.log(f'   ℹ️  Consider fewer epochs (e.g. 5-15) for synthetic pretraining')

        elif data_mode == 'iam':
            # ----------------------------------------------------------------
            # IAM MODE (default — backward compatible, unchanged behavior)
            # All splits from IAM Aachen: train / val / test
            # ----------------------------------------------------------------
            self.log(f'📦 Data mode: IAM (Aachen splits)')

            train_set = HTRDataset(dataset_folder, 'train', fixed_size=fixed_size, transforms=selected_transforms)
            classes = train_set.character_classes

            val_set = HTRDataset(dataset_folder, 'val', fixed_size=fixed_size, transforms=None)
            test_set = HTRDataset(dataset_folder, 'test', fixed_size=fixed_size, transforms=None)
            classes_save_dir = dataset_folder

        else:
            raise ValueError(
                f"Unknown data.mode='{data_mode}'. Expected 'iam' or 'synthetic'."
            )

        # ====================================================================
        # Common path: logging, DataLoaders, character dictionaries
        # ====================================================================
        self.log('# training lines ' + str(train_set.__len__()))
        self.num_train_lines = train_set.__len__()

        self.log('# validation lines ' + str(val_set.__len__()))
        self.num_val_lines = val_set.__len__()

        self.log('# testing lines ' + str(test_set.__len__()))
        self.num_test_lines = test_set.__len__()
        self.charset_size = len(classes)
        self.log('charset size: ' + str(self.charset_size))

        # DataLoaders
        train_loader = DataLoader(train_set, batch_size=config.train.batch_size, 
                                  shuffle=True, num_workers=config.train.num_workers)
        if val_set is not None:
            val_loader = DataLoader(val_set, batch_size=config.eval.batch_size,  
                                    shuffle=False, num_workers=config.eval.num_workers)
        test_loader = DataLoader(test_set, batch_size=config.eval.batch_size,  
                                    shuffle=False, num_workers=config.eval.num_workers)

        self.loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

        # add space to classes, if not already there
        classes += ' ' 
        classes = np.unique(classes)

        # save classes to the active data folder
        np.save(os.path.join(classes_save_dir, 'classes.npy'), classes)

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

        self.log('Preparing Net - Architectural elements:')
        self.log(str(config.arch))

        classes = self.classes['classes']

        net = HTRNet(config.arch, len(classes) + 1)
        
        if config.resume is not None:
            self.log('resuming from checkpoint: {}'.format(config.resume))
            load_dict = torch.load(config.resume)
            load_status = net.load_state_dict(load_dict, strict=True)
            self.log(str(load_status))
        net.to(device)

        # print number of parameters
        n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        self.num_params = n_params
        self.log('Number of parameters: {}'.format(n_params))

        self.net = net

    def prepare_losses(self):
        self.ctc_loss = lambda y, t, ly, lt: nn.CTCLoss(reduction='sum', zero_infinity=True)(F.log_softmax(y, dim=2), t, ly, lt) /self.config.train.batch_size

    def prepare_optimizers(self):
        config = self.config
        
        # Architecture-specific optimizer configuration
        # Each architecture family has different LR, weight decay, and scheduler needs.
        arch_type = getattr(config.arch, 'type', 'cnn_rnn')
        max_epochs = config.train.num_epochs
        
        # ==================================================================
        # CNN-RNN: Original proven settings (run_32: CER 4.3%)
        # ==================================================================
        if arch_type == 'cnn_rnn':
            lr = config.train.lr  # 0.001
            weight_decay = 0.00005  # Original weight decay (run_32 setting)
            
            optimizer = torch.optim.AdamW(self.net.parameters(), lr, weight_decay=weight_decay)
            self.optimizer = optimizer
            
            # MultiStepLR at [50%, 75%] of training
            if config.train.scheduler == 'mstep':
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, [int(.5*max_epochs), int(.75*max_epochs)]
                )
            else:
                raise NotImplementedError('Alternative schedulers not implemented yet')
            
            self.log(f'CNN-RNN optimizer: lr={lr}, weight_decay={weight_decay}, MultiStepLR')
        
        # ==================================================================
        # ViT-RGTS (from scratch): Needs careful small-data training
        # Higher weight decay (DeiT uses 0.05), lower LR, longer warmup
        # ==================================================================
        elif arch_type == 'vit_rgts':
            use_cnn_stem = getattr(config.arch, 'use_cnn_stem', False)
            warmup_epochs = 5  # Shorter warmup — model needs to learn fast on small data
            
            if use_cnn_stem:
                # ---- Hybrid CNN-ViT: 3-group differential LR & weight decay ----
                # CRITICAL FIX (run_49 diagnosis):
                #   - Old WD (0.01/0.05) was 200-1000x higher than CNN baseline (5e-5)
                #   - Old LR multipliers (0.5x/0.3x) were too conservative
                #   - Head (BiLSTM) was lumped with transformer at WD=0.05 — killed RNN learning
                #
                # Group 1: CNN stem — learns local features fast, needs moderate WD
                # Group 2: Transformer — attention/FFN, moderate WD 
                # Group 3: Head (BiLSTM + projection) — needs high LR, very low WD
                stem_lr = config.train.lr           # 1e-3 (full LR for CNN stem)
                transformer_lr = config.train.lr * 0.5  # 5e-4
                head_lr = config.train.lr           # 1e-3 (head must learn fast)
                stem_wd = 0.0005       # Was 0.01 — reduced 20x
                transformer_wd = 0.005  # Was 0.05 — reduced 10x
                head_wd = 0.0001       # Very low — RNNs are sensitive to WD
                
                stem_params = list(self.net.backbone.cnn_stem.parameters()) + \
                              list(self.net.backbone.stem_pool.parameters())
                # Transformer backbone params (excluding CNN stem)
                transformer_params = [p for n, p in self.net.backbone.named_parameters()
                                       if 'cnn_stem' not in n and 'stem_pool' not in n]
                # Head params (BiLSTM + projection) — separate group
                head_params = list(self.net.top.parameters())
                
                optimizer = torch.optim.AdamW([
                    {'params': stem_params, 'lr': stem_lr, 'weight_decay': stem_wd},
                    {'params': transformer_params, 'lr': transformer_lr, 'weight_decay': transformer_wd},
                    {'params': head_params, 'lr': head_lr, 'weight_decay': head_wd},
                ])
                self.log(f'ViT-RGTS (CNN stem) optimizer:')
                self.log(f'  stem:        lr={stem_lr}, wd={stem_wd}')
                self.log(f'  transformer: lr={transformer_lr}, wd={transformer_wd}')
                self.log(f'  head:        lr={head_lr}, wd={head_wd}')
            else:
                # ---- Original from-scratch ViT: single LR ----
                lr = config.train.lr * 0.5  # 5e-4 (was 0.3x)
                weight_decay = 0.005  # Was 0.05 — reduced 10x
                optimizer = torch.optim.AdamW(self.net.parameters(), lr, weight_decay=weight_decay)
                self.log(f'ViT-RGTS optimizer: lr={lr}, weight_decay={weight_decay}')
            
            self.optimizer = optimizer
            
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs - warmup_epochs, eta_min=1e-6
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
            
            self.log(f'  warmup={warmup_epochs} epochs (start_factor=0.1) + cosine annealing')
        
        # ==================================================================
        # HTR-VT (CNN/ResNet stem + ViT + registers + span-mask, from scratch)
        # 2-group differential LR: stem+transformer vs head. Warmup + cosine.
        # ==================================================================
        elif arch_type == 'htrvt':
            warmup_epochs = 5
            backbone_lr = config.train.lr * 0.5     # 5e-4 for stem+transformer
            head_lr = config.train.lr               # 1e-3 for the CTC head
            backbone_wd = 0.005
            head_wd = 0.0001

            backbone_params = list(self.net.backbone.parameters())
            head_params = list(self.net.top.parameters())

            optimizer = torch.optim.AdamW([
                {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': backbone_wd},
                {'params': head_params, 'lr': head_lr, 'weight_decay': head_wd},
            ])
            self.optimizer = optimizer
            self.log(f'HTR-VT optimizer: backbone_lr={backbone_lr} (wd={backbone_wd}), '
                     f'head_lr={head_lr} (wd={head_wd})')

            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs - warmup_epochs, eta_min=1e-6
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
            self.log(f'  warmup={warmup_epochs} epochs (start_factor=0.1) + cosine annealing')

        # ==================================================================
        # TorchVision ViT (pretrained): DIFFERENTIAL learning rates
        # Supports two modes:
        #   1. finetune.enabled=True → LLRD + optional gradual unfreezing
        #   2. Legacy mode → simple 2-group differential LR
        # ==================================================================
        elif arch_type == 'torchvision_vit':
            finetune_enabled = (hasattr(config, 'finetune') and
                                getattr(config.finetune, 'enabled', False))

            if finetune_enabled:
                # --- LLRD Fine-tuning Mode ---
                optimizer = build_finetune_optimizer(self.net, arch_type, config)
                self.optimizer = optimizer
                self.scheduler = build_finetune_scheduler(optimizer, config, max_epochs)

                # Gradual unfreezing
                if getattr(config.finetune, 'gradual_unfreeze', False):
                    warmup_frozen = getattr(config.finetune, 'unfreeze_warmup', 3)
                    unfreeze_every = getattr(config.finetune, 'unfreeze_every', 3)
                    self.unfreezer = GradualUnfreezer(
                        self.net, arch_type,
                        warmup_frozen=warmup_frozen,
                        unfreeze_every=unfreeze_every
                    )
                    self.log(f'TorchVision ViT fine-tuning (LLRD + gradual unfreeze):')
                    self.log(f'  base_lr={config.finetune.base_lr}, head_lr={config.finetune.head_lr}')
                    self.log(f'  lr_decay_rate={config.finetune.lr_decay_rate}')
                    self.log(f'  unfreeze_warmup={warmup_frozen}, unfreeze_every={unfreeze_every}')
                else:
                    self.log(f'TorchVision ViT fine-tuning (LLRD, no gradual unfreeze):')
                    self.log(f'  base_lr={config.finetune.base_lr}, head_lr={config.finetune.head_lr}')
                    self.log(f'  lr_decay_rate={config.finetune.lr_decay_rate}')
            else:
                # --- Legacy 2-group mode ---
                backbone_lr = 2e-5
                head_lr = 5e-4
                backbone_wd = 0.01
                head_wd = 0.0001
                warmup_epochs = 5

                backbone_params = list(self.net.backbone.vit.parameters())
                head_params = list(self.net.top.parameters())
                adapter_params = list(self.net.backbone.gray_to_rgb.parameters())

                if hasattr(self.net.backbone, 'register_tokens'):
                    adapter_params += [self.net.backbone.register_tokens]

                optimizer = torch.optim.AdamW([
                    {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': backbone_wd},
                    {'params': head_params + adapter_params, 'lr': head_lr, 'weight_decay': head_wd},
                ])
                self.optimizer = optimizer

                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
                )
                cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max_epochs - warmup_epochs, eta_min=1e-6
                )
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs]
                )

                self.log(f'TorchVision ViT optimizer (legacy): backbone_lr={backbone_lr}, '
                         f'head_lr={head_lr}, warmup={warmup_epochs} epochs')
        
        # ==================================================================
        # TrOCR (pretrained): Supports three modes:
        #   1. finetune.enabled=True → LLRD + gradual unfreezing (best)
        #   2. freeze_encoder=True → Only train CTC head (fast, baseline)
        #   3. freeze_encoder=False → Simple 2-group differential LR (legacy)
        # ==================================================================
        elif arch_type == 'trocr':
            finetune_enabled = (hasattr(config, 'finetune') and
                                getattr(config.finetune, 'enabled', False))

            if finetune_enabled:
                # --- LLRD Fine-tuning Mode ---
                optimizer = build_finetune_optimizer(self.net, arch_type, config)
                self.optimizer = optimizer
                self.scheduler = build_finetune_scheduler(optimizer, config, max_epochs)

                if getattr(config.finetune, 'gradual_unfreeze', False):
                    warmup_frozen = getattr(config.finetune, 'unfreeze_warmup', 5)
                    unfreeze_every = getattr(config.finetune, 'unfreeze_every', 2)
                    self.unfreezer = GradualUnfreezer(
                        self.net, arch_type,
                        warmup_frozen=warmup_frozen,
                        unfreeze_every=unfreeze_every
                    )
                    self.log(f'TrOCR fine-tuning (LLRD + gradual unfreeze):')
                    self.log(f'  base_lr={config.finetune.base_lr}, head_lr={config.finetune.head_lr}')
                    self.log(f'  lr_decay_rate={config.finetune.lr_decay_rate}')
                    self.log(f'  unfreeze_warmup={warmup_frozen}, unfreeze_every={unfreeze_every}')
                else:
                    self.log(f'TrOCR fine-tuning (LLRD, no gradual unfreeze):')
                    self.log(f'  base_lr={config.finetune.base_lr}, head_lr={config.finetune.head_lr}')
                    self.log(f'  lr_decay_rate={config.finetune.lr_decay_rate}')
            else:
                # --- Legacy modes ---
                freeze_encoder = getattr(config.arch, 'freeze_encoder', True)
                warmup_epochs = 5

                if freeze_encoder:
                    head_lr = 1e-4
                    head_wd = 0.0001
                    trainable_params = [p for p in self.net.parameters() if p.requires_grad]
                    optimizer = torch.optim.AdamW(trainable_params, lr=head_lr, weight_decay=head_wd)
                    self.log(f'TrOCR optimizer (encoder frozen): head_lr={head_lr}, head_wd={head_wd}')
                else:
                    encoder_lr = 3e-5
                    head_lr = 5e-4
                    encoder_params = list(self.net.backbone.encoder.parameters())
                    head_params = list(self.net.top.parameters()) + list(self.net.backbone.gray_to_rgb.parameters())
                    optimizer = torch.optim.AdamW([
                        {'params': encoder_params, 'lr': encoder_lr, 'weight_decay': 0.01},
                        {'params': head_params, 'lr': head_lr, 'weight_decay': 0.0001},
                    ])
                    self.log(f'TrOCR optimizer (encoder unfrozen): encoder_lr={encoder_lr}, head_lr={head_lr}')

                self.optimizer = optimizer

                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
                )
                cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max_epochs - warmup_epochs, eta_min=1e-6
                )
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs]
                )
        
        else:
            raise ValueError(f"Unknown architecture type for optimizer: {arch_type}")

        # ======================================================================
        # SWA: Stochastic Weight Averaging (flag-gated)
        # Activated when config.swa.enabled == True (default: false).
        #
        # What it does:
        #   - Wraps self.net in an AveragedModel that maintains a running mean
        #     of the weights across epochs.
        #   - Replaces the current scheduler with SWALR starting at swa.start_epoch.
        #   - After training, BatchNorm running statistics are recalibrated on the
        #     training set (required because AveragedModel doesn't track BN stats).
        #
        # Design contract:
        #   - self.swa_enabled:   used throughout train() / test() / save()
        #   - self.swa_model:     evaluated instead of self.net in test() once active
        #   - self.swa_scheduler: stepped instead of self.scheduler in the SWA phase
        # ======================================================================
        swa_cfg = getattr(config, 'swa', None)
        self.swa_enabled = bool(getattr(swa_cfg, 'enabled', False)) if swa_cfg else False

        if self.swa_enabled:
            swa_start    = int(getattr(swa_cfg, 'start_epoch',    60))
            anneal_ep    = int(getattr(swa_cfg, 'anneal_epochs',  10))
            anneal_strat = str(getattr(swa_cfg, 'anneal_strategy', 'cos'))
            swa_lr_val   = float(getattr(swa_cfg, 'swa_lr',       5e-4))

            # AveragedModel stores the exponential moving average of net.parameters()
            self.swa_model = torch.optim.swa_utils.AveragedModel(self.net)

            # SWALR drives LR from its current value down to swa_lr over
            # anneal_epochs, then holds it constant.
            self.swa_scheduler = torch.optim.swa_utils.SWALR(
                self.optimizer,
                swa_lr=swa_lr_val,
                anneal_epochs=anneal_ep,
                anneal_strategy=anneal_strat,
            )

            self.log(
                f'SWA enabled: start_epoch={swa_start}, swa_lr={swa_lr_val}, '
                f'anneal_epochs={anneal_ep}, anneal_strategy={anneal_strat}'
            )

    def decode(self, tdec, tdict, blank_id=0):
        
        tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
        dec_transcr = ''.join([tdict[t] for t in tt if t != blank_id])
        
        return dec_transcr
                
    def sample_decoding(self):

        # get a random image from the test set
        img, transcr = self.loaders['val'].dataset[np.random.randint(0, len(self.loaders['val'].dataset))]

        img = img.unsqueeze(0).to(self.config.device)

        self.net.eval()
        with torch.no_grad():
            tst_o = self.net(img)
            # In eval mode, CTCtopB always returns a single tensor
            # No need to unpack tuple - forward() handles this internally
            if isinstance(tst_o, tuple):
                tst_o = tst_o[0]

        self.net.train()

        # Handle different output shapes
        if tst_o.dim() == 3:
            # Standard CTC output: [seq_len, batch, classes]
            tdec = tst_o.argmax(2).permute(1, 0).cpu().numpy()
        elif tst_o.dim() == 2:
            # Already 2D: [seq_len, classes] or [batch, classes]
            tdec = tst_o.argmax(1).cpu().numpy()
        else:
            raise ValueError(f"Unexpected output dimension: {tst_o.dim()}, shape: {tst_o.shape}")
        # Handle single batch case - ensure tdec is 1D array not scalar
        if tdec.ndim > 1:
            tdec = tdec.squeeze(0)
        # remove duplicates
        dec_transcr = self.decode(tdec, self.classes['i2c'])

        self.log('orig:: ' + transcr.strip())
        self.log('pred:: ' + dec_transcr.strip())


    def train(self, epoch):

        config = self.config
        device = config.device

        self.net.train()
        
        # Start epoch timer
        self.epoch_start_time = time.time()
        self.train_losses = []
        
        # Gradient accumulation for ViT models (effective batch = batch_size * accum_steps)
        # ViTs benefit from larger effective batch sizes for stable training.
        # CNN-RNN uses accum_steps=1 (no accumulation, matches run_32 behavior).
        arch_type = getattr(config.arch, 'type', 'cnn_rnn')
        accum_steps = getattr(config.train, 'gradient_accumulation', 
                              2 if arch_type in ['vit_rgts', 'htrvt', 'torchvision_vit', 'trocr'] else 1)
        effective_batch = config.train.batch_size * accum_steps
        if accum_steps > 1:
            self.log(f'Using gradient accumulation: {accum_steps} steps, effective batch size: {effective_batch}')

        t = tqdm.tqdm(self.loaders['train'], file=self.tqdm_logger if self.tqdm_logger else None)
        t.set_description('Epoch {}'.format(epoch))
        self.optimizer.zero_grad()  # Zero grad once at start
        
        for iter_idx, (img, transcr) in enumerate(t):

            img = img.to(device)

            if config.arch.head_type == "both":
                output, aux_output = self.net(img)
            else:
                output = self.net(img)

            act_lens = torch.IntTensor(img.size(0)*[output.size(0)]).to(device)
            
            # Fix: Encode each sample's labels separately, then concatenate
            # CTC expects: concatenated labels for entire batch, with label_lens tracking boundaries
            batch_labels = []
            for transcript in transcr:
                sample_labels = [self.classes['c2i'][c] for c in transcript]
                batch_labels.extend(sample_labels)
            
            labels = torch.IntTensor(batch_labels).to(device)
            label_lens = torch.IntTensor([len(transcript) for transcript in transcr]).to(device)

            loss_val = self.ctc_loss(output, labels, act_lens, label_lens)

            if config.arch.head_type == "both":
                loss_val += 0.1 * self.ctc_loss(aux_output, labels, act_lens, label_lens)

            # Scale loss by accumulation steps for correct gradient magnitude
            scaled_loss = loss_val / accum_steps
            
            tloss_val = loss_val.item()  # Log unscaled loss for readability
            self.train_losses.append(tloss_val)
        
            scaled_loss.backward()
            
            # Step optimizer every accum_steps iterations (or at end of epoch)
            if (iter_idx + 1) % accum_steps == 0 or (iter_idx + 1) == len(self.loaders['train']):
                # Gradient clipping for ViT training stability only
                # CNN-RNN does not need gradient clipping (matches run_32 behavior)
                if arch_type in ['vit_rgts', 'htrvt', 'torchvision_vit', 'trocr']:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=10.0)
                
                self.optimizer.step()
                self.optimizer.zero_grad()

            t.set_postfix(values='loss : {:.2f}'.format(tloss_val))

        self.sample_decoding()

        # SWA: update the averaged model after every epoch in the SWA phase.
        # The caller is responsible for passing the current epoch so we can
        # decide whether averaging should start. We do that in the main loop
        # below; here we expose a helper that the main loop calls.

    def swa_update(self, epoch):
        """
        Update AveragedModel weights for the current epoch (SWA phase only).
        Called from the main training loop after htr_trainer.train(epoch).

        Does nothing when SWA is disabled or before swa.start_epoch.
        """
        if not self.swa_enabled:
            return
        swa_start = int(getattr(getattr(self.config, 'swa', None), 'start_epoch', 60))
        if epoch >= swa_start:
            self.swa_model.update_parameters(self.net)
            self.log(f'  [SWA] weights averaged at epoch {epoch}')
    
    def extract_attention_weights(self, epoch, dataset='val', num_samples=5):
        """Extract and save attention weights from ViT models during training."""
        
        config = self.config
        device = config.device
        
        # Only extract attention for ViT architectures
        arch_type = getattr(config.arch, 'type', 'cnn_rnn')
        if arch_type not in ['vit_rgts', 'htrvt', 'torchvision_vit', 'trocr']:
            return  # Skip for CNN-RNN
        
        if self.attention_dir is None:
            return  # No attention directory setup
        
        self.log(f'Extracting attention weights for {arch_type} at epoch {epoch}...')
        
        # Get loader
        if dataset == 'val':
            loader = self.loaders['val']
        elif dataset == 'test':
            loader = self.loaders['test']
        else:
            loader = self.loaders['train']
        
        self.net.eval()
        
        # Create epoch-specific directory
        epoch_attn_dir = os.path.join(self.attention_dir, f'epoch_{epoch:03d}')
        os.makedirs(epoch_attn_dir, exist_ok=True)
        
        sample_count = 0
        attention_stats = {
            'num_layers': 0,
            'num_heads': 0,
            'avg_attention_entropy': [],
            'avg_token_norms': []
        }
        
        for batch_idx, (imgs, transcrs) in enumerate(loader):
            if sample_count >= num_samples:
                break
            
            imgs = imgs.to(device)
            batch_size = imgs.size(0)
            
            with torch.no_grad():
                # Use forward_explain to get attention maps
                # HTRNet.forward_explain returns: (logits, reg_tokens, attn_maps, token_norms, grid)
                # For trocr, reg_tokens is None
                try:
                    logits, reg_tokens, attn_maps, token_norms, grid = self.net.forward_explain(imgs)
                except Exception as e:
                    self.log(f'Warning: Could not extract attention: {str(e)}')
                    return
            
            # Save attention maps for each sample in batch
            for sample_idx in range(min(batch_size, num_samples - sample_count)):
                sample_id = sample_count + sample_idx
                
                # Save attention maps (one file per layer)
                for layer_idx, attn in enumerate(attn_maps):
                    # attn shape: [B, H, S, S]
                    attn_sample = attn[sample_idx].cpu().numpy()  # [H, S, S]
                    
                    filename = f'sample_{sample_id:03d}_layer_{layer_idx:02d}.npy'
                    filepath = os.path.join(epoch_attn_dir, filename)
                    np.save(filepath, attn_sample)
                
                # Save token norms
                if token_norms is not None:
                    norms_sample = token_norms[sample_idx].cpu().numpy()
                    filename = f'sample_{sample_id:03d}_token_norms.npy'
                    filepath = os.path.join(epoch_attn_dir, filename)
                    np.save(filepath, norms_sample)
                
                # Save register tokens if available
                if reg_tokens is not None:
                    reg_sample = reg_tokens[sample_idx].cpu().numpy()
                    filename = f'sample_{sample_id:03d}_register_tokens.npy'
                    filepath = os.path.join(epoch_attn_dir, filename)
                    np.save(filepath, reg_sample)
                
                # Save ground truth
                gt_filename = f'sample_{sample_id:03d}_groundtruth.txt'
                gt_filepath = os.path.join(epoch_attn_dir, gt_filename)
                with open(gt_filepath, 'w') as f:
                    f.write(transcrs[sample_idx])
            
            # Update stats
            if len(attn_maps) > 0:
                attention_stats['num_layers'] = len(attn_maps)
                attention_stats['num_heads'] = attn_maps[0].shape[1]
                
                # Calculate attention entropy (measure of focus)
                for attn in attn_maps:
                    # Average over batch, heads, and source positions
                    attn_probs = attn.mean(dim=(0, 1, 2))  # [S]
                    entropy = -(attn_probs * torch.log(attn_probs + 1e-10)).sum().item()
                    attention_stats['avg_attention_entropy'].append(entropy)
                
                # Average token norms
                if token_norms is not None:
                    avg_norm = token_norms.mean().item()
                    attention_stats['avg_token_norms'].append(avg_norm)
            
            sample_count += batch_size
        
        # Save metadata
        metadata = {
            'epoch': epoch,
            'architecture': arch_type,
            'num_samples': sample_count,
            'num_layers': attention_stats['num_layers'],
            'num_heads': attention_stats['num_heads'],
            'grid_size': grid if 'grid' in locals() else None,
            'avg_attention_entropy': np.mean(attention_stats['avg_attention_entropy']) if attention_stats['avg_attention_entropy'] else 0,
            'avg_token_norm': np.mean(attention_stats['avg_token_norms']) if attention_stats['avg_token_norms'] else 0
        }
        
        import json
        metadata_path = os.path.join(epoch_attn_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        self.log(f'Saved attention weights to: {epoch_attn_dir}')
        self.log(f'  Layers: {metadata["num_layers"]}, Heads: {metadata["num_heads"]}, Samples: {metadata["num_samples"]}')
        
        self.net.train()

    def swa_finalize(self):
        """
        Recalibrate BatchNorm running statistics for the SWA-averaged model.

        AveragedModel copies weights but does NOT track BatchNorm running_mean /
        running_var during the averaging process. A single forward pass over the
        training set is required to fix this before the SWA model can be evaluated.

        Called once after the last training epoch (or at the end of the main loop).
        Does nothing when SWA is disabled.
        """
        if not self.swa_enabled or self.swa_model is None:
            return

        self.log('  [SWA] Updating BatchNorm statistics (forward pass over train set)...')
        swa_cfg = getattr(self.config, 'swa', None)
        bn_steps = int(getattr(swa_cfg, 'update_bn_steps', 0)) if swa_cfg else 0

        self.swa_model.eval()
        device = self.config.device

        # torch.optim.swa_utils.update_bn handles the BN forward passes.
        # It expects an iterable that yields batches of *inputs only*.
        # We wrap our train loader to strip the label from each tuple.
        input_loader = (imgs.to(device) for imgs, _ in self.loaders['train'])
        torch.optim.swa_utils.update_bn(input_loader, self.swa_model, device=device)

        self.log('  [SWA] BatchNorm statistics updated.')

    def test(self, epoch, tset='test', use_swa=None):
        """
        Evaluate on val or test split.

        use_swa:
          None  (default) — auto: use SWA model when swa_enabled AND swa_model is
                            ready (i.e. swa.start_epoch has been reached).
          True  — force evaluation on the SWA-averaged model.
          False — force evaluation on the base model (self.net).
        """
        config = self.config
        device = config.device

        # Decide which model to evaluate.
        swa_start = int(getattr(getattr(config, 'swa', None), 'start_epoch', 9999))
        _auto_swa = (
            self.swa_enabled
            and self.swa_model is not None
            and epoch >= swa_start
        )
        run_on_swa = _auto_swa if use_swa is None else bool(use_swa)
        eval_model = self.swa_model if run_on_swa else self.net
        eval_model.eval()

        if tset == 'test':
            loader = self.loaders['test']
        elif tset == 'val':
            loader = self.loaders['val']
        else:
            print("not recognized set in test function")

        swa_tag = ' [SWA]' if run_on_swa else ''
        self.log(f'####################### Evaluating {tset} set at epoch {epoch}{swa_tag} #######################')

        cer, wer = CER(), WER(mode=config.eval.wer_mode)
        sample_idx = 0

        for (imgs, transcrs) in tqdm.tqdm(loader, file=self.tqdm_logger if self.tqdm_logger else None):

            imgs = imgs.to(device)
            with torch.no_grad():
                o = eval_model(imgs)
            # In eval mode, CTCtopB returns single tensor
            # But handle tuple case defensively
            if isinstance(o, tuple):
                o = o[0]
            
            tdecs = o.argmax(2).permute(1, 0).cpu().numpy()
            
            # Handle batch dimension properly
            if tdecs.ndim == 1:
                tdecs = tdecs.reshape(1, -1)

            for tdec, transcr in zip(tdecs, transcrs):
                transcr = transcr.strip()
                dec_transcr = self.decode(tdec, self.classes['i2c']).strip()

                # Calculate per-sample metrics
                sample_cer_metric = CER()
                sample_wer_metric = WER(mode=config.eval.wer_mode)
                sample_cer_metric.update(dec_transcr, transcr)
                sample_wer_metric.update(dec_transcr, transcr)
                sample_cer_score = sample_cer_metric.score()
                sample_wer_score = sample_wer_metric.score()
                
                # Log to detailed evaluation CSV
                if self.eval_csv_writer is not None:
                    self.eval_csv_writer.writerow([
                        epoch, tset, sample_idx, transcr, dec_transcr,
                        sample_cer_score, sample_wer_score,
                        len(transcr), len(dec_transcr)
                    ])
                
                sample_idx += 1
                
                cer.update(dec_transcr, transcr)
                wer.update(dec_transcr, transcr)
        
        # Flush the evaluation CSV after each epoch
        if self.eval_csv_file is not None:
            self.eval_csv_file.flush()
        
        cer_score = cer.score()
        wer_score = wer.score()

        self.log('CER at epoch {}{}: {:.3f}'.format(epoch, swa_tag, cer_score))
        self.log('WER at epoch {}{}: {:.3f}'.format(epoch, swa_tag, wer_score))

        # Restore training mode on the base model (SWA model stays in eval)
        self.net.train()

        return cer_score, wer_score

    def save(self, epoch):
        """
        Save model checkpoint(s) to the experiment directory.

        Normal mode:  saves self.net weights as  model.pt
        SWA mode:     saves self.net weights as  model.pt  (current base model)
                      AND saves SWA-averaged weights as  model_swa.pt
                      once swa.start_epoch has been reached.
        """
        self.log('####################### Saving model at epoch {} #######################'.format(epoch))

        if self.experiment_dir is None:
            if not os.path.exists('./saved_models'):
                os.makedirs('./saved_models')
            save_path = './saved_models/htrnet_{}.pt'.format(epoch)
        else:
            save_path = os.path.join(self.experiment_dir, 'model.pt')

        # Always save the base model
        torch.save(self.net.cpu().state_dict(), save_path)
        self.net.to(self.config.device)
        self.log(f'Model saved to: {save_path}')

        # Also save the SWA-averaged model once averaging has started
        if self.swa_enabled and self.swa_model is not None:
            swa_start = int(getattr(getattr(self.config, 'swa', None), 'start_epoch', 9999))
            if epoch >= swa_start:
                swa_save_path = os.path.join(
                    self.experiment_dir if self.experiment_dir else './saved_models',
                    'model_swa.pt'
                )
                # AveragedModel wraps the original module; extract the plain state_dict
                # from the underlying module so it can be loaded back into HTRNet directly.
                swa_state = {
                    k.replace('module.', ''): v.cpu()
                    for k, v in self.swa_model.module.state_dict().items()
                }
                torch.save(swa_state, swa_save_path)
                self.swa_model.to(self.config.device)
                self.log(f'SWA model saved to: {swa_save_path}')


def parse_args():
    # Load base config
    conf = OmegaConf.load(sys.argv[1])
    
    # Separate YAML files from CLI overrides
    yaml_files = []
    cli_overrides = []
    
    for arg in sys.argv[2:]:
        if arg.endswith('.yaml'):
            yaml_files.append(arg)
        elif '=' in arg:
            cli_overrides.append(arg)
        else:
            print(f"Warning: Ignoring unrecognized argument: {arg}")
    
    # Load additional YAML config files
    for config_file in yaml_files:
        additional_conf = OmegaConf.load(config_file)
        conf = OmegaConf.merge(conf, additional_conf)
    
    # Apply CLI overrides (e.g., arch.num_registers=0)
    if cli_overrides:
        cli_conf = OmegaConf.from_dotlist(cli_overrides)
        conf = OmegaConf.merge(conf, cli_conf)

    OmegaConf.set_struct(conf, True)
    return conf


if __name__ == '__main__':
    # ----------------------- initialize configuration ----------------------- #
    config = parse_args()
    max_epochs = config.train.num_epochs

    # Set seed for reproducibility (default: 42)
    seed_value = config.get('seed', 42)
    if seed_value >= 0:
        set_seed(seed_value)

    # Setup experiment directory structure
    experiment_dir, log_file, run_number, csv_file, csv_writer = setup_experiment_dir(config)
    
    def log_print(msg):
        """Helper function to print and log messages"""
        print(msg)
        if log_file:
            log_file.write(msg + '\n')
            log_file.flush()
            
    log_print(f"\n{'='*80}")
    log_print(f"Starting Experiment: run_{run_number}")
    log_print(f"Experiment Directory: {experiment_dir}")
    log_print(f"{'='*80}\n")

    htr_trainer = HTRTrainer(config, experiment_dir, log_file, run_number, csv_file, csv_writer)

    cnt = 1
    htr_trainer.log('Training Started!')
    cer_score, wer_score = htr_trainer.test(0, 'test')
    
    # Track best metrics
    best_cer = cer_score
    best_epoch = 0
    
    for epoch in range(1, max_epochs + 1):

        # Gradual unfreezing step (if enabled)
        if htr_trainer.unfreezer is not None:
            n_unfrozen = htr_trainer.unfreezer.step(epoch)
            if epoch <= 5 or epoch % 3 == 0:
                htr_trainer.log(f'  [Unfreeze] epoch {epoch}: {n_unfrozen} backbone layers unfrozen')

        htr_trainer.train(epoch)

        # ── LR scheduler step ───────────────────────────────────────────────
        # During the SWA phase (epoch >= swa.start_epoch) we switch from the
        # base scheduler to SWALR.  Before that, we advance the base scheduler
        # as usual.
        swa_cfg = getattr(config, 'swa', None)
        swa_start = int(getattr(swa_cfg, 'start_epoch', 9999)) if swa_cfg else 9999
        if htr_trainer.swa_enabled and epoch >= swa_start:
            # SWA phase: step SWALR + accumulate averaged weights
            htr_trainer.swa_scheduler.step()
            htr_trainer.swa_update(epoch)
        else:
            htr_trainer.scheduler.step()

        # save and evaluate the current model
        if epoch % config.train.save_every_k_epochs == 0:
            htr_trainer.save(epoch)
            val_cer, val_wer = htr_trainer.test(epoch, 'val')
            test_cer, test_wer = htr_trainer.test(epoch, 'test')

            # Extract attention weights for ViT models every 5 epochs
            arch_type = getattr(config.arch, 'type', 'cnn_rnn')
            if arch_type in ['vit_rgts', 'torchvision_vit', 'trocr']:
                if epoch % 5 == 0 or epoch == 1:
                    htr_trainer.extract_attention_weights(epoch, dataset='val', num_samples=5)

            # Track best model
            if val_cer < best_cer:
                best_cer = val_cer
                best_epoch = epoch
                htr_trainer.log(f'\n*** New best CER: {best_cer:.3f} at epoch {best_epoch} ***\n')

            # Write CSV row with all metrics
            epoch_time = time.time() - htr_trainer.epoch_start_time
            avg_train_loss = sum(htr_trainer.train_losses) / len(htr_trainer.train_losses) if htr_trainer.train_losses else 0.0
            current_lr = htr_trainer.optimizer.param_groups[0]['lr']
            seed_value = config.get('seed', -1)

            csv_writer.writerow([
                epoch,
                current_lr,
                avg_train_loss,
                val_cer,
                val_wer,
                test_cer,
                test_wer,
                htr_trainer.num_train_lines,
                htr_trainer.num_val_lines,
                htr_trainer.num_test_lines,
                htr_trainer.charset_size,
                htr_trainer.charset_size,
                htr_trainer.charset_size,
                htr_trainer.num_params,
                epoch_time,
                config.device,
                seed_value
            ])
            csv_file.flush()

    # ── Post-training SWA finalization ───────────────────────────────────────
    # After all epochs, recalibrate BatchNorm stats for the averaged model and
    # run a final evaluation so the SWA CER is captured in the log.
    if htr_trainer.swa_enabled:
        htr_trainer.log('\n' + '='*80)
        htr_trainer.log('[SWA] Finalizing: recalibrating BatchNorm statistics...')
        htr_trainer.swa_finalize()

        htr_trainer.log('[SWA] Final evaluation of SWA model:')
        swa_val_cer,  swa_val_wer  = htr_trainer.test(max_epochs, 'val',  use_swa=True)
        swa_test_cer, swa_test_wer = htr_trainer.test(max_epochs, 'test', use_swa=True)
        htr_trainer.log(f'[SWA] Val  CER: {swa_val_cer:.3f}  WER: {swa_val_wer:.3f}')
        htr_trainer.log(f'[SWA] Test CER: {swa_test_cer:.3f}  WER: {swa_test_wer:.3f}')

        if swa_val_cer < best_cer:
            best_cer = swa_val_cer
            best_epoch = max_epochs
            htr_trainer.log(f'[SWA] New best CER: {best_cer:.3f} (SWA model)')
        htr_trainer.log('='*80)

    # Save final summary
    htr_trainer.log("\n" + "="*80)
    htr_trainer.log("Training Completed!")
    htr_trainer.log(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    htr_trainer.log(f"Best Validation CER: {best_cer:.3f} at epoch {best_epoch}")
    htr_trainer.log(f"Final model saved to: {os.path.join(experiment_dir, 'model.pt')}")
    if htr_trainer.swa_enabled:
        htr_trainer.log(f"SWA model saved to:   {os.path.join(experiment_dir, 'model_swa.pt')}")
    htr_trainer.log("="*80)

    # Log evaluation details path before closing files
    if htr_trainer.eval_csv_file:
        htr_trainer.log(f"Detailed evaluation saved to: {htr_trainer.eval_csv_path}")

    # Print final summary before closing files
    log_print(f"\n{'='*80}")
    log_print(f"Experiment run_{run_number} completed!")
    log_print(f"Results saved in: {experiment_dir}")
    log_print(f"Detailed evaluation CSV: {os.path.join(experiment_dir, 'evaluation_details.csv')}")
    log_print(f"{'='*80}\n")

    # Now close all files
    if log_file:
        log_file.close()

    if csv_file:
        csv_file.close()

    if htr_trainer.eval_csv_file:
        htr_trainer.eval_csv_file.close()
    # Final model is already saved into the experiment directory as `model.pt`.
    # Remove legacy/global save to `saved_models/<config.save>` to avoid creating
    # unexpected files like `saved_models/temp.pt`.
    