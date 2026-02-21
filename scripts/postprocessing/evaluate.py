#!/usr/bin/env python3
"""
Evaluate a Trained HTR Model on Validation/Test Sets

Re-runs CER/WER evaluation using a saved model checkpoint and config.
Produces per-sample predictions and error metrics.

Supported Architectures: ALL (cnn_rnn, vit_rgts, torchvision_vit, trocr)
Input: config YAML + model checkpoint (model.pt)
Output: Terminal CER/WER summary, evaluation_summary JSON and evaluation_report TXT in run folder

what it does:
    - Loads model checkpoint and config
    - Prepares dataloaders for val/test sets
    - Runs inference on each sample
    - Computes CER/WER per sample and overall
    - Saves summary JSON and human-readable report to the run folder

Usage:
    # Evaluate CNN-RNN model (run_32)
    python scripts/postprocessing/evaluate.py configs/config.yaml configs/baseline.yaml \
        resume=saved_models/experiments/run_32/model.pt

    # Evaluate ViT-RGTS model (run_54)
    python scripts/postprocessing/evaluate.py configs/config.yaml configs/baseline.yaml \
        configs/baseline_vit_rgts_v2.yaml \
        resume=saved_models/experiments/run_54/model.pt

    # Evaluate TorchVision ViT (run_39)
    python scripts/postprocessing/evaluate.py configs/config.yaml configs/baseline.yaml \
        configs/torchvision_vit.yaml \
        resume=saved_models/experiments/run_39/model.pt

    # Evaluate TrOCR (run_40)
    python scripts/postprocessing/evaluate.py configs/config.yaml configs/baseline.yaml \
        configs/trocr.yaml \
        resume=saved_models/experiments/run_40/model.pt

    # Evaluate on test set only
    python scripts/postprocessing/evaluate.py configs/config.yaml configs/baseline.yaml \
        resume=saved_models/experiments/run_32/model.pt eval_sets=test

    # Custom output directory
    python scripts/postprocessing/evaluate.py configs/config.yaml configs/baseline.yaml \
        configs/baseline_vit_rgts_v2.yaml \
        resume=saved_models/experiments/run_54/model.pt \
        output_dir=output/evaluation/run_54

Arguments:
    config.yaml      Base config file (required)
    extra.yaml        Additional config files merged in order (optional)
    key=value         Override any config parameter (e.g., resume=..., device=cuda)
    eval_sets         Which sets to evaluate: val, test, or both (default: both)
"""

import json
from datetime import datetime
import argparse
from omegaconf import OmegaConf

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.htr_dataset import HTRDataset

from models import HTRNet
from utils.metrics import CER, WER

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
        fixed_size = (config.preproc.image_height, config.preproc.image_width)

        val_set = HTRDataset(dataset_folder, 'val', fixed_size=fixed_size, transforms=None)
        print('# validation lines ' + str(val_set.__len__()))

        test_set = HTRDataset(dataset_folder, 'test', fixed_size=fixed_size, transforms=None)
        print('# testing lines ' + str(test_set.__len__()))

        # load classes from the training set saved in the data folder
        classes = np.load(os.path.join(dataset_folder, 'classes.npy'))

        # Optimize batch_size and num_workers for CPU
        device = config.device
        batch_size = config.eval.batch_size
        num_workers = config.eval.num_workers
        
        # CPU optimization: reduce batch size and disable multiprocessing
        if device == 'cpu':
            batch_size = 1  # Reduce to 1 for CPU to avoid memory issues
            num_workers = 0  # Disable multiprocessing on CPU (faster + less memory)
            print(f'🔧 CPU mode: Reduced batch_size to {batch_size}, num_workers to {num_workers}')
        
        val_loader = DataLoader(val_set, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers, pin_memory=False)

        test_loader = DataLoader(test_set, batch_size=batch_size,  
                                    shuffle=False, num_workers=num_workers, pin_memory=False)

        self.loaders = {'val': val_loader, 'test': test_loader}

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
            print("⚠️ CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
            self.config.device = "cpu"

        print('Preparing Net - Architectural elements:')
        print(config.arch)

        classes = self.classes['classes']
        net = HTRNet(config.arch, len(classes) + 1)

        if config.resume is not None:
            print(f'resuming from checkpoint: {config.resume}')
            load_dict = torch.load(config.resume, map_location=device)
            print(net.load_state_dict(load_dict, strict=True))

        net.to(device)
        self.net = net

    def decode(self, tdec, tdict, blank_id=0):
        
        tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
        dec_transcr = ''.join([tdict[t] for t in tt if t != blank_id])
        
        return dec_transcr
    
    def test(self, epoch, tset='test', output_dir=None):

        config = self.config
        device = config.device

        self.net.eval()

        if tset=='test':
            loader = self.loaders['test']
        elif tset=='val':
            loader = self.loaders['val']
        else:
            print("not recognized set in test function")

        print('####################### Evaluating {} set at epoch {} #######################'.format(tset, epoch))
        
        cer, wer = CER(), WER(mode=config.eval.wer_mode)
        
        # Collect per-sample results for CSV output
        sample_results = []
        
        import gc
        
        total_batches = len(loader)
        imgs, tdecs = None, None  # ensure defined for cleanup after loop
        
        for batch_idx, (imgs, transcrs) in enumerate(tqdm.tqdm(loader, desc=f'Evaluating {tset}')):

            imgs = imgs.to(device)
            with torch.no_grad():
                o = self.net(imgs)
            
            # Move to CPU immediately and clear GPU cache
            tdecs = o.argmax(2).permute(1, 0).cpu().numpy()
            del o  # Delete output immediately
            
            # Handle batch dimension properly
            if tdecs.ndim == 1:
                tdecs = tdecs.reshape(1, -1)

            for tdec, transcr in zip(tdecs, transcrs):
                transcr = transcr.strip()
                dec_transcr = self.decode(tdec, self.classes['i2c']).strip()

                cer.update(dec_transcr, transcr)
                wer.update(dec_transcr, transcr)
                
                # Per-sample CER/WER
                sample_cer = CER()
                sample_wer = WER(mode=config.eval.wer_mode)
                sample_cer.update(dec_transcr, transcr)
                sample_wer.update(dec_transcr, transcr)
                sample_results.append({
                    'ground_truth': transcr,
                    'prediction': dec_transcr,
                    'cer': round(sample_cer.score(), 4),
                    'wer': round(sample_wer.score(), 4),
                    'gt_length': len(transcr),
                    'correct': transcr == dec_transcr,
                })
            
            # Aggressive memory cleanup for CPU (every 10 batches)
            if device == 'cpu' and batch_idx % 10 == 0:
                gc.collect()  # imgs/tdecs reassigned each iter; just trigger GC
            
            # Progress checkpoint every 100 batches
            if batch_idx > 0 and batch_idx % 100 == 0:
                print(f'\n[Checkpoint {batch_idx}/{total_batches}] Current CER: {cer.score():.4f}, WER: {wer.score():.4f}')
        
        # Final cleanup
        if imgs is not None:
            del imgs
        if tdecs is not None:
            del tdecs
        gc.collect()
        
        cer_score = cer.score()
        wer_score = wer.score()

        print('CER at epoch {}: {:.3f}'.format(epoch, cer_score))
        print('WER at epoch {}: {:.3f}'.format(epoch, wer_score))

        # ── Save outputs ────────────────────────────────────────────────────
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # 1) Summary metrics JSON
            exact_matches = sum(1 for r in sample_results if r['correct'])
            summary = {
                'timestamp': datetime.now().isoformat(),
                'set': tset,
                'epoch': epoch,
                'num_samples': len(sample_results),
                'cer': round(cer_score, 4),
                'wer': round(wer_score, 4),
                'exact_match_rate': round(exact_matches / len(sample_results), 4) if sample_results else 0.0,
                'exact_matches': exact_matches,
            }
            json_path = os.path.join(output_dir, f'evaluation_summary_{tset}.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            print(f'Summary saved:           {json_path}')

            # 2) Human-readable report
            report_lines = [
                '=' * 60,
                f'EVALUATION REPORT  [{tset.upper()} SET]',
                '=' * 60,
                f'Timestamp     : {summary["timestamp"]}',
                f'Set           : {tset}',
                f'Samples       : {summary["num_samples"]}',
                '',
                f'CER           : {cer_score:.4f}  ({cer_score*100:.2f}%)',
                f'WER           : {wer_score:.4f}  ({wer_score*100:.2f}%)',
                f'Exact Match   : {exact_matches}/{summary["num_samples"]}'
                f'  ({summary["exact_match_rate"]*100:.1f}%)',
                '',
                '--- Top 10 Worst Samples (by CER) ---',
            ]
            worst = sorted(sample_results, key=lambda x: x['cer'], reverse=True)[:10]
            for i, r in enumerate(worst, 1):
                report_lines.append(
                    f'{i:>2}. CER={r["cer"]:.3f}  GT="{r["ground_truth"]}"  '
                    f'PRED="{r["prediction"]}"'
                )
            report_lines += ['', '--- Top 10 Best Samples (by CER, excluding perfect) ---']
            best = sorted([r for r in sample_results if r['cer'] > 0],
                          key=lambda x: x['cer'])[:10]
            for i, r in enumerate(best, 1):
                report_lines.append(
                    f'{i:>2}. CER={r["cer"]:.3f}  GT="{r["ground_truth"]}"  '
                    f'PRED="{r["prediction"]}"'
                )
            report_lines.append('=' * 60)

            txt_path = os.path.join(output_dir, f'evaluation_report_{tset}.txt')
            report_text = '\n'.join(report_lines)
            print('\n' + report_text)
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(report_text + '\n')
            print(f'\nReport saved:            {txt_path}')

        self.net.train()


def parse_args():
    # Support multiple YAML files: python evaluate.py base.yaml extra.yaml key=value
    yaml_files = [a for a in sys.argv[1:] if a.endswith('.yaml')]
    overrides  = [a for a in sys.argv[1:] if not a.endswith('.yaml')]

    if not yaml_files:
        print('Error: at least one .yaml config file is required.')
        sys.exit(1)

    conf = OmegaConf.load(yaml_files[0])
    for yf in yaml_files[1:]:
        conf = OmegaConf.merge(conf, OmegaConf.load(yf))

    OmegaConf.set_struct(conf, False)
    cli_conf = OmegaConf.from_dotlist(overrides)
    conf = OmegaConf.merge(conf, cli_conf)
    
    # Add eval_sets parameter if not specified (default: both)
    if not hasattr(conf, 'eval_sets'):
        conf.eval_sets = 'both'  # Options: 'val', 'test', 'both'

    # Derive default output_dir from resume path (save in the run folder)
    if not hasattr(conf, 'output_dir') or not conf.output_dir:
        if hasattr(conf, 'resume') and conf.resume:
            conf.output_dir = os.path.dirname(os.path.abspath(conf.resume))
        else:
            conf.output_dir = os.path.join('output', 'evaluation', 'unknown_run')

    return conf


if __name__ == '__main__':
    # ----------------------- initialize configuration ----------------------- #
    config = parse_args()
    max_epochs = config.train.num_epochs

    output_dir = config.output_dir
    print(f'Output directory: {output_dir}')

    htr_eval = HTREval(config)

    # Determine which datasets to evaluate
    eval_sets = getattr(config, 'eval_sets', 'both')
    
    if eval_sets in ['val', 'both']:
        print('\n' + '='*80)
        print('EVALUATING VALIDATION SET')
        print('='*80)
        htr_eval.test(0, 'val', output_dir=output_dir)
    
    if eval_sets in ['test', 'both']:
        print('\n' + '='*80)
        print('EVALUATING TEST SET')
        print('='*80)
        htr_eval.test(0, 'test', output_dir=output_dir)
    