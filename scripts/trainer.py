import argparse
from omegaconf import OmegaConf

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.htr_dataset import HTRDataset

from models import HTRNet
from utils.transforms import aug_transforms

import torch.nn.functional as F

from utils.metrics import CER, WER
import json
import shutil
from datetime import datetime
import csv
import time


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
    """
    base_dir = './saved_models'
    experiments_dir = os.path.join(base_dir, 'experiments')
    
    # Get next run number
    run_number = get_next_run_number(experiments_dir)
    run_dir = os.path.join(experiments_dir, f'run_{run_number}')
    
    # Create directory structure
    os.makedirs(run_dir, exist_ok=True)
    
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
        
        # Create tqdm logger
        self.tqdm_logger = TqdmToLogFile(log_file) if log_file else None
        
        # Tracking variables for CSV
        self.train_losses = []
        self.epoch_start_time = None
        self.num_params = 0
        
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


    def prepare_dataloaders(self):

        config = self.config

        # prepare datset loader
        dataset_folder = config.data.path
        fixed_size = (config.preproc.image_height, config.preproc.image_width)

        train_set = HTRDataset(dataset_folder, 'train', fixed_size=fixed_size, transforms=aug_transforms)
        classes = train_set.character_classes
        self.log('# training lines ' + str(train_set.__len__()))
        self.num_train_lines = train_set.__len__()

        val_set = HTRDataset(dataset_folder, 'val', fixed_size=fixed_size, transforms=None)
        self.log('# validation lines ' + str(val_set.__len__()))
        self.num_val_lines = val_set.__len__()

        test_set = HTRDataset(dataset_folder, 'test', fixed_size=fixed_size, transforms=None)
        self.log('# testing lines ' + str(test_set.__len__()))
        self.num_test_lines = test_set.__len__()
        self.charset_size = len(classes)
        self.log('charset size: ' + str(self.charset_size))

        # augmentation using data sampler
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

        # save classes in data folder
        np.save(os.path.join(dataset_folder, 'classes.npy'), classes)

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
        optimizer = torch.optim.AdamW(self.net.parameters(), config.train.lr, weight_decay=0.00005)

        self.optimizer = optimizer

        max_epochs = config.train.num_epochs
        if config.train.scheduler == 'mstep':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(.5*max_epochs), int(.75*max_epochs)])
        else:
            raise NotImplementedError('Alternative schedulers not implemented yet')

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
            if self.config.arch.head_type == 'both':
                tst_o = tst_o[0]

        self.net.train()

        tdec = tst_o.argmax(2).permute(1, 0).cpu().numpy()
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

        t = tqdm.tqdm(self.loaders['train'], file=self.tqdm_logger if self.tqdm_logger else None)
        t.set_description('Epoch {}'.format(epoch))
        for iter_idx, (img, transcr) in enumerate(t):
            self.optimizer.zero_grad()

            img = img.to(device)

            if config.arch.head_type == "both":
                output, aux_output = self.net(img)
            else:
                output = self.net(img)

            act_lens = torch.IntTensor(img.size(0)*[output.size(0)]).to(device)
            
            # Fix: Encode each sample's labels separately, then concatenate
            # CTC expects: concatenated labels for entire batch, with label_lens tracking boundaries
            batch_labels = []
            for t in transcr:
                sample_labels = [self.classes['c2i'][c] for c in t]
                batch_labels.extend(sample_labels)
            
            labels = torch.IntTensor(batch_labels).to(device)
            label_lens = torch.IntTensor([len(t) for t in transcr]).to(device)

            loss_val = self.ctc_loss(output, labels, act_lens, label_lens)

            if config.arch.head_type == "both":
                loss_val += 0.1 * self.ctc_loss(aux_output, labels, act_lens, label_lens)

            tloss_val = loss_val.item()
            self.train_losses.append(tloss_val)
        
            loss_val.backward()
            self.optimizer.step()    

            t.set_postfix(values='loss : {:.2f}'.format(tloss_val))

        self.sample_decoding()
    
    def test(self, epoch, tset='test'):

        config = self.config
        device = config.device

        self.net.eval()

        if tset=='test':
            loader = self.loaders['test']
        elif tset=='val':
            loader = self.loaders['val']
        else:
            print("not recognized set in test function")

        self.log('####################### Evaluating {} set at epoch {} #######################'.format(tset, epoch))
        
        cer, wer = CER(), WER(mode=config.eval.wer_mode)
        sample_idx = 0
        
        for (imgs, transcrs) in tqdm.tqdm(loader, file=self.tqdm_logger if self.tqdm_logger else None):

            imgs = imgs.to(device)
            with torch.no_grad():
                o = self.net(imgs)
            # if o tuple keep only the first element
            if config.arch.head_type == 'both':
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

        self.log('CER at epoch {}: {:.3f}'.format(epoch, cer_score))
        self.log('WER at epoch {}: {:.3f}'.format(epoch, wer_score))

        self.net.train()
        
        return cer_score, wer_score

    def save(self, epoch):
        """Save model to experiment directory."""
        self.log('####################### Saving model at epoch {} #######################'.format(epoch))
        
        if self.experiment_dir is None:
            # Fallback to old behavior if no experiment directory is set
            if not os.path.exists('./saved_models'):
                os.makedirs('./saved_models')
            save_path = './saved_models/htrnet_{}.pt'.format(epoch)
        else:
            # Save to experiment directory as model.pt
            save_path = os.path.join(self.experiment_dir, 'model.pt')
        
        torch.save(self.net.cpu().state_dict(), save_path)
        self.net.to(self.config.device)
        self.log(f'Model saved to: {save_path}')


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

        htr_trainer.train(epoch)
        htr_trainer.scheduler.step()

        # save and evaluate the current model
        if epoch % config.train.save_every_k_epochs == 0:
            htr_trainer.save(epoch)
            val_cer, val_wer = htr_trainer.test(epoch, 'val')
            test_cer, test_wer = htr_trainer.test(epoch, 'test')
            
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

    # Save final summary
    htr_trainer.log("\n" + "="*80)
    htr_trainer.log("Training Completed!")
    htr_trainer.log(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    htr_trainer.log(f"Best Validation CER: {best_cer:.3f} at epoch {best_epoch}")
    htr_trainer.log(f"Final model saved to: {os.path.join(experiment_dir, 'model.pt')}")
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
    