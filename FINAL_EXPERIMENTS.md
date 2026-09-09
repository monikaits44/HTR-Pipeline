# Final Experiment Reference — HTR-Pipeline

## Canonical Code

| Component | Path | Description |
|-----------|------|-------------|
| Model definitions | `models.py` | HTRNet wrapper — cnn_rnn, vit_rgts, torchvision_vit, trocr, htrvt |
| Trainer | `scripts/trainer.py` | Training loop, experiment dir, SWA, seed control |
| Dataset loader | `utils/htr_dataset.py` | IAM + synthetic data, Aachen splits |
| Augmentations | `utils/transforms.py` | aug_transforms_cnn, aug_transforms_vit |
| Metrics | `utils/metrics.py` | CER / WER computation |
| Fine-tuning | `utils/finetuning.py` | LLRD, GradualUnfreezer |
| Attention extraction | `utils/attention_extractor.py` | Core attention weight extraction |
| Attention metrics | `utils/attention_metrics.py` | Entropy, Gini, sparsity, localization |
| Evaluation | `scripts/postprocessing/single_model/evaluate.py` | Re-evaluation with per-sample output |
| Result aggregation | `scripts/postprocessing/aggregate_results.py` | Unified CSV + LaTeX table |

## Canonical Configs

| Config | Architecture | Key Settings |
|--------|-------------|--------------|
| `configs/config.yaml` | Base settings | seed=42, lr=1e-3, epochs=150, batch=8, IAM data |
| `configs/baseline.yaml` | CNN-RNN | head_type=both |
| `configs/baseline_vit_rgts_v2.yaml` | ViT-RGTS v2 (recommended) | CNN stem, dim=256, depth=6, 4 registers |
| `configs/baseline_vit_rgts.yaml` | ViT-RGTS v1 (legacy) | dim=128, depth=4 |
| `configs/torchvision_vit.yaml` | TorchVision ViT-B/16 | Pretrained ImageNet |
| `configs/trocr.yaml` | TrOCR-base | HuggingFace pretrained |
| `configs/finetune_torchvision_vit.yaml` | Fine-tune ViT-B/16 | LLRD + gradual unfreeze |
| `configs/finetune_trocr.yaml` | Fine-tune TrOCR | LLRD + gradual unfreeze |
| `configs/htrvt_read2016_r4.yaml` | HTR-VT on READ2016 | ResNet stem + 2D grid |
| `configs/beyond_memorization.yaml` | Beyond-Memorization | External repo integration |

## Experiment Execution

### SLURM Scripts (Canonical — `experiments_execution/slurm_modular/`)

All scripts source `run_experiment.sh` which handles environment setup, seed, and logging.

**IAM experiments (20 scripts):**
`iam/01_cnn_rnn.slurm` through `iam/20_finetune_trocr_swa.slurm`

**Ablation studies (22 scripts):**
`ablations/01_no_cnn_stem.slurm` through `ablations/22_seed_456.slurm`

**Synthetic data (16 scripts):**
`synthetic/01_cnn_rnn.slurm` through `synthetic/16_trocr_base_swa.slurm`

**READ2016 dataset variation (5 scripts):**
`read2016/01_cnn_rnn.slurm` through `read2016/05_vit_rgts_v2_16reg.slurm`

### Quick Start

```bash
# Single experiment
sbatch experiments_execution/slurm_modular/iam/07_vit_rgts_v2_4reg.slurm

# All IAM experiments
./experiments_execution/slurm_modular/submit_all_iam.sh

# All ablations
./experiments_execution/slurm_modular/submit_all_ablations.sh

# All READ2016
./experiments_execution/slurm_modular/submit_all_read2016.sh

# All 63 experiments
./experiments_execution/slurm_modular/submit_all.sh

# Batch evaluate all completed runs
bash evaluation_execution/batch_evaluate.sh

# Generate aggregate results + LaTeX tables
bash evaluation_execution/generate_final_results.sh

# Evaluate a specific run
RUN_ID=107 sbatch evaluation_execution/model_evaluation.sh

# Evaluate with specific config
RUN_ID=122 EVAL_CONFIGS="configs/config.yaml configs/baseline_vit_rgts_v2.yaml" \
  CONFIG_OVERRIDES="arch.num_registers=16" sbatch evaluation_execution/model_evaluation.sh

# Aggregate all results
python scripts/postprocessing/aggregate_results.py
```

## Key Experiment Results (83 completed runs, 63 final experiments pending)

*Note: Results below are from prior runs (80 epochs). Final 150-epoch results will be recorded after execution.*
*See `documents/final/FINAL_EXPERIMENT_EXECUTION_GUIDE.md` for the complete experiment matrix.*

| Run | Architecture | Registers | Epochs | SWA | Test CER | Test WER |
|-----|-------------|-----------|--------|-----|----------|----------|
| 107 | CNN-RNN | N/A | 80 | Yes | **4.87%** | — |
| 115 | CNN-RNN | N/A | 80 | No | 5.02% | — |
| 135 | ViT-RGTS v2 | 4 | 120 | No | 5.88% | — |
| 123 | ViT-RGTS v2 | 8 | 80 | No | ~6.1% | — |
| 142 | HTR-VT (READ2016) | 4 | 80 | No | 4.82% | 20.58% |

## Data

| Dataset | Location | Samples |
|---------|----------|---------|
| IAM train | `data/IAM/processed_lines/train/` | 6,482 lines |
| IAM val | `data/IAM/processed_lines/val/` | 976 lines |
| IAM test | `data/IAM/processed_lines/test/` | 2,915 lines |
| READ2016 | `data/READ2016/processed/` | — |
| Character set | 79 chars + 1 CTC blank = 80 classes | |

## Reproducibility

- **Seed**: 42 (set in `configs/config.yaml`, override with `seed=<N>`)
- **Deterministic**: `torch.backends.cudnn.deterministic=True` when seed >= 0
- **Hardware**: SLURM partition `rtx3080` (GPU)
- **Environment**: Python 3.9, `requirements.txt` (pinned versions)
- **Run artifacts**: `saved_models/experiments/run_<N>/` contains:
  - `config.json` — exact config used
  - `model.pt` — best checkpoint
  - `training.log` — epoch-level logs
  - `results.csv` — per-epoch metrics
  - `evaluation_details.csv` — per-sample predictions

## Publication Figures (Scripts)

| Script | Output |
|--------|--------|
| `scripts/pub_fig1_paper_fig5.py` | Character attention grid (Beyond-Memorization style) |
| `scripts/pub_fig2_register_comparison.py` | Register sweep comparison |
| `scripts/pub_fig3_gradcam_quantitative.py` | GradCAM + quantitative metrics |
| `scripts/pub_register_quantitative.py` | Attention quality bar charts |
| `scripts/pub_gradcam_analysis.py` | GradCAM analysis |
| `scripts/pub_fig5_character_attention.py` | Attention rollout with register effect |

Output location: `outputs/pub_figures/`

## Archived Items

| Archive Location | Contents |
|-----------------|----------|
| `archive/exploratory_scripts/` | 8 superseded exploratory analysis scripts |
| `archive/legacy_slurm_scripts/` | 16 monolithic SLURM scripts (replaced by modular) |
| `archive/helper_notes/` | Informal notes, email threads, instruction files |
| `archive/legacy_documents/` | Superseded documentation versions |
| `archive/legacy_backup/` | Early code versions |
| `archive/legacy_postprocessing_v1/` | First postprocessing iteration |
| `archive/legacy_visualization/` | Early visualization attempts |
| `archive/legacy_notebooks/` | Deprecated notebooks |
| `archive/experimental_notebooks/` | Exploratory Jupyter notebooks |
| `documents/archived_latest/` | Superseded docs (March 2024) |
| `documents/archived_most_latest/` | Superseded docs (April-May 2024) |
| `experiments_execution/slurm_scripts_v1/` | Original 16 monolithic scripts |
| `experiments_execution/slurm_modular/ablations/archived/` | Superseded epoch-120 ablation scripts |
