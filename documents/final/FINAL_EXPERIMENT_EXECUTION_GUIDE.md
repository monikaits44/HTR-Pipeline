# HTR-Pipeline: Final Experiment Execution Guide

> **Version**: Final (2026-08-29)
> **Standard Training**: 150 epochs, seed=42
> **Total Experiments**: 63 (20 IAM + 16 Synthetic + 22 Ablations + 5 READ2016)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Prerequisites](#3-prerequisites)
4. [Experiment Matrix](#4-experiment-matrix)
5. [Execution Instructions](#5-execution-instructions)
6. [Evaluation Pipeline](#6-evaluation-pipeline)
7. [Expected Artifacts](#7-expected-artifacts)
8. [Post-Experiment Analysis](#8-post-experiment-analysis)
9. [Troubleshooting](#9-troubleshooting)
10. [Archived Components](#10-archived-components)

---

## 1. Project Overview

### Research Question
How do register tokens (Darcet et al., 2023) in Vision Transformers affect attention quality and interpretability in Handwritten Text Recognition (HTR)?

### Core Hypothesis
Register tokens absorb global/style information, freeing patch tokens to produce fine-grained, character-localized attention maps — improving both recognition performance and explainability for downstream tasks like writer identification.

### Architecture Under Study
**ViT-RGTS v2**: CNN stem (128×1024 → 128 tokens) + 6-layer Transformer (dim=256, 8 heads) + Register tokens (0–16) + Dual CTC head (BiLSTM + CNN shortcut)

### Baselines
- **CNN-RNN**: Traditional CNN encoder + BiLSTM + CTC (7.4M params)
- **TorchVision ViT-B/16**: ImageNet-pretrained ViT (86M params)
- **TrOCR-base-handwritten**: HuggingFace pretrained OCR transformer (334M params)

### Datasets
| Dataset | Train | Val | Test | Characters |
|---------|-------|-----|------|------------|
| **IAM** (Aachen) | 6,482 | 976 | 2,915 | 79 |
| **READ2016** | 8,349 | 1,040 | 1,138 | 88 |

---

## 2. Repository Structure

```
HTR-Pipeline/
├── configs/                          # All experiment configurations
│   ├── config.yaml                   # Base config (150 epochs, seed=42)
│   ├── baseline.yaml                 # CNN-RNN architecture
│   ├── baseline_vit_rgts_v2.yaml     # ViT-RGTS v2 (MAIN)
│   ├── baseline_vit_rgts.yaml        # ViT-RGTS v1 (legacy)
│   ├── torchvision_vit.yaml          # TorchVision ViT-B/16
│   ├── trocr.yaml                    # TrOCR-base
│   ├── finetune_torchvision_vit.yaml # ViT-B/16 fine-tune (2-group LR)
│   └── finetune_trocr.yaml          # TrOCR fine-tune (LLRD)
├── models.py                         # All architecture definitions
├── scripts/
│   ├── trainer.py                    # Main training loop
│   └── postprocessing/               # Evaluation & visualization
├── utils/                            # Dataset, metrics, transforms
├── experiments_execution/
│   └── slurm_modular/                # All SLURM experiment scripts
│       ├── run_experiment.sh         # Central runner
│       ├── iam/           (20 scripts)
│       ├── synthetic/     (16 scripts)
│       ├── ablations/     (22 scripts)
│       └── read2016/      (5 scripts)
├── evaluation_execution/             # Evaluation scripts
│   ├── model_evaluation.sh           # Single-run evaluation
│   ├── batch_evaluate.sh             # Batch evaluation
│   └── generate_final_results.sh     # Aggregate results + plots
└── saved_models/experiments/         # All run outputs (run_1, run_2, ...)
```

---

## 3. Prerequisites

### Environment Setup
```bash
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline
source .venv/bin/activate
```

### Verify Dependencies
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import omegaconf, editdistance, albumentations, nltk; print('OK')"
```

### Verify Data
```bash
wc -l data/IAM/processed_lines/{train,val,test}/gt.txt
wc -l data/READ2016/processed/{train,val,test}/gt.txt
```

### Verify Log Directories
```bash
mkdir -p logs/modular evaluation_execution/logs
```

---

## 4. Experiment Matrix

### 4.1 IAM Core Experiments (20 runs)

All use 150 epochs, seed=42, IAM dataset.

| # | Script | Architecture | Registers | SWA | Walltime |
|---|--------|-------------|-----------|-----|----------|
| 1 | `iam/01_cnn_rnn.slurm` | CNN-RNN | — | No | 8h |
| 2 | `iam/02_cnn_rnn_swa.slurm` | CNN-RNN | — | Yes (ep110) | 10h |
| 3 | `iam/03_vit_rgts_v2_0reg.slurm` | ViT-RGTS v2 | 0 | No | 16h |
| 4 | `iam/04_vit_rgts_v2_0reg_swa.slurm` | ViT-RGTS v2 | 0 | Yes (ep110) | 20h |
| 5 | `iam/05_vit_rgts_v2_2reg.slurm` | ViT-RGTS v2 | 2 | No | 16h |
| 6 | `iam/06_vit_rgts_v2_2reg_swa.slurm` | ViT-RGTS v2 | 2 | Yes (ep110) | 20h |
| 7 | `iam/07_vit_rgts_v2_4reg.slurm` | ViT-RGTS v2 | 4 | No | 16h |
| 8 | `iam/08_vit_rgts_v2_4reg_swa.slurm` | ViT-RGTS v2 | 4 | Yes (ep110) | 20h |
| 9 | `iam/09_vit_rgts_v2_8reg.slurm` | ViT-RGTS v2 | 8 | No | 16h |
| 10 | `iam/10_vit_rgts_v2_8reg_swa.slurm` | ViT-RGTS v2 | 8 | Yes (ep110) | 20h |
| 11 | `iam/11_vit_rgts_v2_16reg.slurm` | ViT-RGTS v2 | 16 | No | 16h |
| 12 | `iam/12_vit_rgts_v2_16reg_swa.slurm` | ViT-RGTS v2 | 16 | Yes (ep110) | 20h |
| 13 | `iam/13_torchvision_vit_b16.slurm` | ViT-B/16 | 0 | No | 16h |
| 14 | `iam/14_torchvision_vit_b16_swa.slurm` | ViT-B/16 | 0 | Yes (ep110) | 20h |
| 15 | `iam/15_finetune_vit_b16.slurm` | ViT-B/16 FT | 4 | No | 20h |
| 16 | `iam/16_finetune_vit_b16_swa.slurm` | ViT-B/16 FT | 4 | Yes (ep110) | 24h |
| 17 | `iam/17_trocr_base.slurm` | TrOCR-base | — | No | 12h |
| 18 | `iam/18_trocr_base_swa.slurm` | TrOCR-base | — | Yes (ep110) | 16h |
| 19 | `iam/19_finetune_trocr.slurm` | TrOCR FT | — | No | 24h |
| 20 | `iam/20_finetune_trocr_swa.slurm` | TrOCR FT | — | Yes (ep110) | 28h |

### 4.2 Synthetic Pretraining (16 runs)

Train on synthetic data, test on IAM. 15 epochs (100× larger dataset).

| # | Script | Architecture | Registers | SWA |
|---|--------|-------------|-----------|-----|
| 1–2 | `synthetic/01-02` | CNN-RNN | — | ±SWA |
| 3–12 | `synthetic/03-12` | ViT-RGTS v2 | 0/2/4/8/16 | ±SWA |
| 13–14 | `synthetic/13-14` | ViT-B/16 | 0 | ±SWA |
| 15–16 | `synthetic/15-16` | TrOCR-base | — | ±SWA |

### 4.3 Ablation Studies (22 runs)

All on IAM, 150 epochs, 4 registers (unless noted). Tests one variable at a time.

| # | Script | Variable | Change | Baseline |
|---|--------|----------|--------|----------|
| 1 | `01_no_cnn_stem` | Architecture | Remove CNN stem → raw patches | iam/07 |
| 2 | `02_rnn_head_only` | Head | RNN only (no CNN shortcut) | iam/07 |
| 3 | `03_cnn_head_only` | Head | CNN only (no RNN) | iam/07 |
| 4 | `04_gru_head` | Head | GRU instead of LSTM | iam/07 |
| 5 | `05_vit_augmentation` | Augmentation | Strong ViT augmentation | iam/07 |
| 6 | `06_depth4` | Depth | 4 transformer layers | iam/07 |
| 7 | `07_depth8` | Depth | 8 transformer layers | iam/07 |
| 8 | `08_dim128` | Dimension | dim=128 (smaller model) | iam/07 |
| 9 | `09_dim512` | Dimension | dim=512 (larger model) | iam/07 |
| 10 | `10_rnn1_layer` | RNN depth | 1-layer BiLSTM | iam/07 |
| 11 | `11_dropout03` | Regularization | dropout=0.3 | iam/07 |
| 12 | `12_swa_early` | SWA timing | SWA start=75 (50% of training) | iam/08 |
| 13 | `13_swa_late` | SWA timing | SWA start=135 (90% of training) | iam/08 |
| 14 | `14_batch16` | Batch size | batch=16, no grad accum | iam/07 |
| 15 | `15_ft_vit_gradual` | Fine-tuning | Gradual unfreezing ViT-B/16 | iam/15 |
| 16 | `16_ft_trocr_gradual` | Fine-tuning | Gradual unfreezing TrOCR | iam/19 |
| 17 | `17_vit_rgts_v1` | Architecture | ViT-RGTS v1 (legacy, dim=128) | iam/07 |
| 18 | `18_lr_5e4` | Learning rate | lr=5e-4 (halved) | iam/07 |
| 19 | `19_swa_lr_high` | SWA LR | swa_lr=1e-3 (doubled) | iam/08 |
| 20 | `20_swa_lr_low` | SWA LR | swa_lr=1e-4 (halved) | iam/08 |
| 21 | `21_seed_123` | Seed | seed=123 (reproducibility) | iam/07 |
| 22 | `22_seed_456` | Seed | seed=456 (reproducibility) | iam/07 |

### 4.4 READ2016 Dataset Variation (5 runs)

Validate that register effects generalize across datasets. 150 epochs.

| # | Script | Architecture | Registers |
|---|--------|-------------|-----------|
| 1 | `read2016/01_cnn_rnn` | CNN-RNN | — |
| 2 | `read2016/02_vit_rgts_v2_0reg` | ViT-RGTS v2 | 0 |
| 3 | `read2016/03_vit_rgts_v2_4reg` | ViT-RGTS v2 | 4 |
| 4 | `read2016/04_vit_rgts_v2_8reg` | ViT-RGTS v2 | 8 |
| 5 | `read2016/05_vit_rgts_v2_16reg` | ViT-RGTS v2 | 16 |

---

## 5. Execution Instructions

### 5.1 Submit All Experiments

```bash
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/slurm_modular

# All 63 experiments (requires confirmation)
bash submit_all.sh

# Or submit by category:
bash submit_all_iam.sh          # 20 IAM core experiments
bash submit_all_synthetic.sh    # 16 synthetic pretraining
bash submit_all_ablations.sh    # 22 ablation studies
bash submit_all_read2016.sh     # 5 READ2016 dataset variation
```

### 5.2 Submit Individual Experiments

```bash
# Single experiment
sbatch experiments_execution/slurm_modular/iam/07_vit_rgts_v2_4reg.slurm

# Register sweep only (10 jobs: 5 registers × ±SWA)
bash experiments_execution/slurm_modular/submit_v2_register_sweep.sh
```

### 5.3 Recommended Execution Order

For optimal resource usage and early debugging:

```
Phase 1 — Validation (2 jobs, ~16h):
  sbatch iam/01_cnn_rnn.slurm          # Fast baseline sanity check
  sbatch iam/07_vit_rgts_v2_4reg.slurm # Main architecture sanity check

Phase 2 — Core Register Sweep (10 jobs, ~20h parallel):
  bash submit_v2_register_sweep.sh     # All register counts ±SWA

Phase 3 — Baselines (8 jobs, ~28h parallel):
  sbatch iam/13_torchvision_vit_b16.slurm through iam/20_finetune_trocr_swa.slurm

Phase 4 — Ablations (22 jobs, ~28h parallel):
  bash submit_all_ablations.sh

Phase 5 — Dataset Variation (5 jobs, ~20h parallel):
  bash submit_all_read2016.sh

Phase 6 — Synthetic Pretraining (16 jobs, ~8h parallel):
  bash submit_all_synthetic.sh
```

### 5.4 Monitoring

```bash
squeue -u $USER                               # All jobs
squeue -u $USER --format="%.10i %.9P %.25j %.8u %.2t %.10M %.6D %R"  # Detailed
tail -f logs/modular/v2_4reg_iam_<JOBID>.out  # Live training log
```

### 5.5 Manual Execution (without SLURM)

```bash
source .venv/bin/activate
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline

# CNN-RNN baseline
python scripts/trainer.py configs/config.yaml configs/baseline.yaml data.mode=iam

# ViT-RGTS v2 with 4 registers
python scripts/trainer.py configs/config.yaml configs/baseline_vit_rgts_v2.yaml \
    data.mode=iam arch.num_registers=4

# ViT-RGTS v2 on READ2016
python scripts/trainer.py configs/config.yaml configs/baseline_vit_rgts_v2.yaml \
    data.mode=iam data.path=./data/READ2016/processed arch.num_registers=4

# With SWA
python scripts/trainer.py configs/config.yaml configs/baseline_vit_rgts_v2.yaml \
    data.mode=iam arch.num_registers=4 \
    swa.enabled=true swa.start_epoch=110 swa.swa_lr=5e-4
```

---

## 6. Evaluation Pipeline

### 6.1 Per-Run Evaluation

```bash
# Evaluate a specific run
RUN_ID=150 sbatch evaluation_execution/model_evaluation.sh

# With custom config
RUN_ID=150 EVAL_CONFIGS="configs/config.yaml configs/baseline_vit_rgts_v2.yaml" \
    sbatch evaluation_execution/model_evaluation.sh
```

### 6.2 Batch Evaluation (All Completed Runs)

```bash
# Evaluate all unevaluated runs
bash evaluation_execution/batch_evaluate.sh

# Evaluate only new runs (from run 150 onward)
bash evaluation_execution/batch_evaluate.sh --from 150

# Preview without submitting
bash evaluation_execution/batch_evaluate.sh --dry-run
```

### 6.3 Aggregate Results

```bash
# Generate unified results table + LaTeX tables
bash evaluation_execution/generate_final_results.sh
```

Output: `outputs/tables/unified_results.csv`, `outputs/tables/*.tex`

---

## 7. Expected Artifacts

### Per-Run Artifacts (saved_models/experiments/run_N/)

| File | Description |
|------|-------------|
| `config.json` | Full experiment configuration |
| `model.pt` | Final model checkpoint |
| `model_swa.pt` | SWA model checkpoint (if SWA enabled) |
| `results.csv` | Per-epoch metrics (loss, CER, WER) |
| `training.log` | Full training console output |
| `evaluation_details.csv` | Per-sample predictions + errors |
| `attention_weights/` | Attention maps per epoch (ViT only) |

### Aggregate Artifacts (outputs/)

| Path | Description |
|------|-------------|
| `outputs/tables/unified_results.csv` | All runs comparison table |
| `outputs/tables/register_sweep.tex` | Register ablation LaTeX table |
| `outputs/tables/ablations.tex` | Ablation study LaTeX table |
| `outputs/tables/architecture_comparison.tex` | Best-per-architecture table |
| `outputs/report_figures/` | Training curves, attention quality plots |

### Key Metrics Per Run

| Metric | Description | Range |
|--------|-------------|-------|
| **CER** | Character Error Rate (edit distance / target length) | [0, 1] |
| **WER** | Word Error Rate (word-level edit distance) | [0, 1] |
| **Exact Match Rate** | Fraction of perfectly predicted lines | [0, 1] |
| **Attention Entropy** | Entropy of attention distribution | Lower = more focused |
| **Gini Sparsity** | Gini coefficient of attention weights | Higher = sparser |

---

## 8. Post-Experiment Analysis

### 8.1 Storyline for Results & Discussion

The experiment matrix is designed to support these narrative threads:

**Thread 1: Register Token Effect (Primary Result)**
- Compare ViT-RGTS v2 with 0/2/4/8/16 registers on IAM
- Show CER/WER improvement curve with register count
- Demonstrate attention quality improvement (entropy, localization)

**Thread 2: SWA Interaction**
- For each register count, compare ±SWA
- Show whether SWA benefits register-augmented models more/less

**Thread 3: Architecture Comparison**
- CNN-RNN vs ViT-RGTS v2 (4reg) vs ViT-B/16 vs TrOCR
- Parameter count vs performance tradeoff

**Thread 4: Generalization (READ2016)**
- Register sweep on READ2016 mirrors IAM findings
- Different charset (88 vs 79 chars) validates robustness

**Thread 5: Ablation Insights**
- CNN stem is critical (vs raw patches)
- Dual head (both) > RNN-only > CNN-only
- dim=256 is the sweet spot (128 too small, 512 overfits)
- 6 layers is optimal (4 underfits, 8 overfits on 6.5K samples)

**Thread 6: Reproducibility**
- Seeds 42/123/456 give consistent rankings
- Standard deviation across seeds < 0.5% CER

### 8.2 Visualization Scripts

```bash
# Per-character attention maps (Beyond-Memorization style)
python scripts/postprocessing/beyond_memorization_viz.py \
    --run-dir saved_models/experiments/run_N

# Register comparison grid
python scripts/pub_fig2_register_comparison.py

# GradCAM analysis
python scripts/pub_fig3_gradcam_quantitative.py

# Attention quality bar charts
python scripts/pub_register_quantitative.py

# Training curves comparison
python scripts/postprocessing/comparative/plot_training_metrics.py \
    --runs-dir saved_models/experiments --output-dir outputs/report_figures
```

### 8.3 Generating LaTeX Tables

```bash
python scripts/postprocessing/aggregate_results.py \
    --runs-dir saved_models/experiments --output-dir outputs/tables
```

Produces:
- `register_sweep.tex` — ViT-RGTS v2 register ablation (for main results table)
- `architecture_comparison.tex` — Cross-architecture comparison
- `ablations.tex` — Ablation study results

---

## 9. Troubleshooting

### Common Issues

| Issue | Fix |
|-------|-----|
| `CUDA out of memory` | Reduce `train.batch_size` or increase `train.gradient_accumulation` |
| `FileNotFoundError: gt.txt` | Verify data path: `ls data/IAM/processed_lines/train/gt.txt` |
| `KeyError in character mapping` | Run with `data.mode=iam` (auto-computes charset) |
| `SWA BN update hangs` | Set `swa.update_bn_steps=100` (limits BN calibration passes) |
| `SLURM timeout` | Increase `--time` in SLURM script |
| `Run number collision` | File locking handles this — retry if `run_lock` is stale |

### Checking a Run's Configuration

```bash
python -c "import json; c=json.load(open('saved_models/experiments/run_N/config.json')); print(json.dumps(c, indent=2))"
```

### Re-evaluating a Run

```bash
python scripts/postprocessing/single_model/evaluate.py \
    configs/config.yaml configs/baseline_vit_rgts_v2.yaml \
    resume=saved_models/experiments/run_N/model.pt
```

---

## 10. Archived Components

### Archived SLURM Scripts
- `experiments_execution/slurm_scripts_v1/` — Original 16 monolithic SLURM scripts (replaced by modular system)
- `experiments_execution/slurm_modular/ablations/archived/` — Superseded epoch-120 ablations

### Archived Code
- `archive/exploratory_scripts/` — 8 early exploration scripts
- `archive/helper_notes/` — Informal development notes
- `archive/legacy_*` — Legacy versions of various components

### Archived Documentation
- `documents/archived_latest/` — Previous documentation iterations
- `documents/archived_most_latest/` — Earlier archived docs

---

## Appendix A: Configuration Reference

### Base Config (config.yaml)
```yaml
seed: 42
train.num_epochs: 150
train.batch_size: 8
train.lr: 1e-3
swa.start_epoch: 110
swa.swa_lr: 5e-4
data.path: ./data/IAM/processed_lines
data.mode: iam
```

### Config Override Precedence
1. `config.yaml` (base defaults)
2. Architecture YAML (e.g., `baseline_vit_rgts_v2.yaml`)
3. CLI overrides (e.g., `arch.num_registers=4 swa.enabled=true`)

### SWA Configuration by Architecture

| Architecture | swa_lr | swa_start | Rationale |
|-------------|--------|-----------|-----------|
| CNN-RNN | 5e-4 | 110 | Standard from-scratch |
| ViT-RGTS v2 | 5e-4 | 110 | Standard from-scratch |
| TorchVision ViT | 5e-5 | 110 | Pretrained: lower LR |
| TrOCR | 5e-5 | 110 | Pretrained: lower LR |
| Finetune ViT | 5e-6 | 110 | Fine-tuned: very conservative |
| Finetune TrOCR | 5e-6 | 110 | Fine-tuned: very conservative |

## Appendix B: GPU-Hours Estimate

| Category | Jobs | Avg Hours | Total GPU-h |
|----------|------|-----------|-------------|
| IAM Core | 20 | 17 | 340 |
| Synthetic | 16 | 6 | 96 |
| Ablations | 22 | 17 | 374 |
| READ2016 | 5 | 18 | 90 |
| **Total** | **63** | — | **~900** |

*Estimate assumes RTX 3080 (10 GB VRAM). Actual times depend on cluster load.*
