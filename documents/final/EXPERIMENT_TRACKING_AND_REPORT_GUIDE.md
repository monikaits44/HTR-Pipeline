# Final Experiment Tracking & Report Writing Guide

> **Submitted**: 2026-08-29 14:02 CEST
> **Total Jobs**: 64 (1 extraction + 63 training)
> **Cluster**: rtx3080 partition (RTX 3080, 10 GB VRAM each)
> **Standard**: 150 epochs, seed=42, CTC loss

---

## 1. Complete Job ID → Experiment Mapping

### Phase 1–2: Baselines + Register Sweep (12 jobs)

| Job ID | Script | Arch | Regs | SWA | Time | Purpose |
|--------|--------|------|------|-----|------|---------|
| 1797650 | iam/01_cnn_rnn | CNN-RNN | — | No | 8h | **Baseline** (traditional CNN+BiLSTM+CTC) |
| 1797652 | iam/02_cnn_rnn_swa | CNN-RNN | — | Yes | 10h | Baseline + weight averaging |
| 1797653 | iam/03_v2_0reg | ViT-RGTS v2 | 0 | No | 16h | **No registers** (attention sink expected) |
| 1797654 | iam/04_v2_0reg_swa | ViT-RGTS v2 | 0 | Yes | 20h | No registers + SWA |
| 1797655 | iam/05_v2_2reg | ViT-RGTS v2 | 2 | No | 16h | Minimal registers |
| 1797656 | iam/06_v2_2reg_swa | ViT-RGTS v2 | 2 | Yes | 20h | Minimal registers + SWA |
| 1797651 | iam/07_v2_4reg | ViT-RGTS v2 | 4 | No | 16h | **Default/recommended** register count |
| 1797657 | iam/08_v2_4reg_swa | ViT-RGTS v2 | 4 | Yes | 20h | Default + SWA |
| 1797658 | iam/09_v2_8reg | ViT-RGTS v2 | 8 | No | 16h | Higher register count |
| 1797659 | iam/10_v2_8reg_swa | ViT-RGTS v2 | 8 | Yes | 20h | Higher + SWA |
| 1797660 | iam/11_v2_16reg | ViT-RGTS v2 | 16 | No | 16h | Maximum registers |
| 1797661 | iam/12_v2_16reg_swa | ViT-RGTS v2 | 16 | Yes | 20h | Maximum + SWA |

### Phase 3: Pretrained Baselines (8 jobs)

| Job ID | Script | Arch | SWA | Time | Purpose |
|--------|--------|------|-----|------|---------|
| 1797662 | iam/13_vit_b16 | TorchVision ViT-B/16 | No | 16h | ImageNet-pretrained ViT (86M params) |
| 1797663 | iam/14_vit_b16_swa | TorchVision ViT-B/16 | Yes | 20h | Pretrained + SWA |
| 1797664 | iam/15_ft_vit | Finetune ViT-B/16 | No | 20h | 2-group differential LR fine-tuning |
| 1797665 | iam/16_ft_vit_swa | Finetune ViT-B/16 | Yes | 24h | Fine-tune + SWA |
| 1797666 | iam/17_trocr | TrOCR-base | No | 12h | HuggingFace pretrained (frozen encoder) |
| 1797667 | iam/18_trocr_swa | TrOCR-base | Yes | 16h | Frozen encoder + SWA |
| 1797668 | iam/19_ft_trocr | Finetune TrOCR | No | 24h | LLRD fine-tuning (all layers trainable) |
| 1797680 | iam/20_ft_trocr_swa | Finetune TrOCR | Yes | 24h | LLRD fine-tune + SWA |

### Phase 4: Architecture Ablations (11 jobs)

All use ViT-RGTS v2 with 4 registers on IAM, 150 epochs, changing **one variable**.

| Job ID | Script | Variable Changed | From → To | Time |
|--------|--------|-----------------|-----------|------|
| 1797669 | abl/01_no_cnn_stem | CNN stem | present → removed | 16h |
| 1797670 | abl/02_rnn_head_only | Head type | both → rnn | 16h |
| 1797671 | abl/03_cnn_head_only | Head type | both → cnn | 16h |
| 1797672 | abl/04_gru_head | RNN type | LSTM → GRU | 16h |
| 1797673 | abl/05_vit_augmentation | Augmentation | cnn (moderate) → vit (strong) | 16h |
| 1797674 | abl/06_depth4 | Transformer depth | 6 → 4 | 12h |
| 1797675 | abl/07_depth8 | Transformer depth | 6 → 8 | 20h |
| 1797676 | abl/08_dim128 | Embedding dim | 256 → 128 | 12h |
| 1797677 | abl/09_dim512 | Embedding dim | 256 → 512 | 24h |
| 1797678 | abl/10_rnn1_layer | RNN layers | 3 → 1 | 12h |
| 1797679 | abl/11_dropout03 | Dropout | 0.1 → 0.3 | 16h |

### Phase 5: SWA / LR / Seed / FT Ablations (11 jobs)

| Job ID | Script | Variable Changed | Detail | Time |
|--------|--------|-----------------|--------|------|
| 1797681 | abl/12_swa_early | SWA start | epoch 110 → 75 (50%) | 20h |
| 1797682 | abl/13_swa_late | SWA start | epoch 110 → 135 (90%) | 20h |
| 1797683 | abl/14_batch16 | Batch size | 8 (accum 4) → 16 (no accum) | 16h |
| 1797684 | abl/15_ft_vit_gu | FT strategy | legacy 2-group → gradual unfreeze | 24h |
| 1797685 | abl/16_ft_trocr_gu | FT strategy | LLRD → gradual unfreeze | 24h |
| 1797686 | abl/17_v1_baseline | Architecture | v2 (CNN stem) → v1 (raw patches) | 12h |
| 1797687 | abl/18_lr_5e4 | Base LR | 1e-3 → 5e-4 | 16h |
| 1797688 | abl/19_swa_lr_high | SWA LR | 5e-4 → 1e-3 | 20h |
| 1797689 | abl/20_swa_lr_low | SWA LR | 5e-4 → 1e-4 | 20h |
| 1797690 | abl/21_seed_123 | Seed | 42 → 123 | 16h |
| 1797691 | abl/22_seed_456 | Seed | 42 → 456 | 16h |

### Phase 6: READ2016 Dataset Variation (5 jobs)

| Job ID | Script | Arch | Regs | Time | Purpose |
|--------|--------|------|------|------|---------|
| 1797692 | r16/01_cnn_rnn | CNN-RNN | — | 10h | Baseline on different dataset |
| 1797693 | r16/02_v2_0reg | ViT-RGTS v2 | 0 | 20h | No registers on READ2016 |
| 1797694 | r16/03_v2_4reg | ViT-RGTS v2 | 4 | 20h | Default registers on READ2016 |
| 1797695 | r16/04_v2_8reg | ViT-RGTS v2 | 8 | 20h | More registers on READ2016 |
| 1797696 | r16/05_v2_16reg | ViT-RGTS v2 | 16 | 20h | Max registers on READ2016 |

### Phase 7: Synthetic Pretraining (1 + 16 jobs)

| Job ID | Script | Detail | Time | Depends On |
|--------|--------|--------|------|------------|
| 1797717 | extract_synthetic | LMDB → 999K PNG files | 4h | — |
| 1797718 | syn/01_cnn_rnn | CNN-RNN, 15 epochs | 24h | 1797717 |
| 1797719 | syn/02_cnn_rnn_swa | CNN-RNN + SWA | 24h | 1797717 |
| 1797720–21 | syn/03-04_v2_0reg±swa | 0 registers | 24h | 1797717 |
| 1797722–23 | syn/05-06_v2_2reg±swa | 2 registers | 24h | 1797717 |
| 1797724–25 | syn/07-08_v2_4reg±swa | 4 registers | 24h | 1797717 |
| 1797726–27 | syn/09-10_v2_8reg±swa | 8 registers | 24h | 1797717 |
| 1797728–29 | syn/11-12_v2_16reg±swa | 16 registers | 24h | 1797717 |
| 1797730–31 | syn/13-14_vit_b16±swa | ViT-B/16 pretrained | 24h | 1797717 |
| 1797732–33 | syn/15-16_trocr±swa | TrOCR-base | 24h | 1797717 |

---

## 2. Estimated Completion Timeline

| Phase | Jobs | Wall-Clock (parallel) | Estimated Completion |
|-------|------|-----------------------|---------------------|
| **1–2** (Baselines + Sweep) | 12 | ~20h | Sat 29 Aug ~10:00 |
| **3** (Pretrained) | 8 | ~24h | Sun 30 Aug ~14:00 |
| **4** (Arch Ablations) | 11 | ~24h | Mon 31 Aug ~14:00 |
| **5** (SWA/Seed Ablations) | 11 | ~24h | Tue 01 Sep ~14:00 |
| **6** (READ2016) | 5 | ~20h | Tue 01 Sep ~10:00 |
| **7** (Synthetic) | 17 | ~4h extract + 24h train | Wed 02 Sep ~14:00 |

**Total estimated completion: ~3–4 days** (depending on cluster load and GPU availability).

*Note: 5 jobs are already running. Remaining jobs queue as GPUs free up. SLURM schedules them based on priority.*

---

## 3. How to Monitor Progress

```bash
# All your jobs
squeue -u $USER

# Detailed view
squeue -u $USER --format="%.8i %.25j %.2t %.10M %.10l %R" --sort=i

# Watch live training of a specific job
tail -f logs/modular/<jobname>_<jobid>.out

# Check completed jobs
sacct -u $USER --starttime=2026-08-29 --format=JobID,JobName,State,Elapsed,ExitCode

# Count running/pending/completed
squeue -u $USER -t R -h | wc -l   # running
squeue -u $USER -t PD -h | wc -l  # pending
```

---

## 4. Per-Run Artifacts (What Each Job Produces)

Every training job creates `saved_models/experiments/run_N/` containing:

| File | Content | Use in Report |
|------|---------|---------------|
| `config.json` | Full experiment config | Appendix: hyperparameters |
| `model.pt` | Final model weights | Inference / attention extraction |
| `model_swa.pt` | SWA model (if enabled) | Compare SWA vs base |
| `results.csv` | Per-epoch: loss, CER, WER | Training curves (Fig.) |
| `training.log` | Full console output | Debugging |
| `evaluation_details.csv` | Per-sample predictions | Error analysis |
| `attention_weights/` | Per-epoch attention maps | Attention visualization (Fig.) |

---

## 5. Post-Experiment Steps (After All Jobs Complete)

### Step 1: Aggregate Results
```bash
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline
source .venv/bin/activate

python scripts/postprocessing/aggregate_results.py \
    --runs-dir saved_models/experiments \
    --output-dir outputs/tables
```
Produces: `unified_results.csv`, `register_sweep.tex`, `ablations.tex`, `architecture_comparison.tex`

### Step 2: Batch Evaluation (re-evaluate all runs uniformly)
```bash
bash evaluation_execution/batch_evaluate.sh --from 143
```

### Step 3: Generate Training Curves
```bash
python scripts/postprocessing/comparative/plot_training_metrics.py \
    --runs-dir saved_models/experiments \
    --output-dir outputs/report_figures
```

### Step 4: Attention Visualization (for key runs)
```bash
# Per-character attention maps (Beyond-Memorization style)
python scripts/postprocessing/beyond_memorization_viz.py --run-dir saved_models/experiments/run_N

# Register comparison grid (0 vs 4 vs 8 vs 16 registers)
python scripts/pub_fig2_register_comparison.py

# Attention quality metrics (entropy, Gini, sparsity)
python scripts/pub_register_quantitative.py
```

### Step 5: Generate LaTeX Tables
```bash
bash evaluation_execution/generate_final_results.sh
```

---

## 6. Report Writing Guide: Results Structure

### 6.1 Main Results Table (Register Sweep on IAM)

**Data source**: Jobs 1797650–1797661 (Phase 1–2)

| What to report | Where to find |
|----------------|--------------|
| CER/WER for 0/2/4/8/16 regs ± SWA | `outputs/tables/register_sweep.tex` |
| Best val CER epoch | `results.csv` → min(val/cer) row |
| Test CER at best val epoch | Same row, test/cer column |
| Training curves | `results.csv` → plot loss, CER, WER vs epoch |

**Key narrative**: "Increasing register tokens from 0 to 4 improves CER by X%, with diminishing returns at 8 and 16."

### 6.2 Architecture Comparison Table

**Data source**: Jobs 1797650, 1797651, 1797662, 1797666 (best per arch)

| Architecture | Params | CER | WER | Notes |
|-------------|--------|-----|-----|-------|
| CNN-RNN | ~7.4M | ? | ? | Traditional baseline |
| ViT-RGTS v2 (4reg) | ~8.9M | ? | ? | Proposed method |
| ViT-B/16 (pretrained) | ~86M | ? | ? | Transfer learning |
| TrOCR-base | ~334M | ? | ? | SOTA pretrained |

**Key narrative**: "ViT-RGTS v2 achieves competitive CER with 10× fewer parameters than pretrained models, while providing interpretable attention maps."

### 6.3 Attention Quality Analysis

**Data source**: Attention weights from runs with 0/4/8/16 registers

| Metric | 0 reg | 4 reg | 8 reg | 16 reg | Interpretation |
|--------|-------|-------|-------|--------|----------------|
| Attention Entropy | ? | ? | ? | ? | Lower = more focused |
| Gini Sparsity | ? | ? | ? | ? | Higher = sparser |
| Character Localization | ? | ? | ? | ? | Higher = better alignment |

**Key narrative**: "Register tokens absorb global/style information, enabling patch tokens to attend locally to character strokes."

### 6.4 Ablation Study Table

**Data source**: Jobs 1797669–1797691 (Phase 4–5)

Organize as: "Effect of X on CER" where baseline is ViT-RGTS v2 (4reg, 150ep).

| Ablation Category | Variants | Report Section |
|-------------------|----------|----------------|
| CNN Stem | with vs without | Architecture Design |
| Head Type | both vs rnn vs cnn vs gru | CTC Head Analysis |
| Model Size | dim 128/256/512, depth 4/6/8 | Scaling Analysis |
| Regularization | dropout 0.1/0.3, SWA ± | Training Strategy |
| SWA Timing | start 75/110/135 | SWA Sensitivity |
| SWA LR | 1e-4 / 5e-4 / 1e-3 | SWA Sensitivity |
| Reproducibility | seed 42/123/456 | Variance Analysis |

### 6.5 Dataset Generalization (READ2016)

**Data source**: Jobs 1797692–1797696 (Phase 6)

| Dataset | Arch | Regs | CER | WER |
|---------|------|------|-----|-----|
| IAM | ViT-RGTS v2 | 0/4/8/16 | ? | ? |
| READ2016 | ViT-RGTS v2 | 0/4/8/16 | ? | ? |

**Key narrative**: "Register token improvements generalize across datasets with different scripts and character sets."

### 6.6 Synthetic Pretraining (If Applicable)

**Data source**: Jobs 1797718–1797733 (Phase 7)

Compare synthetic-pretrained models vs from-scratch on IAM test set.

---

## 7. Key Figures to Generate for Report

| Figure | Script | Data Source |
|--------|--------|-------------|
| Training curves (loss, CER vs epoch) | `plot_training_metrics.py` | results.csv |
| Register sweep bar chart | `pub_register_quantitative.py` | unified_results.csv |
| Per-character attention maps | `beyond_memorization_viz.py` | attention_weights/ |
| Register comparison grid (0 vs 4 vs 16) | `pub_fig2_register_comparison.py` | attention_weights/ |
| Attention entropy vs register count | `attention_quality_metrics.py` | attention_weights/ |
| Architecture comparison bar chart | From aggregate_results.py | unified_results.csv |
| Ablation tornado/sensitivity chart | Manual from ablations.tex | unified_results.csv |

---

## 8. Configuration Summary

### Training Hyperparameters (Standard)
- **Epochs**: 150
- **Batch size**: 8 (effective 32 via gradient accumulation for ViT)
- **Optimizer**: Adam (architecture-specific LR groups)
- **Scheduler**: Warmup (5 ep) + Cosine annealing
- **SWA**: start epoch 110, lr 5e-4 (from-scratch) / 5e-5 (pretrained) / 5e-6 (finetune)
- **Seed**: 42
- **Loss**: CTC (blank index 0, zero_infinity=True)
- **Augmentation**: albumentations (CNN-moderate for most; ViT-strong for ablation)

### ViT-RGTS v2 Architecture
- **CNN Stem**: 4 conv layers, collapses 128×1024 → 128 tokens
- **Transformer**: dim=256, depth=6, heads=8, mlp_dim=1024
- **Registers**: 4 learnable tokens (sweep: 0/2/4/8/16)
- **Head**: Dual CTC (3-layer BiLSTM + CNN shortcut)
- **Params**: ~8.9M total

### Datasets
| Dataset | Train | Val | Test | Chars |
|---------|-------|-----|------|-------|
| IAM (Aachen splits) | 6,482 | 976 | 2,915 | 79 |
| READ2016 | 8,349 | 1,040 | 1,138 | 88 |
| Synthetic (from LMDB) | ~950K | ~50K | IAM test | TBD |

---

## 9. Quick Reference Commands

```bash
# Check which runs completed
ls saved_models/experiments/run_{143..210}/model.pt 2>/dev/null | wc -l

# Get CER for a specific run
tail -1 saved_models/experiments/run_N/results.csv

# Compare all runs quickly
python scripts/postprocessing/aggregate_results.py --runs-dir saved_models/experiments --output-dir outputs/tables
cat outputs/tables/unified_results.csv | column -t -s,

# Cancel all pending jobs (if needed)
scancel -u $USER -t PD

# Cancel specific job
scancel <JOBID>
```
