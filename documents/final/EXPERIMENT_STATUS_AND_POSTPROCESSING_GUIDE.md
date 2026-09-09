# Final Experiment Status Report & Post-Processing Guide

> **Generated**: 2026-08-30
> **Run Range**: run_143 through run_205 (63 experiments)

---

## 1. Execution Summary

| Category | Jobs | Status | Run Numbers |
|----------|------|--------|-------------|
| IAM Core (baselines + register sweep) | 12 | **COMPLETED** | 143–152 |
| IAM Pretrained (ViT-B/16, TrOCR ± FT) | 8 | **COMPLETED** | 153–156, 172–174, 177–178, 180, 182–183 |
| Architecture Ablations | 11 | **COMPLETED** | 157–165 |
| SWA/LR/Seed Ablations | 11 | **COMPLETED** | 167–171, 175–176, 181, 186, 188 |
| READ2016 Dataset Variation | 5 | **COMPLETED** | 166, 184–185, 187, 190 |
| Synthetic Extraction | 1 | **COMPLETED** | (no run) |
| Synthetic Training | 16 | **RUNNING** | 189, 191–205 |
| **Total** | **64** | **47 done / 16 running** | **143–205** |

### Synthetic Runs: Will Reach Walltime Limit

The synthetic dataset has ~950K training samples (vs 6.5K IAM). Per-epoch times:

| Architecture | Seconds/Epoch | Epochs Possible in 24h | Target | Status |
|-------------|---------------|----------------------|--------|--------|
| CNN-RNN | 7,873 (~2.1h) | ~11 | 15 | epoch 10, will reach ~11 |
| ViT-RGTS v2 | 8,221 (~2.3h) | ~10 | 15 | epoch 7–9, will reach ~10 |
| TorchVision ViT | 36,127 (~10h) | ~2 | 15 | epoch 1, will stay ~2 |
| TrOCR | 23,644 (~6.6h) | ~3 | 15 | epoch 2, will reach ~3 |

**Impact**: Synthetic results are usable but partial. Best-model checkpoints are saved progressively. For report purposes, note these as "trained for N epochs" rather than the target 15.

**Estimated completion for all 16 synthetic jobs**: ~2–4 hours from now (walltime timeout).

---

## 2. Complete Run-to-Experiment Mapping

### 2.1 IAM Core — Baselines + Register Sweep (12 runs)

| Run | Architecture | Registers | SWA | Test CER (best val) | Test WER | Params |
|-----|-------------|-----------|-----|--------------------:|--------:|-------:|
| **143** | CNN-RNN | — | No | **4.89%** | 16.47% | 7.36M |
| **145** | CNN-RNN | — | Yes | **4.62%** | 15.70% | 7.36M |
| **144** | ViT-RGTS v2 | 0 | No | 5.74% | 18.87% | 9.48M |
| **147** | ViT-RGTS v2 | 0 | Yes | 5.90% | 19.12% | 9.48M |
| **148** | ViT-RGTS v2 | 2 | No | 5.98% | 19.39% | 9.48M |
| **149** | ViT-RGTS v2 | 2 | Yes | 6.16% | 20.04% | 9.48M |
| **146** | ViT-RGTS v2 | 4 | No | 5.96% | 19.33% | 9.48M |
| **150** | ViT-RGTS v2 | 4 | Yes | 5.77% | 19.05% | 9.48M |
| **151** | ViT-RGTS v2 | 8 | No | 5.94% | 19.32% | 9.48M |
| **156** | ViT-RGTS v2 | 8 | Yes | 5.96% | 19.51% | 9.48M |
| **152** | ViT-RGTS v2 | 16 | No | 5.88% | 19.13% | 9.48M |
| **172** | ViT-RGTS v2 | 16 | Yes | 5.88% | 19.25% | 9.48M |

### 2.2 IAM Pretrained Baselines (8 runs)

| Run | Architecture | SWA | Test CER (best val) | Test WER | Params |
|-----|-------------|-----|--------------------:|--------:|-------:|
| **153** | TorchVision ViT-B/16 | No | 73.14% | 123.71% | 91.9M |
| **173** | TorchVision ViT-B/16 | Yes | 18.34% | 43.72% | 91.9M |
| **174** | Finetune ViT-B/16 | No | 15.09% | 38.19% | 91.9M |
| **177** | Finetune ViT-B/16 | Yes | 15.65% | 40.09% | 91.9M |
| **155** | TrOCR-base (frozen) | No | 71.98% | 99.39% | 3.72M |
| **154** | TrOCR-base (frozen) | Yes | 73.08% | 100.81% | 3.72M |
| **178** | Finetune TrOCR (LLRD) | No | **12.38%** | 32.60% | 90.4M |
| **180** | Finetune TrOCR (LLRD) | Yes | **12.10%** | 32.31% | 90.4M |

### 2.3 Architecture Ablations (11 runs)

Baseline: ViT-RGTS v2, 4 registers, IAM, 150 epochs (run_146, CER=5.96%)

| Run | Ablation | Variable Changed | Test CER (best val) | Test WER | Params |
|-----|----------|-----------------|--------------------:|--------:|-------:|
| **157** | No CNN stem | use_cnn_stem=false | 73.36% | 112.88% | 9.25M |
| **158** | RNN head only | head_type=rnn | 7.07% | 22.46% | 9.42M |
| **159** | CNN head only | head_type=cnn | 9.47% | 30.76% | 5.23M |
| **160** | GRU head | rnn_type=gru | 7.13% | 23.21% | 8.43M |
| **161** | ViT augmentation | augmentation=vit | 6.30% | 20.37% | 9.48M |
| **162** | Depth=4 | depth=4 | 5.93% | 19.30% | 7.90M |
| **163** | Depth=8 | depth=8 | 5.93% | 19.25% | 5.47M |
| **164** | Dim=128 | dim=128 | 6.80% | 22.46% | 6.32M |
| **175** | Dim=512 | dim=512 | 5.79% | 19.00% | 11.1M |
| **165** | RNN 1 layer | rnn_layers=1 | 73.65% | 112.03% | 1.31M |
| **167** | Dropout=0.3 | dropout=0.3 | 6.68% | 21.37% | 9.48M |

### 2.4 SWA/LR/Seed/FT Ablations (11 runs)

| Run | Ablation | Detail | Test CER (best val) | Test WER |
|-----|----------|--------|--------------------:|--------:|
| **176** | SWA early | start_epoch=75 | 6.37% | 20.70% |
| **181** | SWA late | start_epoch=135 | 5.91% | 19.42% |
| **168** | Batch=16 | batch=16, no accum | 5.81% | 19.09% |
| **182** | FT ViT gradual unfreeze | gradual_unfreeze=true | 66.20% | 93.30% |
| **183** | FT TrOCR gradual unfreeze | gradual_unfreeze=true | 13.69% | 35.98% |
| **179** | ViT-RGTS v1 | legacy architecture | 6.57% | 20.85% |
| **186** | LR=5e-4 | base lr halved | 5.92% | 19.25% |
| **188** | SWA LR high | swa_lr=1e-3 | 5.81% | 19.06% |
| **170** | SWA LR low | swa_lr=1e-4 | 5.77% | 18.85% |
| **169** | Seed=123 | reproducibility | 5.55% | 18.26% |
| **171** | Seed=456 | reproducibility | 5.84% | 19.09% |

### 2.5 READ2016 Dataset Variation (5 runs)

| Run | Architecture | Registers | Test CER (best val) | Test WER | Params |
|-----|-------------|-----------|--------------------:|--------:|-------:|
| **166** | CNN-RNN | — | **4.06%** | 18.29% | 7.37M |
| **185** | ViT-RGTS v2 | 0 | 4.73% | 20.65% | 9.49M |
| **184** | ViT-RGTS v2 | 4 | 4.82% | 20.93% | 9.49M |
| **187** | ViT-RGTS v2 | 8 | 4.75% | 20.65% | 9.49M |
| **190** | ViT-RGTS v2 | 16 | 4.78% | 20.77% | 9.50M |

### 2.6 Synthetic Pretraining (16 runs — STILL RUNNING)

| Run | Architecture | Regs | SWA | Epoch | Test CER (so far) |
|-----|-------------|------|-----|-------|-----------------:|
| **189** | CNN-RNN | — | No | 10/15 | 34.70% |
| **191** | CNN-RNN | — | Yes | 9/15 | 32.16% |
| **192** | ViT-RGTS v2 | 0 | No | 9/15 | 30.31% |
| **193** | ViT-RGTS v2 | 0 | Yes | 9/15 | 30.63% |
| **194** | ViT-RGTS v2 | 2 | No | 8/15 | 30.15% |
| **195** | ViT-RGTS v2 | 4 | No | 8/15 | 31.60% |
| **196** | ViT-RGTS v2 | 2 | Yes | 8/15 | 31.70% |
| **197** | ViT-RGTS v2 | 4 | Yes | 8/15 | 31.42% |
| **198** | ViT-RGTS v2 | 8 | No | 8/15 | 30.62% |
| **199** | ViT-RGTS v2 | 16 | No | 7/15 | 30.97% |
| **200** | ViT-RGTS v2 | 8 | Yes | 8/15 | 31.04% |
| **201** | ViT-RGTS v2 | 16 | Yes | 7/15 | 31.61% |
| **202** | TorchVision ViT | 0 | No | 1/15 | 58.37% |
| **203** | TorchVision ViT | 0 | Yes | 1/15 | 57.81% |
| **204** | TrOCR | — | No | 2/15 | 77.00% |
| **205** | TrOCR | — | Yes | 2/15 | 77.32% |

*Note: Synthetic models are evaluated on IAM test set. High CER is expected since they are trained on synthetic data only (no fine-tuning on IAM). These numbers reflect domain gap, not model quality.*

---

## 3. Key Observations (Pre-Analysis)

1. **CNN-RNN is competitive**: Best IAM CER of 4.62% (run_145 with SWA), beating all ViT-RGTS v2 variants
2. **Register count has minimal effect on CER**: 0 regs (5.74%) vs 4 regs (5.77%) vs 16 regs (5.88%) — differences within noise
3. **CNN stem is critical**: Removing it collapses CER to 73% (run_157)
4. **RNN layers matter**: 1-layer RNN collapses to 73% (run_165); 3-layer LSTM is essential
5. **Dual head helps**: `both` (5.96%) > `rnn` (7.07%) > `cnn` (9.47%)
6. **Finetune TrOCR is best pretrained**: CER 12.1% but requires 90M params
7. **SWA has mixed results**: Some runs degraded at final epoch (BN recalibration issue visible in runs 147, 149, 156, 172)
8. **READ2016 generalizes**: Similar patterns, CNN-RNN at 4.06% CER
9. **Seed variance is small**: 5.55% / 5.96% / 5.84% across seeds 123/42/456

---

## 4. Post-Processing Pipeline — Exact Sequence

Execute these commands **after all synthetic jobs finish** (~2–4 hours from now).

### Step 1: Verify All Runs Complete

```bash
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline
source .venv/bin/activate

# Check no jobs still running
squeue -u $USER

# Count completed runs
ls saved_models/experiments/run_{143..205}/model.pt 2>/dev/null | wc -l
# Expected: 63
```

### Step 2: Aggregate Results into CSV + LaTeX Tables

```bash
python scripts/postprocessing/aggregate_results.py \
    --runs-dir saved_models/experiments \
    --output-dir outputs/tables
```

**Produces**:
- `outputs/tables/unified_results.csv` — all runs with CER/WER
- `outputs/tables/register_sweep.tex` — register ablation table
- `outputs/tables/ablations.tex` — architecture ablation table
- `outputs/tables/architecture_comparison.tex` — best per architecture
- `outputs/tables/all_runs.csv` — raw data including duplicates

### Step 3: Batch Evaluation (Re-evaluate all new runs uniformly)

```bash
bash evaluation_execution/batch_evaluate.sh --from 143
```

**Produces** per run: `saved_models/experiments/run_N/evaluation/`
- `evaluation_summary_test.json` — aggregate CER/WER/exact match
- `evaluation_report_test.txt` — worst/best samples analysis

### Step 4: Training Curves

```bash
python scripts/postprocessing/comparative/plot_training_metrics.py \
    --runs-dir saved_models/experiments \
    --output-dir outputs/report_figures
```

**Produces**: Training loss, CER, WER curves per architecture group

### Step 5: Attention Visualization (Key Runs Only)

Run on the most important runs for publication figures:

```bash
# Register comparison: 0 vs 4 vs 8 vs 16 registers
# Uses runs 144 (0reg), 146 (4reg), 151 (8reg), 152 (16reg)
python scripts/pub_fig2_register_comparison.py

# Per-character attention maps (Beyond-Memorization style)
# Run on best ViT-RGTS v2 model
python scripts/postprocessing/beyond_memorization_viz.py \
    --run-dir saved_models/experiments/run_146

# Attention quality bar charts (entropy, Gini, sparsity)
python scripts/pub_register_quantitative.py

# GradCAM analysis
python scripts/pub_fig3_gradcam_quantitative.py
```

**Produces**: Publication-quality figures in `outputs/report_figures/` and `outputs/pub_figures/`

### Step 6: Attention Quality Metrics (Quantitative)

```bash
python scripts/postprocessing/comparative/attention_quality_metrics.py \
    --runs-dir saved_models/experiments \
    --output-dir outputs/report_figures
```

**Produces**: Attention entropy, Gini sparsity, peak sharpness plots

### Step 7: Generate All Final Tables + Figures

```bash
bash evaluation_execution/generate_final_results.sh
```

This wraps Steps 2–6 into a single command.

---

## 5. Recommended Report Figures

| Figure # | Content | Source Runs | Script |
|----------|---------|------------|--------|
| 1 | Architecture diagram (ViT-RGTS v2) | — | Manual/TikZ |
| 2 | Register sweep CER bar chart | 144,148,146,151,152 | aggregate_results.py |
| 3 | Training curves (loss, CER vs epoch) | 143,146,153,155 | plot_training_metrics.py |
| 4 | Architecture comparison bar chart | 143,146,174,178 | aggregate_results.py |
| 5 | Per-character attention maps (0 vs 4 reg) | 144, 146 | beyond_memorization_viz.py |
| 6 | Attention entropy vs register count | 144,148,146,151,152 | attention_quality_metrics.py |
| 7 | Ablation tornado/sensitivity chart | 157–175 | Manual from ablations.tex |
| 8 | READ2016 vs IAM comparison | 166,184–190 | aggregate_results.py |
| 9 | SWA effect analysis | 146 vs 150, 188 | Manual from results |
| 10 | Seed variance error bars | 146,169,171 | Manual |

---

## 6. Recommended Report Tables

| Table # | Content | Data Source |
|---------|---------|-------------|
| 1 | Register sweep results (0/2/4/8/16 ± SWA) | register_sweep.tex |
| 2 | Architecture comparison (CNN-RNN, ViT-RGTS, ViT-B/16, TrOCR) | architecture_comparison.tex |
| 3 | Ablation study results | ablations.tex |
| 4 | READ2016 generalization | Manual from Section 2.5 |
| 5 | Reproducibility (seed variance) | Manual from Section 2.4 |
| 6 | Synthetic pretraining results | Manual from Section 2.6 |
| 7 | SWA timing sensitivity | Manual from Section 2.4 |

---

## 7. Monitoring Commands

```bash
# Check if synthetic jobs still running
squeue -u $USER

# Check a specific run's final metrics
tail -1 saved_models/experiments/run_N/results.csv

# Quick comparison of all new runs
python scripts/postprocessing/aggregate_results.py \
    --runs-dir saved_models/experiments --output-dir outputs/tables
cat outputs/tables/unified_results.csv | column -t -s,

# Check GPU hours consumed
sacct -u $USER --starttime=2026-08-29 \
    --format=JobID%10,JobName%25,State%12,Elapsed%12,ExitCode -n \
    | grep -v "\.\(batch\|extern\)" | grep COMPLETED
```

---

## 8. Known Issues & Notes

1. **SWA BN recalibration instability**: Several SWA runs show degraded final CER (runs 147, 149, 156, 172, 176, 186). The `test_cer_at_best` is reliable; `final_test_cer` is unreliable for SWA runs. **Use best-val-epoch CER for reporting.**

2. **TorchVision ViT-B/16 without fine-tuning** (run_153): CER 73% — the ImageNet→HTR domain gap is too large for this model without proper fine-tuning. Only the 2-group LR fine-tuned version (run_174, CER 15%) is meaningful.

3. **TrOCR frozen encoder** (runs 154, 155): CER ~72% — freezing the encoder and only training a CTC head on 3.7M params is insufficient. The LLRD fine-tuned version (run_178, CER 12.4%) works.

4. **Synthetic pretraining**: High CER (30–77%) on IAM test is expected — these models would need subsequent fine-tuning on IAM to be useful. The synthetic experiment demonstrates domain gap, not failure.

5. **Run numbering is not sequential by experiment type** due to concurrent SLURM scheduling. Use the mapping in Section 2 to identify runs.
