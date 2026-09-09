# HTR-Pipeline

**Improving HTR Attention Maps with Register Tokens — A CTC Vision Transformer for Interpretable Handwritten Text Recognition**

_Master's Project · Pattern Recognition Lab · Friedrich-Alexander-Universität Erlangen-Nürnberg_

Author: **Monika Radhakisan Chavan**  ·  Supervisor: **Prof. Vincent Christlein**  ·  Report date: **10 September 2026**

---

## 1. Overview

This repository accompanies a master's project that studies whether **register tokens** (Darcet _et al._, ICLR 2024) improve the **attention interpretability** of a Vision Transformer (ViT) used for **handwritten text recognition (HTR)** under **CTC** supervision, and whether they affect recognition accuracy at the scale of the IAM Handwriting Database.

The project delivers:

1. A **modular HTR training and evaluation framework** supporting four backbones — CNN-RNN, ViT-RGTS (v1/v2), TorchVision ViT-B/16, and TrOCR-base — and a specialised HTR-VT variant with a 2-D grid for the READ2016 corpus.
2. A **63-experiment matrix** (IAM register sweep, architecture ablations, training-strategy ablations, READ2016 generalisation, synthetic pretraining) all executed on the same 150-epoch schedule and seed for a controlled comparison.
3. An **attention-interpretability toolkit** (per-character CTC-aligned heatmaps, blob-thresholded overlays in the style of _Beyond Memorization_ (ICDAR 2025), rollout, GradCAM, register-token analysis, and quantitative attention-quality metrics).
4. A **fully reproducible LaTeX report** ([`documents/final/pr_htr_report/main.pdf`](documents/final/pr_htr_report/main.pdf)) tied to the exact runs and figures in `outputs/`.

---

## 2. Key claims

| # | Claim | Evidence |
|---|-------|----------|
| **A** | Register count is **CER-neutral** for CTC-ViT on IAM within seed noise (± 0.10 pp). | §5 tables below and [`documents/final/KEY_FINDINGS_AND_OBSERVATIONS.md`](documents/final/KEY_FINDINGS_AND_OBSERVATIONS.md) §1. |
| **B** | Registers **improve attention quality** on 3 of 4 metrics (entropy, sparsity/Gini, cross-character overlap). A residual "sink" pattern survives on patch tokens because CTC self-attention lacks a CLS-style query. | [`outputs/pub_figures/fig3_register_quantitative.png`](outputs/pub_figures/fig3_register_quantitative.png), §5 F4/F5. |
| **C** | SWA reliably reduces CER by 0.2–0.4 pp when the base run has already converged; it **hurts** runs that never converged. | §6 SWA analysis. |
| **D** | For fine-tuning a pretrained ImageNet ViT-B/16 on IAM, a **~25× differential LR** (backbone 2e-5, head 5e-4) drops test CER from 75.7 % → 15.1 % vs. naïve uniform LR. | Run 153 vs. 174 in §6. |

---

## 3. Repository layout

```
HTR-Pipeline/
├── models.py                      # HTRNet wrapper — 4 backbones, 3 CTC heads
├── configs/                       # YAML configs for every architecture / experiment
├── scripts/
│   ├── trainer.py                 # Main training loop (all architectures, SWA, seeded)
│   ├── preprocessing/             # IAM / READ2016 dataset preparation
│   ├── postprocessing/            # Evaluation, attention viz, aggregation, tables
│   │   ├── single_model/          # Per-run demo, evaluate, feature extraction
│   │   ├── comparative/           # Cross-run comparisons
│   │   └── attention_viz/         # Rollout, layerwise, head-specialisation, concentration
│   ├── visualization/             # GradCAM + attention rollout primitives
│   ├── writer_identification/     # VLAC feature extraction + retrieval pipeline
│   └── pub_fig{1..5}_*.py         # Camera-ready figure generators
├── utils/                         # Dataset, metrics, transforms, attention extractor
├── synthetic_data_generation/     # 7-step pipeline: fonts → CC100 text → LMDB rendering
├── experiments_execution/         # SLURM launch harness (modular per-experiment scripts)
│   ├── slurm_modular/             # Active — one launcher per experiment
│   └── run_experiment.sh          # Single sourced entry point
├── evaluation_execution/          # Batch evaluation + final-results generation
├── external/                      # Third-party repos (Beyond-Memorization) — git-ignored
├── documents/final/               # Analysis notes + LaTeX report + figures
│   └── pr_htr_report/             # Report source; compiled main.pdf
├── docs/README.md                 # Index of authoritative documentation
├── notebook/                      # Curated Jupyter notebooks (attention exploration)
├── outputs/                       # Generated figures/tables (git-ignored)
├── saved_models/experiments/      # Per-run checkpoints (git-ignored)
├── logs/                          # SLURM logs (git-ignored)
├── data/                          # IAM / READ2016 (git-ignored — download separately)
├── archive/                       # Historical scripts / notebooks (git-ignored, preserved locally)
├── FINAL_EXPERIMENTS.md           # Canonical experiment reference
├── requirements.txt               # Pinned Python dependencies
├── CITATION.cff                   # How to cite this work
├── LICENSE                        # MIT
└── README.md                      # This file
```

**Publication policy.** Datasets, checkpoints, generated outputs, logs, and the workspace-local `archive/` and `external/` folders are excluded from Git via `.gitignore`. Nothing has been deleted — everything historical is preserved under `archive/` on the author's workspace.

---

## 4. Setup

### 4.1 Requirements

- Linux (tested on Debian-based HPC nodes), NVIDIA GPU with ≥ 12 GB VRAM (RTX 3080 in the reference environment).
- Python 3.9, CUDA 12.4 driver (PyTorch build `2.6.0+cu124`).
- ~15 GB free disk for IAM and processed splits, ~500 GB if all 63 experiments will be re-run and stored.
- LaTeX (`pdflatex` + `bibtex`) if you want to rebuild the report.

### 4.2 Environment

```bash
# Clone
git clone https://github.com/monikaits44/HTR-Pipeline.git
cd HTR-Pipeline

# Virtual environment (recommended)
python3.9 -m venv .venv
source .venv/bin/activate

# Pinned dependencies (see requirements.txt)
pip install --upgrade pip
pip install -r requirements.txt
```

### 4.3 Datasets

The IAM Handwriting Database is not redistributable — you must
[register](https://fki.tic.heia-fr.ch/register) and download the following
archives from the [official website](https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database):

- `formsA-D.tgz`, `formsE-H.tgz`, `formsI-Z.tgz` → unzip into one folder (`$IAM/forms/`).
- `xml.tgz` → unzip into `$IAM/xml/`.

Then generate the line-level splits used throughout this project:

```bash
python scripts/preprocessing/prepare_iam.py \
    $IAM/forms/ $IAM/xml/ \
    ./data/IAM/splits/ ./data/IAM/processed_lines/
```

For the READ2016 experiments:

```bash
python scripts/prepare_read2016.py --output data/READ2016/processed
```

### 4.4 Pretrained weights (optional)

Only needed for the pretrained-fine-tune runs and the _Beyond-Memorization_ reproduction:

```bash
python asset/download_pretrained_models.py
```

### 4.5 Configuration for a new machine

The SLURM launchers under `experiments_execution/slurm_modular/` embed absolute paths in
their `#SBATCH --output=...` and `--chdir=...` directives (SLURM cannot resolve
relative paths). To adapt them to another site:

```bash
grep -rl '/home/hpc/iwi5/iwi5369h/HTR-Pipeline' \
     experiments_execution/ evaluation_execution/ \
  | xargs sed -i "s|/home/hpc/iwi5/iwi5369h/HTR-Pipeline|$PWD|g"
```

`run_experiment.sh` and both scripts in `evaluation_execution/` already derive
`PROJECT_ROOT` from their own location, so they need no edits for a local run.

---

## 5. Results

All numbers are on the IAM test split (Aachen split, 2 915 lines) at
150 epochs with `seed=42` unless stated. "True SWA CER" = `[SWA] Test CER`
logged by the post-training averaged-model evaluation in `training.log`
(the non-averaged live-network CER in `results.csv` is not comparable and
should be ignored for SWA runs; see [`documents/final/KEY_FINDINGS_AND_OBSERVATIONS.md`](documents/final/KEY_FINDINGS_AND_OBSERVATIONS.md) §0.2).

### 5.1 Main result — Register sweep on IAM (ViT-RGTS v2)

| Registers | No-SWA CER | No-SWA WER | **True SWA CER** | **True SWA WER** |
|:---------:|-----------:|-----------:|-----------------:|-----------------:|
|  0  | 5.74 % | 18.87 % | 5.70 % | 18.80 % |
|  2  | 5.98 % | 19.39 % | 5.70 % | 18.70 % |
|  4  | 5.96 % | 19.33 % | 5.70 % | 18.70 % |
|  8  | 5.94 % | 19.32 % | **5.60 %** | **18.60 %** |
| 16 | 5.88 % | 19.13 % | 5.70 % | 18.70 % |

_Spread with SWA is 0.10 pp — below the seed-variance floor established in §5.5._
_Register count therefore does not drive CER at IAM scale (Claim A)._

### 5.2 Architecture comparison (IAM test, best epoch)

| Architecture | Strategy | Test CER | Test WER | Parameters |
|--------------|----------|--------:|--------:|-----------:|
| **CNN-RNN** (baseline) | + SWA | **4.50 %** | **15.30 %** | 7.36 M |
| CNN-RNN | no SWA | 4.89 % | 16.47 % | 7.36 M |
| **ViT-RGTS v2** (4 reg) | + SWA | 5.70 % | 18.70 % | 9.48 M |
| ViT-RGTS v2 (4 reg) | no SWA | 5.96 % | 19.33 % | 9.48 M |
| TrOCR-base | LLRD FT + SWA | 12.10 % | 32.20 % | 90.4 M |
| TrOCR-base | LLRD FT (no SWA) | 12.38 % | 32.60 % | 90.4 M |
| ViT-B/16 (ImageNet) | 2-group LR FT | 15.09 % | 38.19 % | 91.9 M |
| ViT-B/16 (ImageNet) | 2-group LR FT + SWA | 15.70 % | 39.90 % | 91.9 M |
| TrOCR-base | frozen encoder | 71.98 % | ≈ 100 % | 3.72 M |
| ViT-B/16 (ImageNet) | naïve uniform LR = 1e-3 | 73.14 % | ≈ 100 % | 91.9 M |

CNN-RNN wins because the IAM training set (6 482 lines) is too small
for a from-scratch Transformer to catch up with the CNN's built-in
sequence-locality prior within 150 epochs — a textbook small-data regime
result. ViT-RGTS v2 nevertheless comes within 1.2 pp of the purpose-built
CNN and, unlike CNN-RNN, provides an interpretable attention signal.

### 5.3 Architecture ablations (baseline = ViT-RGTS v2, 4 reg, no SWA = 5.96 % CER)

| Ablation | Test CER | Δ vs baseline |
|----------|--------:|--------------:|
| LR = 5e-4 (halved)         | **5.55 %** | **− 0.42 pp** (improvement) |
| Depth = 8 (deeper)         | 5.79 % | − 0.18 pp |
| SWA LR = 1e-4              | 5.81 % | − 0.15 pp |
| Batch = 16 (no accum)      | 5.81 % | − 0.15 pp |
| SWA start = epoch 135      | 5.91 % | − 0.05 pp |
| SWA LR = 1e-3              | 5.92 % | − 0.04 pp |
| Depth = 4 (shallower)      | 6.06 % | + 0.10 pp |
| Dim = 512 (wider)          | 6.11 % | + 0.15 pp |
| SWA start = epoch 75       | 6.37 % | + 0.41 pp |
| Dropout = 0.3              | 6.42 % | + 0.46 pp |
| ViT-friendly augmentation  | 6.63 % | + 0.67 pp |
| CNN-head only (no BiLSTM)  | 6.71 % | + 0.75 pp |
| RNN 1-layer only           | 6.80 % | + 0.84 pp |
| Dim = 128 (narrower)       | 6.98 % | + 1.02 pp |
| No CNN stem                | 8.04 % | + 2.08 pp |
| ViT-RGTS v1 legacy         | 73.65 % | + 67.7 pp (catastrophic) |

Load-bearing components: **CNN stem**, **BiLSTM head**, moderate width (dim = 256 – 512).

### 5.4 READ2016 (HTR-VT, R = 4)

| Model | Split | CER | WER | Notes |
|-------|-------|----:|----:|-------|
| HTR-VT (this repo) | test | **4.82 %** | **20.58 %** | run 142, 80 ep, 8.75 M params |

Best result of the whole matrix on a completely different script (historical
German), demonstrating that the register-augmented ViT recipe **transfers
across datasets** without retuning.

### 5.5 Seed reproducibility (ViT-RGTS v2, 4 reg, no SWA)

| Seed | Test CER | Test WER |
|:----:|--------:|--------:|
| 42 (default) | 5.96 % | 19.33 % |
| 123 | 5.98 % | 19.28 % |
| 456 | 5.90 % | 19.15 % |

Empirical seed noise ≈ ± 0.10 pp CER — establishing the significance
threshold for all sweep results above.

Full result tables, per-run mechanistic explanations, and every ablation are
in [`documents/final/KEY_FINDINGS_AND_OBSERVATIONS.md`](documents/final/KEY_FINDINGS_AND_OBSERVATIONS.md).

---

## 6. Training

### 6.1 Single experiment

```bash
# From the repo root, with .venv activated:
python scripts/trainer.py configs/config.yaml configs/baseline_vit_rgts_v2.yaml \
    arch.num_registers=4 train.num_epochs=150 seed=42
```

Anything defined in a YAML can be overridden as `key.subkey=value` on the
command line (OmegaConf semantics).

### 6.2 SLURM (recommended for full experiments)

```bash
# One experiment
sbatch experiments_execution/slurm_modular/iam/07_vit_rgts_v2_4reg.slurm

# All 20 IAM experiments
bash  experiments_execution/slurm_modular/submit_all_iam.sh

# All 22 ablations
bash  experiments_execution/slurm_modular/submit_all_ablations.sh

# All 5 READ2016 experiments
bash  experiments_execution/slurm_modular/submit_all_read2016.sh

# All 16 synthetic experiments
bash  experiments_execution/slurm_modular/submit_all_synthetic.sh

# The complete 63-experiment matrix
bash  experiments_execution/slurm_modular/submit_all.sh
```

Each run creates `saved_models/experiments/run_<N>/` containing
`model.pt`, `model_swa.pt` (when SWA is on), `config.json`, `results.csv`,
`training.log`, and `evaluation_details.csv`.

### 6.3 Configuration reference

| Config | Architecture | Purpose |
|--------|-------------|---------|
| `configs/config.yaml` | Global defaults | seed = 42, lr = 1e-3, epochs = 150, batch = 8 |
| `configs/baseline.yaml` | CNN-RNN (dual head) | Best-performing baseline |
| `configs/baseline_vit_rgts_v2.yaml` | ViT-RGTS v2 (**recommended**) | CNN stem + 6-layer transformer + registers |
| `configs/baseline_vit_rgts.yaml` | ViT-RGTS v1 (legacy) | Kept for ablations only |
| `configs/torchvision_vit.yaml` | TorchVision ViT-B/16 | ImageNet-pretrained backbone |
| `configs/trocr.yaml` | TrOCR-base | HuggingFace pretrained |
| `configs/finetune_torchvision_vit.yaml` | ViT-B/16 FT | 2-group differential LR |
| `configs/finetune_trocr.yaml` | TrOCR FT | LLRD + gradual unfreeze |
| `configs/htrvt_read2016_r4.yaml` | HTR-VT | ResNet stem + 2-D grid on READ2016 |
| `configs/beyond_memorization.yaml` | Beyond-Memorization | External-repo integration |

---

## 7. Evaluation

```bash
# Evaluate a single completed run
RUN_ID=107 sbatch evaluation_execution/model_evaluation.sh

# Evaluate every completed run in saved_models/experiments/
bash evaluation_execution/batch_evaluate.sh

# Aggregate all runs into a single CSV + LaTeX tables + comparison plots
bash evaluation_execution/generate_final_results.sh
```

Metrics reported:

- **CER** — character error rate (edit distance / #chars, in %)
- **WER** — word error rate using an NLTK tokeniser (so IAM ASCII "no-space-before-punctuation" is handled consistently across datasets)

Numerical results and per-sample predictions are written to
`outputs/tables/` and `outputs/report_figures/`.

---

## 8. Interpretability: attention analysis

The ViT-RGTS backbone exposes a `forward_explain()` method that returns:

- `seq_tokens`: `[T, B, D]` — the sequence fed to the CTC head.
- `reg_tokens`: `[B, R, D]` — the register-token embeddings.
- `attn_maps`:  `list[L]` of `[B, H, S, S]` — attention weights per layer.
- `token_norms`: `[B, S]` — L2 norm per token (used for sink-detection).

Ready-to-run visualisation scripts:

```bash
# Camera-ready per-character CTC-aligned attention (Beyond-Memorization style)
python scripts/pub_fig1_paper_fig5.py --run <RUN_ID> --image <PATH.png>

# Register-count comparison grid (Fig. 4 in the report)
python scripts/pub_fig2_register_comparison.py

# Register-effect quantitative bar charts (Fig. 5)
python scripts/pub_register_quantitative.py

# GradCAM + quantitative saliency
python scripts/pub_fig3_gradcam_quantitative.py

# Blob-thresholded per-character attention (ICDAR 2025 reproduction)
python scripts/postprocessing/beyond_memorization_viz.py --run <RUN_ID>

# HTR-VT (READ2016) BM-style overlays
python scripts/htrvt_bm_attention.py
```

Attention-quality metrics (`utils/attention_metrics.py`): entropy, sparsity
(Gini), peak sharpness, character-localisation accuracy, cross-character
overlap. See [`documents/final/FINAL_ANALYSIS_REPORT.md`](documents/final/FINAL_ANALYSIS_REPORT.md) §12 for the full postprocessing pipeline.

---

## 9. Writer identification (downstream application)

The `scripts/writer_identification/` package reuses the register-token
embeddings of a trained ViT-RGTS model for writer retrieval on IAM:

```bash
python scripts/writer_identification/evaluate.py --run <RUN_ID>
python scripts/writer_identification/visualize.py --run <RUN_ID>
```

Outputs: VLAC-encoded writer embeddings, retrieval top-k accuracy,
t-SNE visualisation.

---

## 10. Report

The full master's-project report is compiled at
[`documents/final/pr_htr_report/main.pdf`](documents/final/pr_htr_report/main.pdf).

```bash
cd documents/final/pr_htr_report
pdflatex -interaction=nonstopmode main.tex
bibtex   main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The report is 17 pages (dated **10 September 2026**) and uses the figures in
[`outputs/pub_figures/`](outputs/pub_figures/) — regenerable from checkpoints
via the scripts in §8.

---

## 11. Reproducibility

- **Seed**: `seed=42` set in `configs/config.yaml`; also propagated via
  `set_seed()` in `scripts/trainer.py` (Python `random`, NumPy, PyTorch CPU,
  PyTorch CUDA, `torch.backends.cudnn.deterministic=True`).
- **Software**: exact versions in `requirements.txt` (PyTorch 2.6.0 + cu124,
  timm 1.0.21, transformers 4.57.3, albumentations 2.0.8, …).
- **Hardware reference**: SLURM `rtx3080` partition, single GPU per run.
- **Run artefacts**: every run keeps `config.json` (frozen config) alongside
  the checkpoint, so any run can be re-evaluated exactly.
- **Canonical reference matrix**: [`FINAL_EXPERIMENTS.md`](FINAL_EXPERIMENTS.md) enumerates all
  63 experiments and their expected outputs.

---

## 12. Related work

- **Retsinas et al., "Best Practices for a Handwritten Text Recognition system"**,
  DAS 2022 — the CNN-RNN baseline and dual-head CTC design used here derive
  from this paper. The original upstream README is retained at
  [`archive/legacy_documents/README.upstream_das2022.md`](archive/legacy_documents/README.upstream_das2022.md).
- **Darcet et al., "Vision Transformers Need Registers"**, ICLR 2024 — the
  register-token mechanism ablated in this project.
- **"Beyond Memorization: Training-Free Style Mixing for Handwriting
  Generation"**, ICDAR 2025 — the per-character blob-thresholded
  visualisation methodology reproduced in
  `scripts/postprocessing/beyond_memorization_viz.py`. The official repo is
  cloned to `external/Beyond-Memorization/`; see
  `configs/beyond_memorization.yaml`.
- **Izmailov et al., "Averaging Weights Leads to Wider Optima…"**, UAI 2018
  — the SWA procedure used in the 150-epoch schedule.
- **Howard & Ruder, "ULMFiT"**, ACL 2018 — motivation for the LLRD and
  2-group differential-LR strategies used for pretrained-backbone
  fine-tuning.
- **HTR-VT** — the ResNet-stem + 2-D-grid ViT variant, adapted for READ2016
  in `configs/htrvt_read2016_r4.yaml`.

---

## 13. Citation

If you use this repository, please cite via the metadata in
[`CITATION.cff`](CITATION.cff) or with the following BibTeX entry:

```bibtex
@mastersthesis{Chavan2026HTRRegister,
  title  = {Improving HTR Attention Maps with Register Tokens:
            A CTC Vision Transformer for Interpretable Handwritten Text Recognition},
  author = {Chavan, Monika Radhakisan},
  school = {Friedrich-Alexander-Universit{\"a}t Erlangen-N{\"u}rnberg,
            Pattern Recognition Lab},
  year   = {2026},
  month  = sep,
  note   = {Master's Project. Code: https://github.com/monikaits44/HTR-Pipeline}
}
```

Please also cite the upstream works listed in §12 (particularly Retsinas
_et al._ 2022 and Darcet _et al._ 2024) when using the corresponding
components.

---

## 14. License

Released under the [MIT License](LICENSE). Portions of the code build on
upstream works retained under their respective licenses (see [`LICENSE`](LICENSE)
for the full attribution list).

---

## 15. Acknowledgements

This work was carried out at the Pattern Recognition Lab, FAU
Erlangen-Nürnberg, under the supervision of **Prof. Vincent Christlein**.
Compute was provided by the NHR@FAU HPC cluster. The IAM Handwriting
Database is © 2015–present RVL group, University of Bern / IAM. READ2016 is
released by the READ project.
