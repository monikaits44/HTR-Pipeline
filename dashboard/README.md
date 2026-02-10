# ✍️ HTR Pipeline — Experiment Dashboard

Interactive Streamlit dashboard for tracking, comparing, and analyzing Handwritten Text Recognition experiments on the IAM dataset.

---

## 🚀 Quick Start

```bash
# 1. Navigate to project root
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Install dashboard dependencies (one-time)
pip install streamlit plotly

# 4. Launch the dashboard
streamlit run dashboard/app.py --server.port 8501

# 5. On an HPC cluster, forward the port via SSH:
#    ssh -L 8501:localhost:8501 <user>@<cluster>
#    Then open http://localhost:8501 in your browser
```

### Alternative: Running on a login node with tunneling

```bash
# On the cluster:
streamlit run dashboard/app.py --server.port 8501 --server.headless true

# On your local machine:
ssh -L 8501:localhost:8501 iwi5369h@tinygpu.rrze.fau.de
# Then open http://localhost:8501
```

---

## 📋 Features

### Tab 1: 📊 Overview
- **Top-level KPIs** — Total experiments, best CER/WER across all runs
- **Per-architecture best results** — Color-coded metric cards (CNN-RNN, ViT-RGTS, TorchVision ViT, TrOCR)
- **CER bar chart** — Visual comparison of best test CER across all runs
- **Full experiment table** — Sortable table with architecture, params, CER, WER, CTC loss

### Tab 2: 📈 Training Curves
- **Interactive multi-run comparison** — Select any combination of runs to overlay
- **Metrics:** CTC Loss, Validation/Test CER, Validation/Test WER, Learning Rate schedule
- **Log scale toggle** — Useful for comparing loss curves across architectures
- **Pre-generated plots** — Static training plots from `output/training_plots/`

### Tab 3: 🔬 Architecture Lab
- **ViT-RGTS register comparison** — CER by register count (0, 2, 4, 8, 16), convergence curves
- **Cross-architecture scatter** — CER vs model size (parameters), bubble size = epochs trained
- **Radar chart** — Normalized comparison of CER, WER, and CTC loss for best run per architecture

### Tab 4: 🔍 Run Explorer
- **Deep-dive into any run** — Full config JSON, training summary, epoch-by-epoch metrics
- **Per-sample error analysis** — CER distribution histogram, CER vs text length scatter
- **Worst/best predictions** — Side-by-side ground truth vs prediction for hardest and easiest samples
- **Run artifacts** — All files in the run directory, including visualizations
- **Training log viewer** — Tail the last N lines of training.log

### Tab 5: 🧠 Attention Maps
- **Character attention** — ViT self-attention overlaid on input showing per-character focus
- **Register analysis** — How register tokens attend across different configurations
- **Grad-CAM** — Gradient-weighted activation maps highlighting model focus regions
- **Run-specific attention** — Browse attention visualizations saved during training

### Tab 6: 📚 Dataset EDA
- **IAM dataset statistics** — Sample counts, split distribution, character/word counts
- **Character frequency** — Top 20 most frequent characters with interactive bar chart
- **Writer distribution** — Number of unique writers contributing samples
- **Pre-generated EDA plots** — Text length, image dimensions, character frequency, writer analysis

### Tab 7: 📡 Monitoring
- **Live SLURM queue** — Current job status from `squeue`
- **Log viewer** — Browse and tail output/error logs from baseline, ViT-RGTS, and pretrained experiments
- **Recently modified runs** — Quick glance at which runs were updated most recently

---

## 🏗️ Data Sources

The dashboard automatically reads data from these project locations:

| Source | Path | Content |
|--------|------|---------|
| Experiment runs | `saved_models/experiments/run_*/` | Config, results CSV, evaluation details, model checkpoints |
| Training plots | `output/training_plots/` | Pre-generated CTC loss, CER, WER, LR plots |
| Data analysis | `output/data_analysis/` | EDA summary JSON, character/length/writer distribution plots |
| Visualizations | `visualizations/` | Character attention, register analysis, Grad-CAM images |
| SLURM logs | `experiments_execution/logs/` | Output and error logs from cluster jobs |

### Per-Run File Format

Each `run_*` directory contains:

```
run_32/
├── config.json              # Full experiment configuration (OmegaConf dump)
├── results.csv              # Epoch-level metrics (17 columns)
├── evaluation_details.csv   # Per-sample predictions and CER/WER
├── model.pt                 # PyTorch model state dict (latest checkpoint)
├── training.log             # Full console output
├── character_attention.png  # (ViT runs only) Character-level attention
├── register_attention.png   # (ViT runs only) Register token attention
├── token_norms.png          # (ViT runs only) Token norm distributions
├── tsne_registers.png       # (ViT runs only) t-SNE of register embeddings
├── gradcam_output.png       # (ViT runs only) Grad-CAM visualization
└── vit_rgts_explain/        # (ViT runs only) Raw .npy attention data
```

### results.csv Columns

| Column | Description |
|--------|-------------|
| `epoch` | Training epoch number |
| `lr` | Current learning rate |
| `train/ctc_loss` | Average CTC loss over training batches |
| `val/cer` | Character Error Rate on validation set |
| `val/wer` | Word Error Rate on validation set |
| `test/cer` | Character Error Rate on test set |
| `test/wer` | Word Error Rate on test set |
| `data/train_lines` | Number of training samples (6,482) |
| `data/val_lines` | Number of validation samples (976) |
| `data/test_lines` | Number of test samples (2,915) |
| `data/train_charset_size` | Unique characters in training set (79) |
| `model/params` | Total trainable parameters |
| `time/epoch(s)` | Wall-clock time per epoch in seconds |
| `device` | GPU/CPU device used |
| `seed` | Random seed (-1 if unset) |

---

## 🎨 Architecture Color Coding

| Architecture | Color | Description |
|-------------|-------|-------------|
| CNN-RNN | 🔵 Blue (#2E86C1) | Baseline CNN backbone + BiLSTM head |
| ViT-RGTS | 🔴 Red (#E74C3C) | Vision Transformer with Register Tokens (from scratch) |
| TorchVision ViT | 🟢 Green (#27AE60) | Pretrained ViT-B/16 from torchvision |
| TrOCR | 🟣 Purple (#8E44AD) | Pretrained TrOCR encoder from HuggingFace |

---

## ⚙️ Configuration

The dashboard auto-discovers all experiments in `saved_models/experiments/`. No additional configuration is needed.

### Caching

Data is cached for 60 seconds (overview, configs, results) or 300 seconds (evaluation details, EDA). Click **🔄 Refresh Data** in the sidebar to clear caches manually.

### Custom Port

```bash
streamlit run dashboard/app.py --server.port <PORT>
```

---

## 📦 Dependencies

```
streamlit>=1.28.0
plotly>=5.15.0
pandas>=1.5.0
numpy>=1.24.0
Pillow>=9.0.0
```

These are in addition to the project's existing PyTorch, torchvision, and transformers dependencies.

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: streamlit` | Run `pip install streamlit plotly` |
| Dashboard shows no experiments | Check that `saved_models/experiments/run_*/config.json` exists |
| SLURM queue empty | Normal if no jobs are running; the monitoring tab handles this gracefully |
| Images not loading | Ensure visualization scripts have been run (see `scripts/postprocessing/`) |
| Port already in use | Use `--server.port <different_port>` |
| Running on headless server | Add `--server.headless true` flag |
