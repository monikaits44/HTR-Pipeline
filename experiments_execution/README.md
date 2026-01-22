# Experiments Execution System

This directory contains a structured system for running all HTR experiment variations with organized logging and tracking.

## 📁 Directory Structure

```
experiments_execution/
├── slurm_scripts/              # Individual SLURM job scripts
│   ├── 01_baseline.slurm
│   ├── 02_vit_rgts_0reg.slurm
│   ├── 03_vit_rgts_2reg.slurm
│   ├── 04_vit_rgts_4reg.slurm
│   ├── 05_vit_rgts_8reg.slurm
│   ├── 06_vit_rgts_16reg.slurm
│   ├── 07_torchvision_vit_b16.slurm
│   └── 08_trocr_base.slurm
├── logs/                       # Organized log files
│   ├── baseline/              # Baseline model logs
│   ├── vit_rgts_registers/    # ViT-RGTS variation logs
│   └── pretrained/            # Pretrained model logs
├── submit_all_experiments.sh   # Master submission script
├── submit_single.sh           # Single experiment submission
├── experiment_tracker.ipynb   # Jupyter notebook for tracking
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Submit All Experiments

```bash
cd experiments_execution
chmod +x submit_all_experiments.sh
./submit_all_experiments.sh
```

You'll be prompted to choose:
- **Sequential**: Jobs run one after another (recommended for resource constraints)
- **Parallel**: All jobs submitted at once (faster if resources available)
- **Manual**: Confirm each job individually

### 2. Submit Single Experiment

```bash
cd experiments_execution
chmod +x submit_single.sh

# Submit specific experiment (1-8)
./submit_single.sh 4  # ViT-RGTS with 4 registers
```

Available experiments:
1. Baseline CNN-RNN
2. ViT-RGTS (0 registers)
3. ViT-RGTS (2 registers)
4. ViT-RGTS (4 registers)
5. ViT-RGTS (8 registers)
6. ViT-RGTS (16 registers)
7. TorchVision ViT-B/16
8. TrOCR Base

## 📊 Monitoring & Tracking

### Check Job Status

```bash
# View all your jobs
squeue -u $USER

# Watch job queue in real-time
watch -n 5 squeue -u $USER
```

### View Logs in Real-Time

```bash
# Monitor all output logs
tail -f logs/*/output_*.log

# Monitor specific category
tail -f logs/vit_rgts_registers/output_*.log

# View error logs
tail -f logs/*/error_*.log
```

### Use Tracking Notebook

Open `experiment_tracker.ipynb` to:
- View job status
- Monitor logs
- Analyze results
- Compare architectures
- Visualize register token impact

```bash
jupyter notebook experiment_tracker.ipynb
```

## 📝 Log File Organization

Logs are automatically organized by experiment type:

### Baseline Logs
```
logs/baseline/
├── output_<job_id>.log
└── error_<job_id>.log
```

### ViT-RGTS Register Variations
```
logs/vit_rgts_registers/
├── output_0reg_<job_id>.log
├── output_2reg_<job_id>.log
├── output_4reg_<job_id>.log
├── output_8reg_<job_id>.log
└── output_16reg_<job_id>.log
```

### Pretrained Models
```
logs/pretrained/
├── output_vit_b16_<job_id>.log
└── output_trocr_base_<job_id>.log
```

## 🔧 Configuration

Each SLURM script contains:
- **Job name**: Descriptive identifier
- **Output/Error paths**: Organized by category
- **Time limit**: 4.5-8 hours depending on model
- **GPU**: 1 GPU (rtx3080 partition)
- **CPUs**: 8 cores

### Modify Resource Allocation

Edit individual `.slurm` files to change:

```bash
#SBATCH --time=06:00:00          # Increase time limit
#SBATCH --gres=gpu:2              # Request 2 GPUs
#SBATCH --cpus-per-task=16       # More CPU cores
#SBATCH --partition=rtx3090      # Different partition
```

## 🎯 Experiment Variations

### Baseline
- CNN-RNN-CTC architecture
- Dual-head output (CTC + attention)

### ViT-RGTS (5 variations)
- 0 registers: Standard ViT
- 2 registers: Minimal configuration
- 4 registers: Recommended (paper default)
- 8 registers: Enhanced
- 16 registers: Maximum

### Pretrained Models
- TorchVision ViT-B/16 (86M params)
- TrOCR Base Handwritten (334M params)

## 📈 Expected Timeline

Approximate training times (on RTX 3080):
- Baseline CNN-RNN: ~3-4 hours
- ViT-RGTS: ~3.5-4.5 hours
- TorchVision ViT: ~4-5 hours
- TrOCR: ~6-8 hours

**Total sequential time**: ~35-40 hours for all experiments

## 🛠️ Useful Commands

### Job Management

```bash
# Cancel specific job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER

# Job details
scontrol show job <job_id>

# Job efficiency report (after completion)
seff <job_id>
```

### Log Analysis

```bash
# Count total lines in output log
wc -l logs/vit_rgts_registers/output_4reg_*.log

# Search for specific metric in logs
grep "CER:" logs/*/output_*.log

# Find errors in logs
grep -i "error" logs/*/error_*.log

# Check last 100 lines of latest log
tail -n 100 $(ls -t logs/*/output_*.log | head -1)
```

### Results Summary

```bash
# List all completed experiments
ls -lh ../saved_models/experiments/run_*/model.pt

# View specific experiment config
cat ../saved_models/experiments/run_10/config.json

# Quick CER comparison
for run in ../saved_models/experiments/run_*; do
    if [ -f "$run/evaluation_details.csv" ]; then
        echo "$run:"
        tail -1 "$run/evaluation_details.csv"
    fi
done
```

## 🔍 Troubleshooting

### Job Not Starting
- Check partition availability: `sinfo -p rtx3080`
- Check job reason: `squeue -u $USER -o "%.18i %.20j %.8T %.10M %.20R"`

### Out of Memory
- Reduce batch size in config files
- Request more memory: `#SBATCH --mem=32G`

### Job Timeout
- Increase time limit in SLURM script
- Check current usage: `squeue -u $USER -o "%.18i %.20j %.10M %.9l"`

### Logs Not Generated
- Check directory permissions: `ls -la logs/`
- Verify SLURM script paths are correct
- Ensure parent directories exist

## 📊 Post-Experiment Analysis

After experiments complete, use these tools:

```bash
# Evaluate all models
cd ..
for run in saved_models/experiments/run_*; do
    python scripts/postprocessing/evaluate.py \
        $run/config.json \
        resume=$run/model.pt
done

# Compare results
python scripts/postprocessing/analyze_evaluation.py saved_models/experiments/

# Generate visualizations
python scripts/postprocessing/visualize_register_attention.py \
    saved_models/experiments/run_X/vit_rgts_explain
```

## 📚 Additional Resources

- Main documentation: `../README.md`
- Quick start guide: `../QUICK_START_GUIDE.md`
- Architecture details: `../documents/IMPLEMENTATION_SUMMARY.md`
- Explainability guide: `../documents/EXPLAINABILITY_README.md`

## ✅ Checklist Before Running

- [ ] Virtual environment activated
- [ ] IAM dataset preprocessed
- [ ] Config files reviewed
- [ ] Sufficient disk space (check: `df -h`)
- [ ] GPU partition available (check: `sinfo`)
- [ ] Scripts have execute permissions (check: `ls -l *.sh`)
- [ ] Log directories exist (auto-created)

## 🎉 Ready to Run!

Start your experiments with:

```bash
cd experiments_execution
./submit_all_experiments.sh
```

Good luck with your research! 🚀
