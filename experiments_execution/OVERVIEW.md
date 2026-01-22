# Experiments Execution System - Overview

## 🎯 What This System Does

This structured execution system automates running all HTR experiment variations with:
- ✅ **8 pre-configured SLURM scripts** for different model configurations
- ✅ **Organized log files** separated by experiment category
- ✅ **Master submission scripts** for batch or individual execution
- ✅ **Jupyter notebook** for real-time tracking and analysis
- ✅ **Automatic timestamping** and job identification

## 📊 Experiment Coverage

### Complete Experiment Matrix

| # | Experiment | Config | Registers | Expected Time | Log Location |
|---|------------|--------|-----------|---------------|--------------|
| 1 | Baseline CNN-RNN | baseline.yaml | N/A | ~3-4h | logs/baseline/ |
| 2 | ViT-RGTS | baseline_vit_rgts.yaml | 0 | ~3.5-4.5h | logs/vit_rgts_registers/ |
| 3 | ViT-RGTS | baseline_vit_rgts.yaml | 2 | ~3.5-4.5h | logs/vit_rgts_registers/ |
| 4 | ViT-RGTS | baseline_vit_rgts.yaml | 4 | ~3.5-4.5h | logs/vit_rgts_registers/ |
| 5 | ViT-RGTS | baseline_vit_rgts.yaml | 8 | ~3.5-4.5h | logs/vit_rgts_registers/ |
| 6 | ViT-RGTS | baseline_vit_rgts.yaml | 16 | ~3.5-4.5h | logs/vit_rgts_registers/ |
| 7 | TorchVision ViT-B/16 | torchvision_vit.yaml | 4 | ~4-5h | logs/pretrained/ |
| 8 | TrOCR Base | trocr.yaml | N/A | ~6-8h | logs/pretrained/ |

**Total: 8 experiments | Sequential Time: ~35-40 hours | Parallel Time: ~6-8 hours**

## 🗂️ Directory Structure Explained

```
experiments_execution/
│
├── slurm_scripts/                    # All SLURM job scripts
│   ├── 01_baseline.slurm            # Baseline CNN-RNN
│   ├── 02_vit_rgts_0reg.slurm       # ViT without register tokens
│   ├── 03_vit_rgts_2reg.slurm       # ViT with 2 register tokens
│   ├── 04_vit_rgts_4reg.slurm       # ViT with 4 register tokens (recommended)
│   ├── 05_vit_rgts_8reg.slurm       # ViT with 8 register tokens
│   ├── 06_vit_rgts_16reg.slurm      # ViT with 16 register tokens
│   ├── 07_torchvision_vit_b16.slurm # Pretrained TorchVision ViT
│   └── 08_trocr_base.slurm          # Pretrained TrOCR model
│
├── logs/                             # All experiment logs (auto-organized)
│   ├── baseline/                    # output_<jobid>.log, error_<jobid>.log
│   ├── vit_rgts_registers/          # output_<N>reg_<jobid>.log (0,2,4,8,16)
│   └── pretrained/                  # output_<model>_<jobid>.log
│
├── submit_all_experiments.sh         # Master script (3 modes: sequential/parallel/manual)
├── submit_single.sh                  # Submit individual experiments (1-8)
├── experiment_tracker.ipynb          # Real-time tracking & analysis notebook
├── README.md                         # Comprehensive documentation
└── QUICK_REFERENCE.txt               # Command cheat sheet
```

## 🚀 Usage Examples

### Example 1: Run All Experiments Sequentially

```bash
cd experiments_execution
./submit_all_experiments.sh
# Select option 1 (Sequential)
```

**What happens:**
- Job 1 (Baseline) starts immediately
- Job 2 (ViT-0reg) waits for Job 1 to complete
- Job 3 (ViT-2reg) waits for Job 2 to complete
- ... and so on

**Benefit:** Guaranteed resource availability, no queue competition

### Example 2: Run All Experiments in Parallel

```bash
cd experiments_execution
./submit_all_experiments.sh
# Select option 2 (Parallel)
```

**What happens:**
- All 8 jobs submitted simultaneously
- Jobs run as GPUs become available
- Faster completion if resources available

**Benefit:** Fastest total completion time (if cluster has capacity)

### Example 3: Run Specific Experiment

```bash
cd experiments_execution
./submit_single.sh 4  # ViT-RGTS with 4 registers
```

**What happens:**
- Only Job 4 is submitted
- Runs immediately when GPU available

**Benefit:** Test specific configuration quickly

### Example 4: Monitor Progress

```bash
# Check job queue
squeue -u $USER

# Monitor logs in real-time
tail -f logs/vit_rgts_registers/output_4reg_*.log

# Or use the notebook
jupyter notebook experiment_tracker.ipynb
```

## 📝 Log File Naming Convention

### Pattern: `<type>_<config>_<jobid>.log`

**Examples:**
```
logs/baseline/output_12345.log                    # Baseline output, job 12345
logs/baseline/error_12345.log                     # Baseline errors, job 12345

logs/vit_rgts_registers/output_0reg_12346.log     # ViT-RGTS 0 registers, job 12346
logs/vit_rgts_registers/output_4reg_12348.log     # ViT-RGTS 4 registers, job 12348
logs/vit_rgts_registers/error_4reg_12348.log      # ViT-RGTS 4 reg errors

logs/pretrained/output_vit_b16_12350.log          # TorchVision ViT, job 12350
logs/pretrained/output_trocr_base_12351.log       # TrOCR, job 12351
```

**Benefits:**
- Easy identification of experiment type
- Chronological sorting by job ID
- Quick search for specific configurations
- Separate error tracking

## 🎨 Workflow Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│                    BEFORE SUBMISSION                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. Review configs: configs/baseline*.yaml, trocr.yaml   │   │
│  │ 2. Check resources: sinfo -p rtx3080                    │   │
│  │ 3. Verify dataset: data/IAM/processed_lines/            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    SUBMIT EXPERIMENTS                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ ./submit_all_experiments.sh                              │   │
│  │   → Sequential: Jobs run one after another              │   │
│  │   → Parallel: All jobs at once                          │   │
│  │   → Manual: Interactive confirmation                    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    JOBS RUNNING                                 │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐      │
│  │ Job 1    │ Job 2    │ Job 3    │ Job 4    │ Job 5-8  │      │
│  │ Baseline │ ViT-0reg │ ViT-2reg │ ViT-4reg │   ...    │      │
│  │ ✓ Done   │ Running  │ Queued   │ Queued   │ Queued   │      │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘      │
│                                                                 │
│  Logs streaming to: experiments_execution/logs/                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MONITORING                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ • squeue -u $USER            (job status)               │   │
│  │ • tail -f logs/*/output_*.log (live logs)               │   │
│  │ • experiment_tracker.ipynb   (analysis)                 │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    RESULTS                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ saved_models/experiments/                                │   │
│  │   ├── run_X/model.pt  ← Baseline                        │   │
│  │   ├── run_Y/model.pt  ← ViT-RGTS (0 reg)                │   │
│  │   ├── run_Z/model.pt  ← ViT-RGTS (4 reg)                │   │
│  │   └── ...                                                 │   │
│  │                                                           │   │
│  │ logs/                                                     │   │
│  │   ├── baseline/output_*.log                              │   │
│  │   ├── vit_rgts_registers/output_*reg_*.log               │   │
│  │   └── pretrained/output_*.log                            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    ANALYSIS                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ python scripts/postprocessing/analyze_evaluation.py \    │   │
│  │     saved_models/experiments/                            │   │
│  │                                                           │   │
│  │ Results:                                                  │   │
│  │   • CER/WER comparison across all models                 │   │
│  │   • Register token impact visualization                  │   │
│  │   • Best configuration identification                    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 🔍 Key Features

### 1. Automatic Organization
- Logs automatically sorted by experiment type
- Unique job IDs prevent file conflicts
- Clear naming convention for easy identification

### 2. Flexible Execution
- **Sequential Mode**: Safe, resource-conscious
- **Parallel Mode**: Fast, requires available GPUs
- **Manual Mode**: Full control over each submission

### 3. Easy Monitoring
- Real-time log viewing
- Job status tracking
- Jupyter notebook for analysis

### 4. Comprehensive Coverage
- All baseline variations
- Complete register token sweep (0, 2, 4, 8, 16)
- Pretrained model comparison

## 📚 Additional Documentation

| File | Purpose |
|------|---------|
| [README.md](README.md) | Full documentation with commands and troubleshooting |
| [QUICK_REFERENCE.txt](QUICK_REFERENCE.txt) | Quick command reference card |
| [experiment_tracker.ipynb](experiment_tracker.ipynb) | Interactive tracking and analysis |

## 🎓 Research Value

This system enables systematic study of:

1. **Register Token Impact**: Compare 0, 2, 4, 8, 16 registers
2. **Architecture Comparison**: Baseline vs ViT-RGTS vs Pretrained
3. **Explainability Analysis**: Attention map quality across models
4. **Performance Trade-offs**: CER/WER vs model complexity

## ✨ Benefits Over Manual Execution

| Manual Approach | This System |
|-----------------|-------------|
| One script at a time | Automated batch submission |
| Logs scattered everywhere | Organized by category |
| Hard to track progress | Real-time monitoring |
| Manual result collection | Automated tracking notebook |
| Prone to errors | Pre-configured & tested |
| No experiment history | Complete log archive |

## 🎯 Next Steps

1. **Verify Setup**
   ```bash
   cd experiments_execution
   ls -la  # Check all files are present
   ```

2. **Test Single Job**
   ```bash
   ./submit_single.sh 4  # Test with ViT-RGTS 4 registers
   ```

3. **Monitor Test**
   ```bash
   squeue -u $USER
   tail -f logs/vit_rgts_registers/output_4reg_*.log
   ```

4. **Run Full Suite**
   ```bash
   ./submit_all_experiments.sh  # Choose Sequential mode
   ```

5. **Analyze Results**
   ```bash
   jupyter notebook experiment_tracker.ipynb
   ```

---

**Ready to run structured experiments with organized logging! 🚀**
