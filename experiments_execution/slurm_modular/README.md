# Modular SLURM Experiment System

## Architecture

```
slurm_modular/
├── run_experiment.sh          # Shared runner (all scripts source this)
├── submit_all.sh              # Submit ALL 63 experiments
├── submit_all_iam.sh          # Submit all 20 IAM experiments
├── submit_all_synthetic.sh    # Submit all 16 synthetic experiments
├── submit_all_ablations.sh    # Submit all 22 ablation experiments
├── submit_all_read2016.sh     # Submit all 5 READ2016 experiments
├── submit_v2_register_sweep.sh # Submit only ViT-RGTS v2 register sweep
├── submit_single.sh           # Submit by name/pattern
│
├── iam/                       # Train + eval on IAM (Aachen splits)
│   ├── 01_cnn_rnn.slurm               # CNN-RNN baseline
│   ├── 02_cnn_rnn_swa.slurm           # CNN-RNN + SWA
│   ├── 03_vit_rgts_v2_0reg.slurm      # ViT-RGTS v2, 0 registers
│   ├── 04_vit_rgts_v2_0reg_swa.slurm  # ViT-RGTS v2, 0 registers + SWA
│   ├── 05_vit_rgts_v2_2reg.slurm      # ViT-RGTS v2, 2 registers
│   ├── 06_vit_rgts_v2_2reg_swa.slurm  # ViT-RGTS v2, 2 registers + SWA
│   ├── 07_vit_rgts_v2_4reg.slurm      # ViT-RGTS v2, 4 registers ⭐
│   ├── 08_vit_rgts_v2_4reg_swa.slurm  # ViT-RGTS v2, 4 registers + SWA
│   ├── 09_vit_rgts_v2_8reg.slurm      # ViT-RGTS v2, 8 registers
│   ├── 10_vit_rgts_v2_8reg_swa.slurm  # ViT-RGTS v2, 8 registers + SWA
│   ├── 11_vit_rgts_v2_16reg.slurm     # ViT-RGTS v2, 16 registers
│   ├── 12_vit_rgts_v2_16reg_swa.slurm # ViT-RGTS v2, 16 registers + SWA
│   ├── 13_torchvision_vit_b16.slurm   # ViT-B/16 pretrained
│   ├── 14_torchvision_vit_b16_swa.slurm
│   ├── 15_finetune_vit_b16.slurm      # ViT-B/16 fine-tune (2-group LR)
│   ├── 16_finetune_vit_b16_swa.slurm
│   ├── 17_trocr_base.slurm            # TrOCR (encoder frozen)
│   ├── 18_trocr_base_swa.slurm
│   ├── 19_finetune_trocr.slurm        # TrOCR fine-tune (LLRD)
│   └── 20_finetune_trocr_swa.slurm
│
├── synthetic/                 # Train on synthetic, test on IAM
│   ├── 01_cnn_rnn.slurm               # CNN-RNN
│   ├── 02_cnn_rnn_swa.slurm
│   ├── 03-12: ViT-RGTS v2 sweep (0/2/4/8/16 reg × ±SWA)
│   ├── 13_torchvision_vit_b16.slurm
│   ├── 14_torchvision_vit_b16_swa.slurm
│   ├── 15_trocr_base.slurm
│   └── 16_trocr_base_swa.slurm
│
└── ablations/                 # Architectural ablation studies
    ├── 01_no_cnn_stem.slurm           # Remove CNN stem → raw patch embed
    ├── 02_rnn_head_only.slurm         # CTCtopR only (no CNN shortcut)
    ├── 03_cnn_head_only.slurm         # CTCtopC only (linear projection)
    ├── 04_gru_head.slurm              # BiGRU vs BiLSTM
    ├── 05_vit_augmentation.slurm      # Strong ViT aug vs moderate CNN aug
    ├── 06_depth4.slurm                # Transformer depth=4 (shallower)
    ├── 07_depth8.slurm                # Transformer depth=8 (deeper)
    ├── 08_dim128.slurm                # dim=128 (smaller, like v1)
    ├── 09_dim512.slurm                # dim=512 (larger)
    ├── 10_rnn1_layer.slurm            # 1-layer BiLSTM (lighter head)
    ├── 11_dropout03.slurm             # Higher dropout (0.3 vs 0.1)
    ├── 12_swa_early.slurm             # SWA start at 50% (epoch 75)
    ├── 13_swa_late.slurm              # SWA start at 90% (epoch 135)
    ├── 14_batch16.slurm               # Real BS=16 vs accum BS=16
    ├── 15_ft_vit_gradual_unfreeze.slurm  # ViT-B/16 + gradual unfreeze
    ├── 16_ft_trocr_gradual_unfreeze.slurm # TrOCR + gradual unfreeze
    ├── 17_vit_rgts_v1.slurm           # v1 architecture (comparison)
    ├── 18_lr_5e4.slurm                # Lower base LR
    ├── 19_swa_lr_high.slurm           # SWA LR=1e-3 (doubled)
    ├── 20_swa_lr_low.slurm            # SWA LR=1e-4 (halved)
    ├── 21_seed_123.slurm              # Seed=123 (reproducibility)
    ├── 22_seed_456.slurm              # Seed=456 (reproducibility)
    └── archived/                       # Superseded ablation scripts
│
└── read2016/                  # READ2016 dataset variation
    ├── 01_cnn_rnn.slurm               # CNN-RNN baseline on READ2016
    ├── 02_vit_rgts_v2_0reg.slurm      # ViT-RGTS v2, 0 registers
    ├── 03_vit_rgts_v2_4reg.slurm      # ViT-RGTS v2, 4 registers
    ├── 04_vit_rgts_v2_8reg.slurm      # ViT-RGTS v2, 8 registers
    └── 05_vit_rgts_v2_16reg.slurm     # ViT-RGTS v2, 16 registers
```

## Quick Start

```bash
cd experiments_execution/slurm_modular

# Submit primary register sweep (10 jobs)
bash submit_v2_register_sweep.sh

# Submit single experiment
bash submit_single.sh 07_vit_rgts_v2_4reg

# Submit all IAM experiments (20 jobs)
bash submit_all_iam.sh

# Submit everything (63 jobs — use with care)
bash submit_all.sh
```

## Experiment Matrix (63 total, all 150 epochs unless noted)

| Category | Models | Variations | SWA | Count |
|----------|--------|------------|-----|-------|
| **IAM** | CNN-RNN | 1 | ±SWA | 2 |
| **IAM** | ViT-RGTS v2 | 5 (reg sweep) | ±SWA | 10 |
| **IAM** | ViT-B/16 | legacy + finetune | ±SWA | 4 |
| **IAM** | TrOCR | frozen + finetune | ±SWA | 4 |
| **Synthetic** | All architectures | 8 | ±SWA | 16 |
| **Ablations** | ViT-RGTS v2 | 14 architectural | partial | 14 |
| **Ablations** | SWA variations | 4 (timing + LR) | Yes | 4 |
| **Ablations** | Seeds | 2 (123, 456) | — | 2 |
| **Ablations** | Pretrained | 2 (gradual unfreeze) | — | 2 |
| **READ2016** | CNN-RNN + ViT-RGTS v2 | reg sweep (0/4/8/16) | — | 5 |
| | | | **Total** | **54** |

## SWA Configuration

| Architecture | SWA Start | SWA LR | Rationale |
|-------------|-----------|--------|-----------|
| CNN-RNN (80ep) | 60 | 5e-4 | Last 25% of training |
| ViT-RGTS v2 (80ep) | 60 | 5e-4 | Last 25% of training |
| ViT-B/16 legacy (80ep) | 60 | 5e-5 | Lower — pretrained features |
| ViT-B/16 finetune (60ep) | 45 | 5e-6 | Very conservative |
| TrOCR frozen (80ep) | 60 | 5e-5 | Lower — head-only |
| TrOCR finetune (50ep) | 38 | 5e-6 | Very conservative |
| Synthetic mode (15ep) | 11 | * | Last ~25% of shorter training |
