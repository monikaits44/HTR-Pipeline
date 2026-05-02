# HTR Pipeline: Complete End-to-End Execution Flow

**Last Updated**: February 7, 2026  
**Purpose**: Comprehensive documentation of all execution paths, file dependencies, and data flow

---

## Table of Contents

1. [Project Structure Overview](#1-project-structure-overview)
2. [Execution Phases](#2-execution-phases)
3. [Detailed File-to-File Flow](#3-detailed-file-to-file-flow)
4. [Data Flow Diagrams](#4-data-flow-diagrams)
5. [Module Dependencies](#5-module-dependencies)
6. [Execution Sequences](#6-execution-sequences)

---

## 1. Project Structure Overview

```
HTR-Pipeline/
│
├── configs/                          # Configuration files (YAML)
│   ├── baseline.yaml                 # CNN-RNN baseline config
│   ├── baseline_vit_rgts.yaml        # ViT-RGTS baseline config
│   ├── torchvision_vit.yaml          # TorchVision ViT config
│   ├── trocr.yaml                    # TrOCR config
│   └── config.yaml                   # Generic config template
│
├── models.py                         # Core model architectures (1298 lines)
│   ├── CNN, BasicBlock, ResNet       # CNN components
│   ├── ViTRGTSBackbone              # Vision Transformer with registers
│   ├── TorchVisionViTBackbone       # Pre-trained ViT wrapper
│   ├── TrOCREncoderBackbone         # TrOCR encoder wrapper
│   └── HTRNet                        # Main HTR wrapper model
│
├── utils/                            # Utility modules
│   ├── __init__.py                   # Package initialization
│   ├── htr_dataset.py               # Dataset loading & preprocessing
│   ├── transforms.py                # Data augmentation pipelines
│   ├── preprocessing.py             # Image preprocessing utilities
│   ├── metrics.py                   # CER/WER calculation
│   ├── attention_extractor.py       # Attention weight extraction
│   └── visualizer.py                # Visualization utilities
│
├── scripts/                          # Executable scripts
│   ├── trainer.py                   # Main training script (602 lines)
│   ├── verify_architectures.py      # Architecture verification
│   ├── pipeline_summary.py          # Pipeline summary tool
│   ├── run_register_experiments.py  # Batch experiment runner
│   │
│   ├── preprocessing/               # Data preparation scripts
│   │   ├── prepare_iam.py          # IAM dataset preparation
│   │   ├── exploratory_data_analysis.py  # EDA visualization
│   │   └── validate_setup.py       # Setup validation
│   │
│   └── postprocessing/              # Analysis & visualization scripts
│       ├── evaluate.py              # Model evaluation
│       ├── demo.py                  # Interactive demo
│       ├── plot_training_metrics.py # Training visualization
│       ├── analyze_register_impact.py  # Register analysis
│       ├── visualize_character_vit_updated.py  # Character attention
│       ├── visualize_register_attention.py     # Register attention
│       └── [14 more visualization scripts]
│
├── experiments_execution/            # Experiment management
│   ├── slurm_scripts/               # SLURM batch scripts
│   │   ├── 01_baseline.slurm
│   │   ├── 02_vit_rgts_0reg.slurm
│   │   ├── 03_vit_rgts_2reg.slurm
│   │   └── [5 more experiment configs]
│   ├── submit_all_experiments.sh    # Batch submission script
│   ├── submit_single.sh             # Single experiment submission
│   └── experiment_tracker.ipynb     # Jupyter tracker
│
├── data/                             # Dataset storage
│   └── IAM/
│       └── processed_lines/
│           ├── train/               # Training split
│           │   ├── gt.txt          # Ground truth labels
│           │   └── *.png           # Image files
│           ├── val/                # Validation split
│           ├── test/               # Test split
│           └── classes.npy         # Character vocabulary
│
├── saved_models/                     # Trained model storage
│   └── experiments/
│       └── run_X/                   # Individual experiment
│           ├── config.json          # Saved configuration
│           ├── training.log         # Training logs
│           ├── results.csv          # Epoch-wise metrics
│           ├── htrnet.pt           # Best model checkpoint
│           └── classes.npy         # Character classes
│
├── output/                           # Generated outputs
│   ├── data_analysis/               # EDA visualizations
│   ├── training_plots/              # Training curves
│   └── register_analysis/           # Register impact analysis
│
├── visualizations/                   # Attention visualizations
│   ├── character_attention/         # Character-level attention
│   ├── character_gradcam/          # GradCAM visualizations
│   └── register_analysis/          # Register token analysis
│
└── documents/                        # Documentation
    ├── TRAINING_PIPELINE_DOCUMENTATION.md
    ├── EXPLAINABILITY_README.md
    └── [8 more documentation files]
```

---

## 2. Execution Phases

### Phase 0: Environment Setup
```
User Action → Installation & Configuration
└─> Setup Python environment
    └─> Install dependencies (requirements.txt)
        └─> Verify setup (validate_setup.py)
```

### Phase 1: Data Preparation
```
Raw IAM Dataset → Preprocessing → Ready for Training
└─> scripts/preprocessing/prepare_iam.py
    └─> scripts/preprocessing/exploratory_data_analysis.py (optional)
```

### Phase 2: Training
```
Config + Data → Training Loop → Saved Model
└─> SLURM script or direct invocation
    └─> scripts/trainer.py
        └─> models.py + utils/
```

### Phase 3: Evaluation
```
Saved Model + Test Data → Metrics
└─> scripts/postprocessing/evaluate.py
```

### Phase 4: Visualization & Analysis
```
Saved Model + Images → Attention Maps + Plots
└─> scripts/postprocessing/[visualization scripts]
```

---

## 3. Detailed File-to-File Flow

### 3.1 Training Execution Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ ENTRY POINT: SLURM Job or Direct Invocation                        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ experiments_execution/slurm_scripts/XX_experiment.slurm             │
│                                                                     │
│ #!/bin/bash                                                         │
│ #SBATCH --job-name=vit_rgts_4reg                                  │
│ #SBATCH --output=logs/vit_rgts_registers/%x_%j.out                │
│ #SBATCH --gpus=1                                                   │
│                                                                     │
│ python scripts/trainer.py \                                        │
│     --config configs/baseline_vit_rgts.yaml \                      │
│     --override arch.num_registers=4                                │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ scripts/trainer.py (Main Training Script)                          │
│                                                                     │
│ EXECUTION FLOW:                                                     │
│ 1. parse_arguments()          ← Command line args                  │
│ 2. load_config()              ← configs/*.yaml                     │
│ 3. setup_experiment_dir()     → saved_models/experiments/run_X/    │
│ 4. HTRTrainer.__init__()      ← Initialize trainer                 │
│ 5. trainer.train()            → Start training loop                │
│ 6. trainer.save_checkpoint()  → Save best model                    │
└─────────────────────────────────────────────────────────────────────┘
          ↓                    ↓                    ↓
          │                    │                    │
    [Load Data]           [Build Model]        [Training Loop]
          │                    │                    │
          ↓                    ↓                    ↓
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ utils/           │  │ models.py        │  │ Training Loop    │
│ htr_dataset.py   │  │                  │  │ Components       │
│                  │  │ Classes:         │  │                  │
│ HTRDataset       │  │ - CNN            │  │ - Forward pass   │
│ ├─ __init__()    │  │ - ResNet         │  │ - CTC loss       │
│ │  ├─ Load gt.txt│  │ - ViTRGTSBackbone│  │ - Backprop       │
│ │  └─ Extract    │  │ - TorchVisionViT │  │ - Optimizer step │
│ │     classes    │  │ - TrOCREncoder   │  │ - Validation     │
│ ├─ __getitem__() │  │ - HTRNet (main)  │  │ - Save checkpoint│
│ │  ├─ Load image │  │                  │  │                  │
│ │  ├─ Resize     │  │ HTRNet           │  │ Calls:           │
│ │  ├─ Augment    │  │ ├─ __init__()    │  │ - model()        │
│ │  └─ Return     │  │ ├─ forward()     │  │ - criterion()    │
│ └─ __len__()     │  │ └─ attn_weights  │  │ - optimizer()    │
│                  │  │    property       │  │ - evaluate()     │
│ Uses:            │  │                  │  │                  │
│ - preprocessing.py│  │ Uses:            │  │ Uses:            │
│ - transforms.py  │  │ - torch.nn       │  │ - metrics.py     │
│                  │  │ - einops         │  │   (CER, WER)     │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

### 3.2 Detailed Training Loop Sequence

```
EPOCH LOOP (for epoch in range(num_epochs)):
│
├─> 1. TRAINING PHASE
│   │
│   ├─> FOR EACH BATCH in train_dataloader:
│   │   │
│   │   ├─> a) Data Loading
│   │   │   └─> utils/htr_dataset.py::HTRDataset.__getitem__()
│   │   │       ├─> utils/preprocessing.py::load_image()
│   │   │       ├─> utils/preprocessing.py::preprocess()
│   │   │       └─> utils/transforms.py::aug_transforms_vit_strong()
│   │   │           ├─> Affine (rotation, shear, scale)
│   │   │           ├─> Perspective transform
│   │   │           ├─> ElasticTransform / GridDistortion
│   │   │           ├─> Morphological (dilation/erosion)
│   │   │           ├─> RandomBrightnessContrast
│   │   │           ├─> RandomGamma
│   │   │           ├─> GaussNoise
│   │   │           ├─> CoarseDropout
│   │   │           ├─> GaussianBlur / MotionBlur
│   │   │           └─> Sharpen
│   │   │
│   │   ├─> b) Forward Pass
│   │   │   └─> models.py::HTRNet.forward()
│   │   │       ├─> Extract features via backbone:
│   │   │       │   ├─> CNN (baseline)
│   │   │       │   ├─> ViTRGTSBackbone (with registers)
│   │   │       │   ├─> TorchVisionViTBackbone
│   │   │       │   └─> TrOCREncoderBackbone
│   │   │       │
│   │   │       ├─> Store attention weights (if ViT)
│   │   │       │   └─> self.attn_weights.append(attn)
│   │   │       │
│   │   │       ├─> RNN processing (if applicable)
│   │   │       │   └─> LSTM layers
│   │   │       │
│   │   │       └─> Classification head
│   │   │           └─> Linear(hidden_dim, num_classes)
│   │   │
│   │   ├─> c) Loss Calculation
│   │   │   └─> torch.nn.CTCLoss()(logits, labels, input_lengths, label_lengths)
│   │   │
│   │   ├─> d) Backward Pass
│   │   │   ├─> loss.backward()
│   │   │   └─> Compute gradients
│   │   │
│   │   ├─> e) Optimization Step
│   │   │   ├─> optimizer.step()
│   │   │   └─> optimizer.zero_grad()
│   │   │
│   │   └─> f) Logging
│   │       ├─> Update progress bar
│   │       └─> Log to training.log
│   │
│   └─> Average training loss for epoch
│
├─> 2. VALIDATION PHASE
│   │
│   ├─> FOR EACH BATCH in val_dataloader:
│   │   │
│   │   ├─> a) Forward pass (no gradients)
│   │   │   └─> with torch.no_grad():
│   │   │       └─> model(images)
│   │   │
│   │   ├─> b) Decode predictions
│   │   │   └─> Greedy CTC decoding
│   │   │
│   │   └─> c) Calculate metrics
│   │       └─> utils/metrics.py
│   │           ├─> CER(predictions, ground_truth)
│   │           └─> WER(predictions, ground_truth)
│   │
│   └─> Average validation CER/WER
│
├─> 3. TEST PHASE
│   │
│   └─> Same as validation, on test split
│
├─> 4. CHECKPOINT SAVING
│   │
│   ├─> IF val_cer < best_val_cer:
│   │   └─> torch.save({
│   │           'model_state_dict': model.state_dict(),
│   │           'epoch': epoch,
│   │           'val_cer': val_cer,
│   │       }, 'saved_models/experiments/run_X/htrnet.pt')
│   │
│   └─> Save character classes
│       └─> np.save('classes.npy', classes)
│
├─> 5. LOGGING
│   │
│   ├─> Write to training.log
│   ├─> Write to results.csv
│   │   └─> epoch, lr, train_loss, val_cer, val_wer, test_cer, test_wer, ...
│   │
│   └─> Print to console
│
└─> 6. LEARNING RATE SCHEDULING
    │
    └─> IF epoch == 14:
        └─> lr = lr * 0.1  (decay to 0.0001)
        IF epoch == 21:
        └─> lr = lr * 0.1  (decay to 0.00001)
```

### 3.3 Model Architecture Instantiation

```
┌─────────────────────────────────────────────────────────────────────┐
│ models.py::HTRNet.__init__()                                        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────────────┐
                    │ Check arch_type │
                    └─────────────────┘
                              ↓
        ┌──────────────┬──────┴──────┬──────────────┬──────────────┐
        │              │             │              │              │
        v              v             v              v              v
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ CNN-RNN     │ │ ViT-RGTS    │ │ TorchVision │ │ TrOCR       │ │ Other       │
│ (baseline)  │ │ (custom)    │ │ ViT         │ │ (pretrained)│ │             │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
        │              │             │              │              │
        │              │             │              │              │
        v              v             v              v              v
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ CNN()       │ │ViTRGTS      │ │TorchVision  │ │TrOCR        │ │CNN()        │
│ ↓           │ │Backbone()   │ │ViTBackbone()│ │Encoder      │ │ ↓           │
│ RNN         │ │ ↓           │ │ ↓           │ │Backbone()   │ │ FC          │
│ ↓           │ │ No RNN      │ │ No RNN      │ │ ↓           │ │             │
│ FC Head     │ │ ↓           │ │ ↓           │ │ No RNN      │ │             │
│             │ │ FC Head     │ │ FC Head     │ │ ↓           │ │             │
│             │ │             │ │             │ │ FC Head     │ │             │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘

BACKBONE DETAILS:

ViTRGTSBackbone (models.py):
├─> PatchEmbedding
│   └─> Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
│
├─> Register Tokens (if num_registers > 0)
│   └─> nn.Parameter(torch.zeros(1, num_registers, embed_dim))
│
├─> Positional Encoding
│   └─> Fixed sinusoidal or learned
│
├─> Transformer Encoder
│   └─> For each layer (depth=6):
│       ├─> Multi-Head Self-Attention
│       │   ├─> Q = K = V = input
│       │   ├─> Attention = softmax(QK^T / sqrt(d))
│       │   └─> Output = Attention @ V
│       │   └─> STORE attn_weights via hook
│       │
│       ├─> LayerNorm + Residual
│       ├─> MLP (Feed-Forward)
│       │   ├─> Linear(embed_dim, mlp_dim)
│       │   ├─> GELU()
│       │   └─> Linear(mlp_dim, embed_dim)
│       └─> LayerNorm + Residual
│
└─> Remove register tokens (if present)
    └─> output[:, num_registers:, :]
```

### 3.4 Data Augmentation Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ Image Loading: utils/preprocessing.py::load_image()                │
│ Input: PIL Image or numpy array                                    │
│ Output: Grayscale numpy array [H, W]                               │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Initial Preprocessing: utils/preprocessing.py::preprocess()        │
│ - Binarization (Otsu's threshold)                                  │
│ - Normalization to [0, 1]                                          │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Random Resize (Training Only): htr_dataset.py::__getitem__()      │
│ - Width: 0.75x to 1.25x                                            │
│ - Height: 0.9x to 1.1x (aspect ratio adjusted)                    │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Augmentation: utils/transforms.py::aug_transforms_vit_strong       │
│                                                                     │
│ Applied sequentially with probabilities:                           │
│                                                                     │
│ 1. Affine Transform (p=0.8)                                        │
│    ├─> Rotation: -15° to +15°                                     │
│    ├─> Shear: x: -40° to +40°, y: -10° to +10°                   │
│    ├─> Scale: 0.8x to 1.2x                                        │
│    └─> Translate: ±5% of image size                               │
│                                                                     │
│ 2. Perspective Transform (p=0.5)                                   │
│    └─> Scale: 0.02 to 0.1                                         │
│                                                                     │
│ 3. Distortion (p=0.9, OneOf)                                       │
│    ├─> ElasticTransform (alpha=100, sigma=12)                     │
│    └─> GridDistortion (distort_limit=±0.2)                        │
│                                                                     │
│ 4. Morphological (p=0.7, OneOf)                                    │
│    ├─> Dilation (scale 2-5 pixels)                                │
│    └─> Erosion (scale 2-5 pixels)                                 │
│                                                                     │
│ 5. Brightness/Contrast (p=0.8)                                     │
│    └─> ±40% adjustment                                            │
│                                                                     │
│ 6. Gamma Correction (p=0.7)                                        │
│    └─> Gamma: 0.6 to 1.4                                          │
│                                                                     │
│ 7. Gaussian Noise (p=0.6)                                          │
│    └─> Std: 0.02 to 0.1                                           │
│                                                                     │
│ 8. Random Erasing (p=0.5)                                          │
│    └─> 1-5 holes, 4-10 × 8-20 pixels                             │
│                                                                     │
│ 9. Blur (p=0.4, OneOf)                                             │
│    ├─> GaussianBlur (kernel 1-5)                                  │
│    └─> MotionBlur (kernel 3-7)                                    │
│                                                                     │
│ 10. Sharpen (p=0.4)                                                │
│     └─> Alpha: 0.1 to 0.4                                         │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Fixed Size Resize: htr_dataset.py::__getitem__()                  │
│ - Height: 128 pixels (fixed)                                       │
│ - Width: Maintains aspect ratio (variable)                        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Normalization: preprocess()                                        │
│ - Pixel values: [0, 1]                                            │
│ - Add channel dimension: [H, W] → [1, H, W]                       │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                    Ready for Model Input
```

---

## 4. Data Flow Diagrams

### 4.1 Complete Pipeline Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RAW IAM DATASET                                  │
│ - ASCII text files with ground truth                               │
│ - PNG images (handwritten text lines)                              │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────────────┐
                    │ prepare_iam.py  │
                    └─────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                PROCESSED DATASET STRUCTURE                          │
│ data/IAM/processed_lines/                                          │
│ ├── train/                                                         │
│ │   ├── gt.txt  (img_id transcription pairs)                      │
│ │   └── *.png   (6,482 images)                                    │
│ ├── val/                                                           │
│ │   ├── gt.txt                                                     │
│ │   └── *.png   (976 images)                                      │
│ ├── test/                                                          │
│ │   ├── gt.txt                                                     │
│ │   └── *.png   (2,915 images)                                    │
│ └── classes.npy  (79 unique characters)                           │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────────────┐
                    │ trainer.py      │
                    │ + HTRDataset    │
                    └─────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   TRAINING DATA LOADER                              │
│ DataLoader(HTRDataset)                                             │
│ - Batch size: 8-16                                                 │
│ - Shuffle: True (train), False (val/test)                         │
│ - Collate: Custom padding to max length in batch                  │
│ - Augmentation: aug_transforms_vit_strong                          │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────────────┐
                    │ Model (HTRNet)  │
                    └─────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    MODEL OUTPUT                                     │
│ Logits: [Batch, Time, Num_Classes]                                │
│ - Batch: Number of images in batch                                │
│ - Time: Sequence length (variable per image)                      │
│ - Num_Classes: 79 characters                                      │
│                                                                     │
│ Attention Weights (stored): [Batch, Heads, Seq, Seq]              │
│ - Stored in model.attn_weights list                               │
│ - One tensor per transformer layer                                │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────┬─────────────────────┬──────────────────────────┐
│                   │                     │                          │
│   CTC Loss        │   Greedy Decode     │   Attention Storage     │
│   Calculation     │   (Inference)       │   (for visualization)   │
│                   │                     │                          │
└───────────────────┴─────────────────────┴──────────────────────────┘
        ↓                     ↓                        ↓
┌───────────────┐    ┌────────────────┐    ┌────────────────────────┐
│ Backpropagation│    │ CER/WER Metrics│    │ Saved for later       │
│ + Optimization │    │ Calculation    │    │ visualization         │
└───────────────┘    └────────────────┘    └────────────────────────┘
        ↓                     ↓
┌───────────────────────────────────┐
│   SAVED CHECKPOINT                │
│ saved_models/experiments/run_X/   │
│ ├── htrnet.pt                     │
│ ├── classes.npy                   │
│ ├── config.json                   │
│ ├── training.log                  │
│ └── results.csv                   │
└───────────────────────────────────┘
                ↓
┌───────────────────────────────────┐
│   POSTPROCESSING                  │
│ - evaluate.py                     │
│ - plot_training_metrics.py        │
│ - visualize_*.py scripts          │
└───────────────────────────────────┘
                ↓
┌───────────────────────────────────┐
│   OUTPUTS                         │
│ - Attention maps (PNG)            │
│ - Training curves (PNG)           │
│ - Performance metrics (CSV)       │
│ - Analysis reports (JSON)         │
└───────────────────────────────────┘
```

### 4.2 Attention Weight Flow (ViT Models)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FORWARD PASS (ViT)                              │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Input Image [B, 1, H, W]                                           │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Patch Embedding: Conv2d(kernel=patch_size, stride=patch_size)     │
│ Output: [B, num_patches, embed_dim]                               │
│ - num_patches = (H // patch_height) × (W // patch_width)          │
│ - For 128×512 image with 4×64 patches: 32×8 = 256 patches         │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Add Register Tokens (if num_registers > 0)                        │
│ registers = nn.Parameter([1, num_registers, embed_dim])           │
│ Output: [B, num_registers + num_patches, embed_dim]               │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Add Positional Encoding                                            │
│ pos_embed = [1, num_registers + num_patches, embed_dim]           │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                 TRANSFORMER ENCODER LAYERS (×6)                     │
│                                                                     │
│ For each layer:                                                     │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐ │
│   │ Multi-Head Self-Attention (num_heads=8)                     │ │
│   │                                                              │ │
│   │ 1. Compute Q, K, V projections                              │ │
│   │    Q = input @ W_q  [B, N, embed_dim]                       │ │
│   │    K = input @ W_k  [B, N, embed_dim]                       │ │
│   │    V = input @ W_v  [B, N, embed_dim]                       │ │
│   │                                                              │ │
│   │ 2. Reshape for multi-head                                    │ │
│   │    Q = reshape(Q, [B, num_heads, N, head_dim])              │ │
│   │    K = reshape(K, [B, num_heads, N, head_dim])              │ │
│   │    V = reshape(V, [B, num_heads, N, head_dim])              │ │
│   │                                                              │ │
│   │ 3. Compute attention scores                                  │ │
│   │    scores = Q @ K.transpose(-2, -1) / sqrt(head_dim)        │ │
│   │    attn = softmax(scores, dim=-1)                           │ │
│   │    ►►► ATTENTION WEIGHTS: [B, num_heads, N, N] ◄◄◄         │ │
│   │    ►►► STORED via hook in model.attn_weights ◄◄◄           │ │
│   │                                                              │ │
│   │ 4. Apply attention to values                                 │ │
│   │    output = attn @ V                                        │ │
│   │    output = reshape(output, [B, N, embed_dim])              │ │
│   └─────────────────────────────────────────────────────────────┘ │
│                              ↓                                      │
│   ┌─────────────────────────────────────────────────────────────┐ │
│   │ Add & Norm (Residual connection + LayerNorm)                │ │
│   └─────────────────────────────────────────────────────────────┘ │
│                              ↓                                      │
│   ┌─────────────────────────────────────────────────────────────┐ │
│   │ MLP (Feed-Forward Network)                                   │ │
│   │ - Linear(embed_dim → mlp_dim)                               │ │
│   │ - GELU activation                                            │ │
│   │ - Linear(mlp_dim → embed_dim)                               │ │
│   └─────────────────────────────────────────────────────────────┘ │
│                              ↓                                      │
│   ┌─────────────────────────────────────────────────────────────┐ │
│   │ Add & Norm (Residual connection + LayerNorm)                │ │
│   └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Remove Register Tokens                                             │
│ output = output[:, num_registers:, :]                             │
│ Output: [B, num_patches, embed_dim]                               │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Classification Head                                                 │
│ - Flatten/Pool spatial dimensions                                  │
│ - Linear(embed_dim, num_classes)                                  │
│ Output: [B, Time, num_classes]                                    │
└─────────────────────────────────────────────────────────────────────┘

ATTENTION WEIGHTS STORAGE:
- Stored in: model.attn_weights (list)
- Format: List of [B, num_heads, N, N] tensors
- Length: Number of transformer layers (6 for ViT-RGTS)
- Accessible via: model.attn_weights property
- Used for: Visualization scripts (visualize_*.py)
```

---

## 5. Module Dependencies

### 5.1 Import Dependency Graph

```
scripts/trainer.py
├─> models.py
│   ├─> torch.nn
│   ├─> torch.nn.functional
│   ├─> einops
│   └─> transformers (for TrOCR)
│
├─> utils/htr_dataset.py
│   ├─> utils/preprocessing.py
│   │   ├─> PIL (Image)
│   │   ├─> cv2 (OpenCV)
│   │   └─> skimage
│   │
│   └─> utils/transforms.py
│       └─> albumentations
│
├─> utils/metrics.py
│   ├─> numpy
│   └─> editdistance
│
└─> omegaconf (config management)

scripts/postprocessing/visualize_*.py
├─> models.py
├─> utils/attention_extractor.py
│   └─> torch
│
├─> utils/visualizer.py
│   ├─> matplotlib
│   ├─> seaborn
│   └─> PIL
│
└─> utils/preprocessing.py
```

### 5.2 Configuration Dependency Chain

```
SLURM Script
    └─> --config configs/baseline_vit_rgts.yaml
        └─> --override arch.num_registers=4
            └─> OmegaConf merges config + overrides
                └─> trainer.py receives merged config
                    └─> HTRTrainer uses config values
                        ├─> arch: model architecture params
                        ├─> train: training hyperparams
                        └─> data: dataset paths
```

---

## 6. Execution Sequences

### 6.1 Complete Training Sequence (Step-by-Step)

```
STEP 1: Job Submission
├─> User: sbatch experiments_execution/slurm_scripts/03_vit_rgts_2reg.slurm
└─> SLURM: Allocate resources (GPU, CPU, memory)

STEP 2: Environment Setup
├─> Load modules (CUDA, cuDNN)
├─> Activate virtual environment
└─> Set environment variables

STEP 3: Script Invocation
└─> Execute: python scripts/trainer.py \
              --config configs/baseline_vit_rgts.yaml \
              --override arch.num_registers=2

STEP 4: Configuration Loading (trainer.py:main())
├─> Parse command line arguments
├─> Load base config from YAML
├─> Apply overrides
└─> Merge into final config object

STEP 5: Experiment Directory Setup (trainer.py:setup_experiment_dir())
├─> Create: saved_models/experiments/run_X/
├─> Save: config.json
├─> Initialize: training.log
├─> Initialize: results.csv (with header)
└─> Return: run_dir, log_file, run_number

STEP 6: Trainer Initialization (trainer.py:HTRTrainer.__init__())
├─> Set device (CUDA/CPU)
├─> Log configuration details
└─> Store config and output paths

STEP 7: Dataset Loading (trainer.py:HTRTrainer.train())
├─> Initialize HTRDataset for train split
│   ├─> Load gt.txt
│   ├─> Extract character classes
│   ├─> Store image paths and transcriptions
│   └─> Set augmentation transforms
│
├─> Initialize HTRDataset for val split
│   └─> Use same character classes
│
├─> Initialize HTRDataset for test split
│   └─> Use same character classes
│
├─> Create DataLoaders
│   ├─> Train: batch_size=8, shuffle=True
│   ├─> Val: batch_size=8, shuffle=False
│   └─> Test: batch_size=8, shuffle=False
│
└─> Log dataset statistics

STEP 8: Model Initialization (models.py:HTRNet.__init__())
├─> Determine architecture type from config
│
├─> IF arch_type == 'vit_rgts':
│   ├─> Create ViTRGTSBackbone
│   │   ├─> Initialize patch embedding
│   │   ├─> Create register tokens (num_registers=2)
│   │   ├─> Initialize positional encoding
│   │   ├─> Create transformer encoder layers
│   │   └─> Register attention hooks
│   │
│   └─> Create classification head
│       └─> Linear(embed_dim, num_classes)
│
├─> Count model parameters
├─> Move model to device (GPU)
└─> Log model architecture

STEP 9: Optimizer & Scheduler Setup
├─> Initialize Adam optimizer (lr=0.001)
├─> Create learning rate scheduler
│   └─> Step decay at epochs 15 and 22
└─> Initialize CTC loss criterion

STEP 10: Training Loop (epochs 1-30)
    │
    ├─> FOR epoch in range(1, 31):
    │       │
    │       ├─> TRAINING PHASE
    │       │   ├─> Set model to train mode
    │       │   ├─> FOR batch in train_dataloader:
    │       │   │   ├─> Load images and labels
    │       │   │   ├─> Apply augmentations
    │       │   │   ├─> Forward pass → logits
    │       │   │   ├─> Calculate CTC loss
    │       │   │   ├─> Backward pass
    │       │   │   ├─> Optimizer step
    │       │   │   └─> Accumulate loss
    │       │   │
    │       │   └─> Average training loss
    │       │
    │       ├─> VALIDATION PHASE
    │       │   ├─> Set model to eval mode
    │       │   ├─> Disable gradients
    │       │   ├─> FOR batch in val_dataloader:
    │       │   │   ├─> Forward pass
    │       │   │   ├─> Greedy CTC decode
    │       │   │   ├─> Calculate CER
    │       │   │   ├─> Calculate WER
    │       │   │   └─> Accumulate metrics
    │       │   │
    │       │   └─> Average val CER/WER
    │       │
    │       ├─> TEST PHASE
    │       │   └─> Same as validation on test split
    │       │
    │       ├─> CHECKPOINT SAVING
    │       │   ├─> IF val_cer < best_val_cer:
    │       │   │   ├─> Save model state_dict
    │       │   │   ├─> Save classes.npy
    │       │   │   └─> Update best_val_cer
    │       │   │
    │       │   └─> Always save last checkpoint
    │       │
    │       ├─> LOGGING
    │       │   ├─> Write to training.log
    │       │   ├─> Append to results.csv
    │       │   └─> Print to console
    │       │
    │       └─> LR SCHEDULING
    │           └─> Update learning rate if epoch in [15, 22]
    │
    └─> Training complete

STEP 11: Cleanup
├─> Close log file
├─> Close CSV file
└─> Log final statistics

STEP 12: SLURM Completion
├─> Write output to logs/vit_rgts_registers/jobname_jobid.out
└─> Exit with code 0
```

### 6.2 Evaluation Sequence

```
STEP 1: Script Invocation
└─> python scripts/postprocessing/evaluate.py \
      --model-path saved_models/experiments/run_33/ \
      --data-path data/IAM/processed_lines \
      --split test

STEP 2: Load Model (evaluate.py)
├─> Load config.json from model_path
├─> Load classes.npy
├─> Initialize HTRNet with saved config
├─> Load model weights from htrnet.pt
└─> Move model to device

STEP 3: Load Dataset
├─> Initialize HTRDataset for specified split
├─> No augmentations (evaluation mode)
└─> Create DataLoader

STEP 4: Inference Loop
├─> Set model to eval mode
├─> FOR batch in dataloader:
│   ├─> Forward pass (no gradients)
│   ├─> Greedy CTC decode
│   ├─> Calculate CER for each sample
│   ├─> Calculate WER for each sample
│   └─> Accumulate predictions
│
└─> Aggregate metrics

STEP 5: Report Results
├─> Print average CER
├─> Print average WER
├─> Save predictions to JSON
└─> Optional: Save per-sample metrics
```

### 6.3 Visualization Sequence

```
STEP 1: Script Invocation
└─> python scripts/postprocessing/visualize_character_vit_updated.py \
      --model-path saved_models/experiments/run_34/ \
      --image-path notebook/sample_images/a01-000u-00.png \
      --output-dir visualizations/character_attention/

STEP 2: Model Loading
├─> Load config and classes
├─> Initialize HTRNet
├─> Load weights
└─> Enable attention capture hooks

STEP 3: Image Processing
├─> Load image
├─> Preprocess (normalize)
├─> Resize to fixed size
└─> Convert to tensor

STEP 4: Forward Pass with Attention Capture
├─> model(image)
│   ├─> Forward through ViT backbone
│   ├─> Attention hooks capture weights
│   └─> Store in model.attn_weights
│
├─> Get predictions (greedy decode)
└─> Extract attention weights

STEP 5: Attention Processing
├─> Select attention layer (e.g., last layer)
├─> Average over attention heads
├─> Remove register token attention
├─> Reshape to 2D spatial grid
└─> Interpolate to image size

STEP 6: Visualization Generation
├─> FOR each character in prediction:
│   ├─> Extract attention for character position
│   ├─> Overlay on original image
│   ├─> Add text annotations
│   └─> Save individual attention map
│
└─> Create composite visualization

STEP 7: Optional: Gradient Saliency
├─> Enable gradients
├─> Forward pass
├─> Backward from prediction
├─> Extract gradients w.r.t. input
├─> Generate saliency map
└─> Save alongside attention maps

STEP 8: Save Outputs
├─> Save attention maps as PNG
├─> Save saliency maps as PNG
├─> Save metadata as JSON
└─> Print completion message
```

### 6.4 Register Analysis Sequence

```
STEP 1: Script Invocation
└─> python scripts/postprocessing/analyze_register_impact.py \
      --model-path saved_models/experiments/run_X/ --registers 0 \
      --model-path saved_models/experiments/run_Y/ --registers 2 \
      --model-path saved_models/experiments/run_Z/ --registers 4

STEP 2: Load Results from Each Model
├─> FOR each model_path:
│   ├─> Load results.csv
│   ├─> Extract model name
│   ├─> Store dataframe with register count
│   └─> Log loading status
│
└─> Validate all models loaded successfully

STEP 3: Statistical Analysis
├─> FOR each model:
│   ├─> Calculate best test CER
│   ├─> Calculate epochs to <10% CER
│   ├─> Calculate overfitting gap (final - best)
│   ├─> Calculate training stability (loss variance)
│   └─> Calculate improvement rate
│
└─> Identify best configuration

STEP 4: Generate Comparison Plots
├─> Plot 1: Training loss convergence
├─> Plot 2: Validation CER comparison
├─> Plot 3: Test CER comparison
├─> Plot 4: Best CER vs register count (bar chart)
├─> Plot 5: Convergence speed analysis
├─> Plot 6: Overfitting analysis
├─> Plot 7: Training stability
├─> Plot 8: Improvement rate
└─> Plot 9: Summary statistics table

STEP 5: Print Analysis Report
├─> Print table of key metrics
├─> Identify best configuration
├─> Identify fastest convergence
├─> Identify most stable training
└─> Print recommendations

STEP 6: Save Outputs
├─> Save comprehensive figure as PNG
└─> Print completion message
```

---

## 7. File-Level Execution Matrix

| File | Entry Point | Called By | Calls To | Input | Output |
|------|-------------|-----------|----------|-------|--------|
| **trainer.py** | ✅ `main()` | SLURM script | models.py, utils/* | Config YAML | Trained model |
| **models.py** | ❌ Module | trainer.py | torch.nn | Config dict | Model instance |
| **htr_dataset.py** | ❌ Class | trainer.py | preprocessing.py | Image paths | Batched tensors |
| **transforms.py** | ❌ Module | htr_dataset.py | albumentations | Config | Transform pipeline |
| **metrics.py** | ❌ Functions | trainer.py | editdistance | Predictions | CER/WER scores |
| **evaluate.py** | ✅ `main()` | User/Script | models.py, htr_dataset.py | Model path | Metrics report |
| **visualize_*.py** | ✅ `main()` | User/Script | models.py, attention_extractor.py | Model + image | Attention maps |
| **plot_training_metrics.py** | ✅ `main()` | User/Script | pandas, matplotlib | results.csv | Training plots |
| **analyze_register_impact.py** | ✅ `main()` | User/Script | pandas, matplotlib | Multiple results.csv | Comparison plots |
| **prepare_iam.py** | ✅ `main()` | User (one-time) | os, shutil | Raw IAM data | Processed splits |
| **exploratory_data_analysis.py** | ✅ `main()` | User (optional) | pandas, matplotlib | Processed data | EDA visualizations |

---

## 8. Common Execution Patterns

### Pattern 1: Full Experiment Workflow
```bash
# 1. Data preparation (one-time)
python scripts/preprocessing/prepare_iam.py

# 2. Optional: EDA
python scripts/preprocessing/exploratory_data_analysis.py \
    --data-path data/IAM/processed_lines

# 3. Training
sbatch experiments_execution/slurm_scripts/03_vit_rgts_2reg.slurm

# 4. Monitor training
tail -f saved_models/experiments/run_X/training.log

# 5. Evaluation
python scripts/postprocessing/evaluate.py \
    --model-path saved_models/experiments/run_X/

# 6. Visualization
python scripts/postprocessing/visualize_character_vit_updated.py \
    --model-path saved_models/experiments/run_X/ \
    --image-path notebook/sample_images/a01-000u-00.png

# 7. Analysis
python scripts/postprocessing/plot_training_metrics.py \
    --model-path saved_models/experiments/run_X/
```

### Pattern 2: Multiple Experiment Comparison
```bash
# 1. Train multiple configurations
sbatch experiments_execution/submit_all_experiments.sh

# 2. Wait for completion
squeue -u $USER

# 3. Compare results
python scripts/postprocessing/plot_training_metrics.py \
    --model-path saved_models/experiments/run_33/ \
    --model-path saved_models/experiments/run_34/ \
    --model-path saved_models/experiments/run_35/

# 4. Register analysis
python scripts/postprocessing/analyze_register_impact.py \
    --model-path saved_models/experiments/run_X/ --registers 0 \
    --model-path saved_models/experiments/run_Y/ --registers 2 \
    --model-path saved_models/experiments/run_Z/ --registers 4
```

### Pattern 3: Quick Inference
```bash
# Load model and predict on single image
python scripts/postprocessing/demo.py \
    --model-path saved_models/experiments/run_34/ \
    --image-path test_image.png
```

---

## 9. Critical Code Paths

### Path 1: Model Instantiation
```
trainer.py:train()
  └─> models.py:HTRNet.__init__()
      ├─> Check config.arch.type
      ├─> IF 'vit_rgts':
      │   └─> models.py:ViTRGTSBackbone.__init__()
      │       ├─> Create patch embedding
      │       ├─> Initialize register tokens
      │       ├─> Create transformer blocks
      │       └─> Register attention hooks
      │
      ├─> Create classification head
      └─> Return initialized model
```

### Path 2: Data Loading & Augmentation
```
trainer.py:train()
  └─> HTRDataset.__init__()
      ├─> Load gt.txt
      └─> Extract character classes
  
  └─> DataLoader.__iter__()
      └─> HTRDataset.__getitem__()
          ├─> preprocessing.py:load_image()
          ├─> Random resize (training only)
          ├─> transforms.py:aug_transforms_vit_strong()
          ├─> Fixed size resize
          ├─> preprocessing.py:preprocess()
          └─> Return (image_tensor, transcription)
```

### Path 3: Forward Pass & Loss
```
trainer.py:train()
  └─> FOR batch in train_loader:
      ├─> images, labels = batch
      ├─> logits = model(images)
      │   └─> models.py:HTRNet.forward()
      │       ├─> backbone(images)
      │       │   └─> Store attention weights
      │       ├─> Optional: RNN processing
      │       └─> classification_head(features)
      │
      ├─> loss = criterion(logits, labels)
      │   └─> torch.nn.CTCLoss()
      │
      ├─> loss.backward()
      └─> optimizer.step()
```

### Path 4: Attention Extraction for Visualization
```
visualize_*.py:main()
  └─> Load model
      └─> model.attn_weights = []  # Clear previous
  
  └─> Forward pass
      └─> model(image)
          └─> ViT layers trigger hooks
              └─> Hooks append to model.attn_weights
  
  └─> Extract attention
      ├─> attn = model.attn_weights[-1]  # Last layer
      ├─> Average over heads
      ├─> Remove register tokens
      ├─> Reshape to 2D grid
      └─> Interpolate to image size
```

---

## 10. Configuration Override Mechanism

```
BASE CONFIG (YAML)
    ↓
OmegaConf.load('config.yaml')
    ↓
MERGE with command line overrides
    --override arch.num_registers=4
    --override train.batch_size=16
    ↓
FINAL CONFIG OBJECT
    ↓
Used throughout trainer.py
    config.arch.type
    config.arch.num_registers
    config.train.batch_size
    config.train.learning_rate
    etc.
```

---

## 11. Error Handling & Logging

### Logging Hierarchy
```
Console Output (stdout)
    ├─> Progress bars (tqdm)
    ├─> Epoch summaries
    └─> Warnings/Errors

training.log
    ├─> Configuration
    ├─> Dataset statistics
    ├─> Model architecture
    ├─> Epoch-by-epoch metrics
    └─> Checkpoint saves

results.csv
    └─> Structured metrics per epoch
        (for plotting and analysis)

SLURM output file
    └─> Full console output + errors
        logs/experiment/jobname_jobid.out
```

### Error Propagation
```
Low-level error (e.g., file not found)
    ↓
Exception raised
    ↓
Caught in try-except block
    ↓
Logged to training.log
    ↓
Logged to console
    ↓
Optionally: Exit with error code
```

---

## 12. Performance Considerations

### Memory Management
- **Attention Storage**: Only stored during visualization, not training
- **Gradient Accumulation**: Optional for large batches
- **DataLoader Workers**: Set num_workers based on CPU count
- **Pin Memory**: Enabled for faster GPU transfer

### Computation Optimization
- **Mixed Precision**: Can be enabled with torch.cuda.amp
- **Batch Size**: Tuned based on GPU memory
- **Gradient Checkpointing**: Available for very deep models

---

## Summary

This HTR pipeline follows a **modular, extensible design** where:

1. **Data flows** from raw images → preprocessing → augmentation → model → predictions
2. **Execution paths** are well-defined: training, evaluation, visualization
3. **File dependencies** are minimal and explicit
4. **Configuration** is centralized in YAML files with override capability
5. **Logging** is comprehensive at multiple levels
6. **Visualization** is decoupled from training for flexibility

The complete execution can be traced from:
- **Entry**: SLURM script
- **Middle**: trainer.py + models.py + utils
- **Exit**: saved model + logs + visualizations

Every file has a specific role, and the data flow is unidirectional and transparent.

---

**End of Documentation**
