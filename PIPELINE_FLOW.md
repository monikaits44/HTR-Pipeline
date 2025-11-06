# HTR Pipeline: End-to-End Flow Explanation

## 🎯 Overview
This pipeline implements a **Handwritten Text Recognition (HTR)** system that converts images of handwritten text into digital text using deep learning.

---

## 📊 Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         1. INITIALIZATION                            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │  Load config.yaml              │
                    │  - Device (CPU/GPU)            │
                    │  - Hyperparameters             │
                    │  - Architecture settings       │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      2. EXPERIMENT SETUP                             │
│  saved_models/experiments/run_X/                                     │
│  ├── config.json          (saved configuration)                     │
│  ├── training.log         (training logs)                           │
│  ├── results.csv          (epoch-level metrics)                     │
│  └── evaluation_details.csv (sample-level predictions)              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    3. DATA LOADING & PREPROCESSING                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            ▼                       ▼                       ▼
    ┌─────────────┐         ┌─────────────┐       ┌─────────────┐
    │ TRAIN SET   │         │ VAL SET     │       │ TEST SET    │
    │ (augmented) │         │ (no augment)│       │ (no augment)│
    └─────────────┘         └─────────────┘       └─────────────┘
            │                       │                       │
            └───────────────────────┴───────────────────────┘
                                    │
                            ┌───────▼────────┐
                            │ HTRDataset     │
                            │ (utils/)       │
                            └────────────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    │  For each image:               │
                    │  1. Load PNG image             │
                    │  2. Convert to grayscale       │
                    │  3. Normalize (invert: 1-x/255)│
                    │  4. Resize intelligently       │
                    │  5. Apply padding              │
                    │  6. Random augmentation (train)│
                    │  7. Convert to tensor          │
                    │  Output: (128 x 1024) tensor   │
                    └───────────────┬────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        4. MODEL ARCHITECTURE                         │
│                            HTRNet                                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │      Optional STN (Spatial    │
                    │      Transformer Network)     │
                    │      [Currently Disabled]     │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   FEATURE EXTRACTION: CNN     │
                    │   (models.py - CNN class)     │
                    └───────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        │   Initial Conv: 7x7       │   Configured layers:      │
        │   1 → 32 channels         │   [[2,64], 'M',          │
        │   Stride: [4,2]           │    [3,128], 'M',         │
        │   Padding: 3               │    [2,256]]              │
        └───────────────────────────┼───────────────────────────┘
                                    │
                            ┌───────▼─────────┐
                            │ BasicBlock      │
                            │ (ResNet-style)  │
                            │ - Conv3x3       │
                            │ - BatchNorm     │
                            │ - ReLU          │
                            │ - Skip Connect  │
                            └─────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────┐
                    │   MaxPool Flattening        │
                    │   Height → 1                │
                    │   Width preserved           │
                    │   Output: [B, 256, 1, W]    │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    5. SEQUENCE MODELING HEAD                         │
│                 (head_type: 'both' in config)                        │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │      CTCtopB (Both)       │
                    │    RNN path + CNN path    │
                    └─────────────┬─────────────┘
                                  │
            ┌─────────────────────┴─────────────────────┐
            │                                           │
            ▼                                           ▼
    ┌──────────────────┐                    ┌─────────────────┐
    │   RNN PATH       │                    │   CNN SHORTCUT  │
    │   (Main output)  │                    │   (Auxiliary)   │
    └──────────────────┘                    └─────────────────┘
            │                                           │
    ┌───────▼───────┐                        ┌─────────▼────────┐
    │ Bidirectional │                        │ Conv2d 1x3       │
    │ LSTM          │                        │ 256 → nclasses   │
    │ 3 layers      │                        │ (Direct mapping) │
    │ Hidden: 256   │                        └─────────┬────────┘
    │ Dropout: 0.2  │                                  │
    └───────┬───────┘                                  │
            │                                          │
    ┌───────▼───────┐                                  │
    │ Fully Connect │                                  │
    │ 512 → nclasses│                                  │
    │ Dropout: 0.5  │                                  │
    └───────┬───────┘                                  │
            │                                          │
            └──────────────────┬───────────────────────┘
                               │
                    Output: [T, B, C] where:
                    T = time steps (sequence length)
                    B = batch size
                    C = number of classes (characters + blank)
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          6. TRAINING LOOP                            │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │   Epoch 0: Baseline Test  │
                    │   Evaluate before training│
                    └─────────────┬─────────────┘
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │                                                    │
        │    FOR EACH EPOCH (1 to num_epochs):              │
        │                                                    │
        │    ┌────────────────────────────────────┐         │
        │    │   TRAINING PHASE                   │         │
        │    │   (trainer.train method)           │         │
        │    └────────────────────────────────────┘         │
        │              │                                     │
        │    ┌─────────▼─────────────────┐                  │
        │    │ For each batch:           │                  │
        │    │ 1. Forward pass           │                  │
        │    │ 2. CTC Loss calculation   │                  │
        │    │    - Main RNN output      │                  │
        │    │    - 0.1 × Aux CNN output │                  │
        │    │ 3. Backward pass          │                  │
        │    │ 4. Optimizer step (AdamW) │                  │
        │    │ 5. Log loss to tqdm       │                  │
        │    └───────────────────────────┘                  │
        │              │                                     │
        │    ┌─────────▼─────────────────┐                  │
        │    │ Learning rate adjustment  │                  │
        │    │ (MultiStepLR scheduler)   │                  │
        │    │ - Drop at 50% epochs      │                  │
        │    │ - Drop at 75% epochs      │                  │
        │    └───────────────────────────┘                  │
        │              │                                     │
        │    ┌─────────▼─────────────────┐                  │
        │    │ If epoch % save_every == 0│                  │
        │    │ (Currently: every epoch)  │                  │
        │    └───────────┬───────────────┘                  │
        │                │                                   │
        │    ┌───────────▼───────────────┐                  │
        │    │ EVALUATION PHASE          │                  │
        │    │ (trainer.test method)     │                  │
        │    └───────────────────────────┘                  │
        │                │                                   │
        │    ┌───────────┴───────────────┐                  │
        │    │                           │                  │
        │    ▼                           ▼                  │
        │ ┌─────────────┐         ┌─────────────┐          │
        │ │ Val Set     │         │ Test Set    │          │
        │ │ Evaluation  │         │ Evaluation  │          │
        │ └─────────────┘         └─────────────┘          │
        │        │                        │                 │
        │        └────────────┬───────────┘                 │
        │                     │                             │
        │    ┌────────────────▼────────────────┐           │
        │    │ For each sample:                │           │
        │    │ 1. Forward pass (no grad)       │           │
        │    │ 2. Argmax → character indices   │           │
        │    │ 3. CTC decode (remove blanks)   │           │
        │    │ 4. Calculate CER & WER          │           │
        │    │ 5. Log to evaluation_details.csv│           │
        │    └─────────────────────────────────┘           │
        │                     │                             │
        │    ┌────────────────▼────────────────┐           │
        │    │ Log metrics to results.csv:     │           │
        │    │ - Epoch, LR, Train Loss         │           │
        │    │ - Val CER/WER, Test CER/WER     │           │
        │    │ - Dataset sizes, model params   │           │
        │    │ - Time, device, seed            │           │
        │    └─────────────────────────────────┘           │
        │                     │                             │
        │    ┌────────────────▼────────────────┐           │
        │    │ Save model.pt if best val_cer   │           │
        │    └─────────────────────────────────┘           │
        │                                                    │
        └────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       7. CTC LOSS & DECODING                         │
└─────────────────────────────────────────────────────────────────────┘

    LOSS (Training):
    ┌─────────────────────────────────────────────────┐
    │ CTC Loss (Connectionist Temporal Classification)│
    │ - Allows variable-length input/output           │
    │ - No explicit segmentation needed               │
    │ - Handles blank character automatically         │
    │                                                  │
    │ Total Loss = Main_RNN_Loss + 0.1 × CNN_Aux_Loss │
    └─────────────────────────────────────────────────┘

    DECODING (Inference):
    ┌─────────────────────────────────────────────────┐
    │ 1. Get argmax of output probabilities           │
    │    [T, B, C] → [T, B]                           │
    │                                                  │
    │ 2. Remove consecutive duplicates                │
    │    [5, 5, 12, 12, 0, 8] → [5, 12, 0, 8]        │
    │                                                  │
    │ 3. Remove CTC blank (index 0)                   │
    │    [5, 12, 0, 8] → [5, 12, 8]                  │
    │                                                  │
    │ 4. Map indices to characters                    │
    │    [5, 12, 8] → "The"                          │
    └─────────────────────────────────────────────────┘

                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        8. EVALUATION METRICS                         │
└─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────┐
    │ CER (Character Error Rate)                       │
    │ = Levenshtein Distance / Total Characters        │
    │ - Measures character-level accuracy              │
    │ - Lower is better (0.0 = perfect)                │
    │                                                   │
    │ WER (Word Error Rate)                            │
    │ = Word Edit Distance / Total Words               │
    │ - Measures word-level accuracy                   │
    │ - Lower is better (0.0 = perfect)                │
    │ - Mode: 'tokenizer' (uses nltk word tokenizer)  │
    └─────────────────────────────────────────────────┘

                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          9. OUTPUT FILES                             │
└─────────────────────────────────────────────────────────────────────┘

    saved_models/experiments/run_X/
    │
    ├── config.json                  # Configuration snapshot
    ├── training.log                 # Detailed training logs
    ├── results.csv                  # Epoch-level metrics
    │   └── Columns: epoch, lr, loss, val_cer, val_wer, 
    │                test_cer, test_wer, dataset_sizes, etc.
    │
    ├── evaluation_details.csv       # Sample-level predictions
    │   └── Columns: epoch, dataset, sample_idx, ground_truth,
    │                prediction, sample_cer, sample_wer, lengths
    │
    ├── metrics_description.txt      # Metrics documentation
    └── model.pt                     # Best trained model weights

---

## 🧠 Model Architecture Details

### Current Configuration (from config.yaml)

```yaml
arch: 
  cnn_cfg: [[2, 64], 'M', [3, 128], 'M', [2, 256]]
  head_type: 'both'      # RNN + CNN hybrid
  rnn_type: 'lstm'       # Long Short-Term Memory
  rnn_layers: 3          # 3 stacked LSTM layers
  rnn_hidden_size: 256   # 256 hidden units per direction
  flattening: 'maxpool'  # Height reduction method
  stn: False             # Spatial transformer disabled
```

### Architecture Breakdown

```
INPUT IMAGE
   ↓
[1, 1, 128, 1024]  ← Batch=1, Channels=1, Height=128, Width=1024
   ↓
┌──────────────────┐
│ Initial Conv7x7   │
│ 1 → 32 channels   │
│ Stride [4, 2]     │
└────────┬──────────┘
         ↓
[1, 32, 32, 512]   ← Height reduced by 4, Width reduced by 2
   ↓
┌──────────────────┐
│ 2× BasicBlock    │
│ 32 → 64 channels │
└────────┬──────────┘
         ↓
┌──────────────────┐
│ MaxPool 2×2      │
└────────┬──────────┘
         ↓
[1, 64, 16, 256]   ← Height & Width halved
   ↓
┌──────────────────┐
│ 3× BasicBlock    │
│ 64 → 128 channels│
└────────┬──────────┘
         ↓
┌──────────────────┐
│ MaxPool 2×2      │
└────────┬──────────┘
         ↓
[1, 128, 8, 128]   ← Height & Width halved again
   ↓
┌──────────────────┐
│ 2× BasicBlock    │
│ 128 → 256 channels│
└────────┬──────────┘
         ↓
[1, 256, 8, 128]
   ↓
┌──────────────────┐
│ MaxPool (H→1)    │
│ Flatten height   │
└────────┬──────────┘
         ↓
[1, 256, 1, 128]   ← Sequence of 128 time steps, 256 features each
   ↓
┌──────────────────────────────────────┐
│         DUAL HEAD OUTPUT              │
│                                       │
│  RNN PATH (Main)      CNN PATH (Aux)  │
│  ↓                    ↓               │
│  Bi-LSTM 3 layers     Conv 1×3        │
│  256 hidden           256→classes     │
│  ↓                    ↓               │
│  Linear 512→classes   Permute         │
│  ↓                    ↓               │
│  [128, 1, C]          [128, 1, C]     │
└──────────────────────────────────────┘
         ↓
Output: Character probabilities per time step
C = number of character classes + 1 (blank)
```

### Why This Architecture?

1. **CNN Feature Extraction**: Learns visual features from handwriting
2. **ResNet BasicBlocks**: Skip connections help gradient flow
3. **MaxPool Flattening**: Converts 2D feature maps to 1D sequences
4. **Bi-LSTM**: Captures left-to-right AND right-to-left context
5. **Dual Heads**: 
   - RNN path: Better for sequence modeling
   - CNN path: Provides direct visual features (auxiliary)
6. **CTC**: No need for character-level annotations, just line-level transcriptions

---

## 🔄 Training Process Summary

1. **Initialization**: Load config, create experiment directory
2. **Data Loading**: IAM dataset with train/val/test splits
3. **Model Creation**: HTRNet with CNN + Bi-LSTM architecture
4. **Baseline Evaluation**: Test before any training (epoch 0)
5. **Training Loop**:
   - Forward pass through model
   - Calculate CTC loss (main + 0.1×auxiliary)
   - Backpropagation and weight updates
   - Learning rate decay at 50% and 75% of training
6. **Periodic Evaluation**: 
   - Validate on validation set
   - Test on test set
   - Log all predictions to CSV
   - Save model if best validation CER
7. **Final Outputs**: Trained model + comprehensive logs

---

## 📈 Key Parameters (from config.yaml)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `image_height` | 128 | Input image height |
| `image_width` | 1024 | Input image width |
| `batch_size` | 16 (train), 32 (eval) | Samples per batch |
| `lr` | 1e-3 | Learning rate |
| `num_epochs` | 15 | Total training epochs |
| `save_every_k_epochs` | 1 | Evaluation frequency |
| `num_workers` | 8 | Parallel data loading |
| `device` | 'cpu' | Compute device |

---

## 🎓 What Makes This HTR System Work?

1. **CTC Loss**: Allows learning without character-level segmentation
2. **Bidirectional LSTM**: Understands context from both directions
3. **ResNet-style CNN**: Deep feature extraction with gradient flow
4. **Dual Heads**: Combines sequential (RNN) and spatial (CNN) information
5. **Data Augmentation**: Random scaling/distortion for train robustness
6. **Comprehensive Logging**: Track every prediction for analysis

This is a state-of-the-art architecture for offline handwritten text recognition! 🚀
