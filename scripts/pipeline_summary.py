#!/usr/bin/env python3
"""
HTR Pipeline Flow Visualizer
Quick reference for the end-to-end pipeline
"""

PIPELINE_SUMMARY = """
════════════════════════════════════════════════════════════════════════════════
                        HTR PIPELINE: QUICK REFERENCE
════════════════════════════════════════════════════════════════════════════════

1. START: python scripts/trainer.py config.yaml
   ├─ Load configuration (config.yaml)
   ├─ Create experiment directory (saved_models/experiments/run_X/)
   └─ Initialize logging (training.log, results.csv, evaluation_details.csv)

2. DATA PIPELINE (utils/htr_dataset.py)
   ├─ Load: data/IAM/processed_lines/{train,val,test}/gt.txt
   ├─ Preprocess: 
   │  ├─ Load PNG → Grayscale → Normalize (1 - img/255)
   │  ├─ Resize with aspect ratio preservation
   │  ├─ Pad to fixed size: (128, 1024)
   │  └─ Random augmentation (training only)
   └─ Output: Batches of [B, 1, 128, 1024] tensors

3. MODEL ARCHITECTURE (models.py → HTRNet)
   
   INPUT: [B, 1, 128, 1024]
     ↓
   ┌─────────────────────────────────────────┐
   │ CNN FEATURE EXTRACTOR                   │
   │ - Initial Conv7x7: 1→32 channels        │
   │ - ResNet BasicBlocks:                   │
   │   • 2 blocks: 32→64  [MaxPool]          │
   │   • 3 blocks: 64→128 [MaxPool]          │
   │   • 2 blocks: 128→256                   │
   │ - MaxPool Height→1 (flattening)         │
   │ Output: [B, 256, 1, W] ≈ [B, 256, 1, 128]│
   └─────────────────────────────────────────┘
     ↓
   ┌─────────────────────────────────────────┐
   │ DUAL HEAD (head_type='both')            │
   │                                         │
   │ RNN PATH (Main):                        │
   │ - Bi-LSTM: 3 layers, 256 hidden        │
   │ - Fully Connected: 512→nclasses        │
   │                                         │
   │ CNN PATH (Auxiliary):                   │
   │ - Conv2d 1×3: 256→nclasses             │
   │                                         │
   │ Loss = RNN_loss + 0.1 × CNN_loss       │
   └─────────────────────────────────────────┘
     ↓
   OUTPUT: [Time_steps, Batch, Classes]
           Character probabilities per timestep

4. TRAINING LOOP (scripts/trainer.py)
   
   EPOCH 0: Baseline evaluation (test set)
   
   FOR epoch = 1 to num_epochs:
   
     ┌── TRAINING PHASE ──────────────────────┐
     │ For each batch:                        │
     │ 1. Forward pass                        │
     │ 2. CTC Loss (main + 0.1×auxiliary)     │
     │ 3. Backward pass                       │
     │ 4. AdamW optimizer step                │
     │ 5. Log to tqdm progress bar            │
     └────────────────────────────────────────┘
     
     ┌── LEARNING RATE SCHEDULE ──────────────┐
     │ MultiStepLR: Drops at 50% and 75%     │
     └────────────────────────────────────────┘
     
     IF epoch % save_every_k_epochs == 0:
     
       ┌── EVALUATION PHASE ──────────────────┐
       │ Validation Set:                      │
       │ - Forward pass (no gradients)        │
       │ - CTC decode predictions             │
       │ - Calculate CER & WER                │
       │ - Log each sample to CSV             │
       │                                      │
       │ Test Set:                            │
       │ - Same as validation                 │
       │                                      │
       │ Save model if best validation CER    │
       └──────────────────────────────────────┘
       
       ┌── LOGGING ───────────────────────────┐
       │ results.csv:                         │
       │ - Epoch-level metrics                │
       │                                      │
       │ evaluation_details.csv:              │
       │ - Every single prediction            │
       │ - Ground truth vs predicted          │
       │ - Per-sample CER/WER                 │
       └──────────────────────────────────────┘

5. CTC DECODING (Inference)
   
   Raw Output: [T, B, C] logits
     ↓
   Argmax: Get most likely character per timestep
     ↓
   Remove Duplicates: "hheelloo" → "helo"
     ↓
   Remove Blanks: CTC blank character (index 0)
     ↓
   Map to Characters: indices → text
     ↓
   Final Text: "hello"

6. METRICS
   
   CER (Character Error Rate):
   - Levenshtein distance at character level
   - CER = edit_distance / total_chars
   - Range: 0.0 (perfect) to 1.0+ (terrible)
   
   WER (Word Error Rate):
   - Levenshtein distance at word level
   - WER = word_edit_distance / total_words
   - Mode: 'tokenizer' (uses nltk)
   - Range: 0.0 (perfect) to 1.0+ (terrible)

7. OUTPUT FILES (saved_models/experiments/run_X/)
   
   ├── config.json               ← Configuration snapshot
   ├── training.log              ← Detailed logs
   ├── results.csv               ← Epoch metrics (CER, WER, loss, etc.)
   ├── evaluation_details.csv    ← All predictions with per-sample metrics
   ├── metrics_description.txt   ← Metrics documentation
   └── model.pt                  ← Best model weights

════════════════════════════════════════════════════════════════════════════════
                            KEY COMPONENTS
════════════════════════════════════════════════════════════════════════════════

FILE                         PURPOSE
────────────────────────────────────────────────────────────────────────────────
config.yaml                  Main configuration file
scripts/trainer.py           Training loop & experiment management
models.py                    HTRNet architecture (CNN + RNN)
utils/htr_dataset.py         Data loading & preprocessing
utils/preprocessing.py       Image preprocessing functions
utils/transforms.py          Data augmentation
utils/metrics.py             CER & WER calculations

════════════════════════════════════════════════════════════════════════════════
                         CURRENT CONFIGURATION
════════════════════════════════════════════════════════════════════════════════

Parameter              Value                Purpose
────────────────────────────────────────────────────────────────────────────────
Input Size             128 × 1024           Fixed input image dimensions
Batch Size             16 (train)           Samples per training batch
                       32 (eval)            Samples per evaluation batch
Learning Rate          1e-3                 Initial learning rate
Epochs                 15                   Total training epochs
Scheduler              MultiStepLR          Drops at epochs 7 and 11
Save Frequency         Every 1 epoch        Model save & evaluation frequency
Device                 CPU                  Compute device

CNN Config             [[2,64], 'M',        Feature extraction layers
                        [3,128], 'M',       M = MaxPool
                        [2,256]]            [n,c] = n blocks of c channels

RNN Config             Type: LSTM           Bidirectional LSTM
                       Layers: 3            Stacked layers
                       Hidden: 256          Units per direction

Head Type              'both'               RNN (main) + CNN (auxiliary)
Flattening             'maxpool'            Height reduction method

════════════════════════════════════════════════════════════════════════════════
                             MODEL SIZE
════════════════════════════════════════════════════════════════════════════════

Total Parameters: ~5-10M (depends on number of character classes)

Breakdown:
- CNN Feature Extractor: ~1-2M parameters
- Bi-LSTM (3 layers): ~3-5M parameters
- Output Heads (RNN + CNN): ~500K parameters

════════════════════════════════════════════════════════════════════════════════
"""

if __name__ == '__main__':
    print(PIPELINE_SUMMARY)
