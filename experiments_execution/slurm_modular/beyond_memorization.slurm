#!/bin/bash
#SBATCH --job-name=beyond_mem
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/logs/beyond_memorization/output_%j.log
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/experiments_execution/logs/beyond_memorization/error_%j.log
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080

# =============================================================================
# Beyond-Memorization (ICDAR 2025) — paper reproduction
#   Training-free style mixing via writer-embedding injection in a pretrained
#   WordStylist latent diffusion model. Generates style-mixed handwriting and
#   the per-character cross-attention maps (paper Fig. 5/6).
#
# Runs the charLocation sweep [-1, 0, 1, 2, 3] internally (one invocation).
# Prereqs (see run_from_config.py validation):
#   - external/Beyond-Memorization/models/ema_ckpt.pt      (download_pretrained.py)
#   - external/Beyond-Memorization/stable-diffusion-v1-5/vae/  (download_pretrained.py)
#   - data/IAM/words_64x256/*.png                          (prepare_images.py; needs IAM access)
# =============================================================================

source ~/.bashrc
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline
source /home/hpc/iwi5/iwi5369h/HTR-Pipeline/.venv/bin/activate

mkdir -p experiments_execution/logs/beyond_memorization
mkdir -p outputs/beyond_memorization

echo "========================================="
echo "Beyond-Memorization — paper reproduction"
echo "Started at: $(date)"
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "========================================="

/home/hpc/iwi5/iwi5369h/HTR-Pipeline/.venv/bin/python \
    external/Beyond-Memorization/run_from_config.py \
    --config configs/beyond_memorization.yaml

echo "========================================="
echo "Completed at: $(date)"
echo "Outputs under: outputs/beyond_memorization/"
echo "========================================="
