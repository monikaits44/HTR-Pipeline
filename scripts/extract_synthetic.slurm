#!/bin/bash
#SBATCH --job-name=extract_synth
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/extract_synth_%j.out
#SBATCH --error=logs/extract_synth_%j.err

# ============================================================================
# Extract synthetic LMDB data to IAM-compatible folder format
# ============================================================================
# Input:  LMDB at /home/woody/.../synthesized_images_1M.lmdb (999,488 entries)
# Output: /home/woody/.../synthetic_processed_lines/{train,val}/
#         with gt.txt + sharded PNG subdirectories
#
# Estimated runtime: 2-6 hours (depends on I/O speed)
# Estimated disk:    ~15-25 GB on woody
# Memory:            <100 MB (streaming extraction)
# ============================================================================

echo "============================================"
echo "SYNTHETIC DATA EXTRACTION"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "Started:   $(date)"
echo "============================================"

# Activate environment
source /home/hpc/iwi5/iwi5369h/HTR-Pipeline/.venv/bin/activate
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline

# Check woody disk space before starting
echo ""
echo "Woody disk usage before extraction:"
du -sh /home/woody/iwi5/iwi5369h/projects/synth_htr/ 2>/dev/null || echo "  (directory not yet accessed)"
echo ""

# Run extraction
python scripts/extract_synthetic_to_iam_format.py \
    --lmdb_path /home/woody/iwi5/iwi5369h/projects/synth_htr/synthesized_images_1M.lmdb \
    --output_dir /home/woody/iwi5/iwi5369h/projects/synth_htr/synthetic_processed_lines \
    --val_ratio 0.05 \
    --seed 42

EXIT_CODE=$?

echo ""
echo "============================================"
echo "Woody disk usage after extraction:"
du -sh /home/woody/iwi5/iwi5369h/projects/synth_htr/
du -sh /home/woody/iwi5/iwi5369h/projects/synth_htr/synthetic_processed_lines/ 2>/dev/null
echo ""
echo "File counts:"
echo -n "  train images: " && find /home/woody/iwi5/iwi5369h/projects/synth_htr/synthetic_processed_lines/train -name "*.png" 2>/dev/null | wc -l
echo -n "  val images:   " && find /home/woody/iwi5/iwi5369h/projects/synth_htr/synthetic_processed_lines/val -name "*.png" 2>/dev/null | wc -l
echo ""
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE
