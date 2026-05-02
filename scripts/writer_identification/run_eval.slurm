#!/bin/bash
#SBATCH --job-name=writer_id_eval
#SBATCH --output=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/output/writer_id/slurm_output_%j.log
#SBATCH --error=/home/hpc/iwi5/iwi5369h/HTR-Pipeline/output/writer_id/slurm_error_%j.log
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080

source ~/.bashrc
source /home/hpc/iwi5/iwi5369h/HTR-Pipeline/.venv/bin/activate
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline

echo "========================================="
echo "Writer Identification Evaluation"
echo "Started at: $(date)"
echo "========================================="

python scripts/writer_identification/evaluate.py \
    --run-dirs \
        saved_models/experiments/run_55 \
        saved_models/experiments/run_58 \
        saved_models/experiments/run_63 \
        saved_models/experiments/run_71 \
    --data-dir data/IAM/processed_lines \
    --split test \
    --device cuda:0 \
    --save-dir output/writer_id

echo "========================================="
echo "Completed at: $(date)"
echo "========================================="

# Generate visualizations
python scripts/writer_identification/visualize.py \
    --results-csv output/writer_id/writer_id_results.csv \
    --save-dir output/writer_id

echo "Visualization complete."
