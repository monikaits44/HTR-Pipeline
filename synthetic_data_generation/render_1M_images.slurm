#!/bin/bash
#SBATCH --job-name=synth_render_1M
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=/home/woody/iwi5/iwi5369h/projects/synth_htr/render_1M_%j.log
#SBATCH --error=/home/woody/iwi5/iwi5369h/projects/synth_htr/render_1M_%j.err

echo "=========================================="
echo "SYNTHETIC IMAGE GENERATION - 1M IMAGES"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo ""

cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline/synthetic_data_generation
source /home/hpc/iwi5/iwi5369h/HTR-Pipeline/.venv/bin/activate

python3 "6. text_rendering_filter_not_support_all_charcters_LMDB.py"

echo ""
echo "End time: $(date)"
echo "=========================================="
ls -lh /home/woody/iwi5/iwi5369h/projects/synth_htr/synthesized_images_1M.lmdb/
