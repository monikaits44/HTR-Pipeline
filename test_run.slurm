#!/bin/bash
#SBATCH --job-name=test_run             # Job name (this will show up in SLURM job queue)
#SBATCH --output=job_output.log             # Output log file (where stdout will be written)
#SBATCH --error=job_error.log              # Error log file (where stderr will be written)
#SBATCH --time=04:30:00                    # Time limit in the format hh:mm:ss (e.g., 2 hours)
#SBATCH --gres=gpu:1                        # Request 1 GPU (adjust this if you need more)
#SBATCH --cpus-per-task=8                  # Number of CPU cores per task
#SBATCH --partition=rtx3080                # Specify the partition to use (typically for GPU nodes)

# Activate the virtual environment
source ~/.bashrc
source ./.venv/bin/activate

# # Run the Python script
# # /home/hpc/iwi5/iwi5369h/HTR-Pipeline/.venv/bin/python scripts/trainer.py configs/config.yaml configs/baseline.yaml
/home/hpc/iwi5/iwi5369h/HTR-Pipeline/.venv/bin/python scripts/trainer.py configs/config.yaml configs/baseline_vit_rgts.yaml