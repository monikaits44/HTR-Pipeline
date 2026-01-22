#!/bin/bash

# Quick submission script for individual experiments
# Usage: ./submit_single.sh <experiment_number>
# Example: ./submit_single.sh 4  (submits ViT-RGTS with 4 registers)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/slurm_scripts"

if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_number>"
    echo ""
    echo "Available experiments:"
    echo "  1 - Baseline CNN-RNN"
    echo "  2 - ViT-RGTS (0 registers)"
    echo "  3 - ViT-RGTS (2 registers)"
    echo "  4 - ViT-RGTS (4 registers)"
    echo "  5 - ViT-RGTS (8 registers)"
    echo "  6 - ViT-RGTS (16 registers)"
    echo "  7 - TorchVision ViT-B/16"
    echo "  8 - TrOCR Base"
    echo ""
    exit 1
fi

case $1 in
    1)
        script="01_baseline.slurm"
        desc="Baseline CNN-RNN"
        ;;
    2)
        script="02_vit_rgts_0reg.slurm"
        desc="ViT-RGTS (0 registers)"
        ;;
    3)
        script="03_vit_rgts_2reg.slurm"
        desc="ViT-RGTS (2 registers)"
        ;;
    4)
        script="04_vit_rgts_4reg.slurm"
        desc="ViT-RGTS (4 registers)"
        ;;
    5)
        script="05_vit_rgts_8reg.slurm"
        desc="ViT-RGTS (8 registers)"
        ;;
    6)
        script="06_vit_rgts_16reg.slurm"
        desc="ViT-RGTS (16 registers)"
        ;;
    7)
        script="07_torchvision_vit_b16.slurm"
        desc="TorchVision ViT-B/16"
        ;;
    8)
        script="08_trocr_base.slurm"
        desc="TrOCR Base"
        ;;
    *)
        echo "Error: Invalid experiment number '$1'"
        echo "Please choose a number between 1 and 8"
        exit 1
        ;;
esac

echo "Submitting: $desc"
job_id=$(sbatch --parsable "$script")
echo "Job ID: $job_id"
echo ""
echo "To monitor this job:"
echo "  squeue -j $job_id"
echo ""
echo "To view logs:"
echo "  tail -f ../logs/*/${script%.slurm}*.log"
