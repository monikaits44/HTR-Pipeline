
# Master script to submit all experiments step by step
# Can be run either directly (./submit_all_experiments.sh) or via sbatch
# This script submits jobs sequentially to avoid overloading the cluster

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/slurm_scripts"

echo "========================================="
echo "HTR Pipeline - Experiment Execution"
echo "========================================="
echo ""
echo "This script will submit all experiment variations:"
echo "  1. Baseline CNN-RNN"
echo "  2-6. ViT-RGTS (0, 2, 4, 8, 16 registers)"
echo "  7. TorchVision ViT-B/16"
echo "  8. TrOCR Base"
echo ""
echo "Logs will be saved in: experiments_execution/logs/"
echo ""
echo "========================================="
echo ""

# Function to submit a job and optionally wait
submit_job() {
    local script=$1
    local description=$2
    local wait_flag=$3
    
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] Submitting: $description"
    
    if [ "$wait_flag" = "wait" ]; then
        # Submit with dependency on previous job
        job_id=$(sbatch --parsable "$script" 2>&1)
        if [[ $job_id =~ ^[0-9]+$ ]]; then
            echo "  Job ID: $job_id"
            echo "  Status: Queued/Running"
        else
            echo "  Error: $job_id"
            return 1
        fi
        echo ""
        echo "$job_id"
    else
        # Submit without dependency
        job_id=$(sbatch --parsable "$script" 2>&1)
        if [[ $job_id =~ ^[0-9]+$ ]]; then
            echo "  Job ID: $job_id"
            echo "  Status: Queued"
        else
            echo "  Error: $job_id"
            return 1
        fi
        echo ""
        echo "$job_id"
    fi
}

# Function to submit with dependency
submit_with_dependency() {
    local script=$1
    local description=$2
    local dependency=$3
    
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] Submitting: $description"
    
    if [ -n "$dependency" ]; then
        job_id=$(sbatch --parsable --dependency=afterany:$dependency "$script" 2>&1)
        if [[ $job_id =~ ^[0-9]+$ ]]; then
            echo "  Job ID: $job_id"
            echo "  Dependency: $dependency"
            echo "  Status: Queued (will start after previous job)"
        else
            echo "  Error: $job_id"
            return 1
        fi
    else
        job_id=$(sbatch --parsable "$script" 2>&1)
        if [[ $job_id =~ ^[0-9]+$ ]]; then
            echo "  Job ID: $job_id"
            echo "  Status: Queued"
        else
            echo "  Error: $job_id"
            return 1
        fi
    fi
    echo ""
    echo "$job_id"
}

# Determine submission mode
# If run via sbatch, default to parallel mode (no user interaction)
# If run directly, ask user
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Running via SLURM batch job (ID: $SLURM_JOB_ID)"
    echo "Defaulting to Parallel mode (all jobs submitted at once)"
    mode=2
else
    # Ask user for submission mode
    echo "Select submission mode:"
    echo "  1) Sequential (each job waits for previous to complete)"
    echo "  2) Parallel (all jobs submitted at once)"
    echo "  3) Manual (submit one at a time with confirmation)"
    echo ""
    read -p "Enter choice [1-3]: " mode
fi

case $mode in
    1)
        echo ""
        echo "Mode: Sequential Execution"
        echo "Each job will wait for the previous one to complete."
        echo ""
        
        # Submit first job
        prev_job=$(submit_with_dependency "01_baseline.slurm" "Baseline CNN-RNN" "")
        
        # Submit remaining jobs with dependencies
        prev_job=$(submit_with_dependency "02_vit_rgts_0reg.slurm" "ViT-RGTS (0 registers)" "$prev_job")
        prev_job=$(submit_with_dependency "03_vit_rgts_2reg.slurm" "ViT-RGTS (2 registers)" "$prev_job")
        prev_job=$(submit_with_dependency "04_vit_rgts_4reg.slurm" "ViT-RGTS (4 registers)" "$prev_job")
        prev_job=$(submit_with_dependency "05_vit_rgts_8reg.slurm" "ViT-RGTS (8 registers)" "$prev_job")
        prev_job=$(submit_with_dependency "06_vit_rgts_16reg.slurm" "ViT-RGTS (16 registers)" "$prev_job")
        prev_job=$(submit_with_dependency "07_torchvision_vit_b16.slurm" "TorchVision ViT-B/16" "$prev_job")
        prev_job=$(submit_with_dependency "08_trocr_base.slurm" "TrOCR Base" "$prev_job")
        ;;
        
    2)
        echo ""
        echo "Mode: Parallel Execution"
        echo "All jobs will be submitted at once."
        echo ""
        
        submit_job "01_baseline.slurm" "Baseline CNN-RNN"
        submit_job "02_vit_rgts_0reg.slurm" "ViT-RGTS (0 registers)"
        submit_job "03_vit_rgts_2reg.slurm" "ViT-RGTS (2 registers)"
        submit_job "04_vit_rgts_4reg.slurm" "ViT-RGTS (4 registers)"
        submit_job "05_vit_rgts_8reg.slurm" "ViT-RGTS (8 registers)"
        submit_job "06_vit_rgts_16reg.slurm" "ViT-RGTS (16 registers)"
        submit_job "07_torchvision_vit_b16.slurm" "TorchVision ViT-B/16"
        submit_job "08_trocr_base.slurm" "TrOCR Base"
        ;;
        
    3)
        echo ""
        echo "Mode: Manual Execution"
        echo "You will confirm each job submission."
        echo ""
        
        jobs=("01_baseline.slurm:Baseline CNN-RNN"
              "02_vit_rgts_0reg.slurm:ViT-RGTS (0 registers)"
              "03_vit_rgts_2reg.slurm:ViT-RGTS (2 registers)"
              "04_vit_rgts_4reg.slurm:ViT-RGTS (4 registers)"
              "05_vit_rgts_8reg.slurm:ViT-RGTS (8 registers)"
              "06_vit_rgts_16reg.slurm:ViT-RGTS (16 registers)"
              "07_torchvision_vit_b16.slurm:TorchVision ViT-B/16"
              "08_trocr_base.slurm:TrOCR Base")
        
        for job_info in "${jobs[@]}"; do
            script="${job_info%%:*}"
            desc="${job_info##*:}"
            
            read -p "Submit $desc? [y/n]: " answer
            if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
                submit_job "$script" "$desc"
            else
                echo "  Skipped"
                echo ""
            fi
        done
        ;;
        
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo "========================================="
echo "Submission Complete!"
echo "========================================="
echo ""
echo "To check job status, use:"
echo "  squeue -u \$USER"
echo ""
echo "To check logs in real-time:"
echo "  tail -f experiments_execution/logs/*/output_*.log"
echo ""
echo "To cancel all jobs:"
echo "  scancel -u \$USER"
echo ""
echo "Logs are organized in:"
echo "  - experiments_execution/logs/baseline/"
echo "  - experiments_execution/logs/vit_rgts_registers/"
echo "  - experiments_execution/logs/pretrained/"
echo ""
