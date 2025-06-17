#!/bin/bash

sbatch slurm/cl9_experimentA.slurm
sbatch slurm/cl9_experimentB.slurm

# Define your array of numbers
numbers=(0 0.5 1 2 3 4 6)  # You can change these values as needed

# Get current date in YYYYMMDD format
current_date=$(date +"%Y%m%d_%H%M%S")

# Create output directory
output_dir="auto_generated_${current_date}"
mkdir -p "$output_dir"

# Loop over each number
for num in "${numbers[@]}"; do
    # Generate new config path
    config_path="${output_dir}/${num}.config"

    # Modify config file
    sed -e "s|^data_dir .*|data_dir exp_cl1_${num}|" \
        -e "s|^t0_offset .*|t0_offset ${num}|" \
        params/cl1_experiment.config > "$config_path"

    # Generate new slurm script path
    slurm_path="${output_dir}/${num}.slurm"

    # Modify slurm file
    sed -e "s|^#SBATCH --job-name=exp|#SBATCH --job-name=exp_${current_date}|" \
        -e "s|^#SBATCH --output=.*/output_exp.txt|#SBATCH --output=/home/althueser/phd/cpp/HigherHarmonics/output_exp_${current_date}.txt|" \
        -e "s|mpirun ./build_cluster/hhg .*|mpirun ./build_cluster/hhg ${config_path}|" \
        slurm/cl9_experiment.slurm > "$slurm_path"
    
    sbatch "$slurm_path"
done

