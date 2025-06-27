#!/bin/bash

# Set architecture: "IceLake" or "CascadeLake"
arch="IceLake"

sbatch slurm/experimentA.slurm
sbatch slurm/experimentB.slurm

shifts=(0 0.25 0.5 1 2 3 4 6)
current_date=$(date +"%Y%m%d_%H%M%S")

output_dir="auto_generated_${current_date}"
mkdir -p "$output_dir"

for num in "${shifts[@]}"; do
    config_path="${output_dir}/${num}.config"
    sed -e "s|^t0_offset .*|t0_offset ${num}|" \
        params/cl1_experiment.config > "$config_path"

    slurm_path="${output_dir}/${num}.slurm"
    sed -e "s|^#SBATCH --job-name=exp|#SBATCH --job-name=exp_${current_date}_${num}|" \
        -e "s|^#SBATCH --output=.*/output_exp.txt|#SBATCH --output=/home/althueser/phd/cpp/HigherHarmonics/output_exp_${current_date}_${num}.txt|" \
        -e "s|^#SBATCH --constraint=.*|#SBATCH --constraint=${arch}|" \
        -e "s|mpirun ./build_.*/hhg .*|mpirun ./build_${arch}/hhg ${config_path}|" \
        slurm/experiment.slurm > "$slurm_path"
    
    sbatch "$slurm_path"
done
