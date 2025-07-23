#!/bin/bash

# Set architecture: "IceLake" or "CascadeLake"
arch="CascadeLake"

shifts=(0)
current_date=$(date +"%Y%m%d_%H%M%S")

output_dir="auto_generated_${current_date}"
mkdir -p "$output_dir"

cp params/cascade_experiment_A.config "$output_dir/A.config"
sed -e "s|mpirun ./build_.*/hhg .*|mpirun ./build_${arch}/hhg ${output_dir}/A.config|" \
        slurm/cascade_experiment.slurm > "${output_dir}/A.slurm"

cp params/cascade_experiment_B.config "$output_dir/B.config"
sed -e "s|mpirun ./build_.*/hhg .*|mpirun ./build_${arch}/hhg ${output_dir}/B.config|" \
        slurm/cascade_experiment.slurm > "${output_dir}/B.slurm"

sbatch "${output_dir}/A.slurm"
sbatch "${output_dir}/B.slurm"

for num in "${shifts[@]}"; do
    config_path="${output_dir}/${num}.config"
    sed -e "s|^t0_offset .*|t0_offset ${num}|" \
        params/cascade_experiment.config > "$config_path"

    slurm_path="${output_dir}/${num}.slurm"
    sed -e "s|^#SBATCH --job-name=exp|#SBATCH --job-name=exp_${current_date}_${num}|" \
        -e "s|^#SBATCH --output=.*/output_exp.txt|#SBATCH --output=/home/althueser/phd/cpp/HigherHarmonics/output_exp_${current_date}_${num}.txt|" \
        -e "s|^#SBATCH --constraint=.*|#SBATCH --constraint=${arch}|" \
        -e "s|mpirun ./build_.*/hhg .*|mpirun ./build_${arch}/hhg ${config_path}|" \
        slurm/cascade_experiment.slurm > "$slurm_path"
    
    sbatch "$slurm_path"
done
