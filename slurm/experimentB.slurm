#!/bin/bash

## For more information on the following SBATCH commands, please refer to the manual page (man sbatch) in the terminal
## or look up the slurm documentation under https://slurm.schedmed.com/sbatch.html

## Mandatory:
#SBATCH --job-name=expB
#SBATCH --output=/home/althueser/phd/cpp/HigherHarmonics/output_expB.txt
#SBATCH --time=48:00:00         ## maximum runtime; hours:minutes:seconds
#SBATCH --partition=long        ## choose queue

#SBATCH --ntasks=224            ## sets number of total mpi processes
#SBATCH --tasks-per-node=56     ## mpi processes spawned per node
#SBATCH --cpus-per-task=1       ## logical cores used per mpi process: should be 1, except if you want to combine OpenMP with$

##SBATCH --mem=40gb             ## sets maximum allowed memory per node
#SBATCH --mem-per-cpu=128mb     ## sets maximum allowed memory per mpi process

#SBATCH --mail-user=joshua.althueser@tu-dortmund.de     ## replace mail by personal mail address
#SBATCH --mail-type=END ## most relevant options: NONE, BEGIN, END, FAIL

## Optional:
#SBATCH --hint=nomultithread    ## deactivate Hyperthreading (recommended); for Hyperthreading comment out this line
#SBATCH --constraint=IceLake    ## chose a specific feature, e.g., only nodes with Haswell-architecture
                                ## Feature-Output by "cat /etc/slurm/slurm.conf | grep Feature"
module purge

cpu_family=$(grep -m1 "cpu family" /proc/cpuinfo | awk '{print $4}')
cpu_model=$(grep -m1 "model[^ name]" /proc/cpuinfo | awk '{print $3}')

if [[ "$cpu_family" == "6" && "$cpu_model" == "85" ]]; then
    echo "Cascade Lake detected"
    module load gcc/11.3.0-full mpi/openmpi3-x86_64 boost/1.71.0 cmake/3.23.2
elif [[ "$cpu_family" == "6" && "$cpu_model" == "106" ]]; then
    echo "Ice Lake Server detected"
    module load mpi
else
    echo "Unknown microarchitecture (family $cpu_family, model $cpu_model)"
fi

cd /home/althueser/phd/cpp/HigherHarmonics/        ## go to working directory

date
echo "--- START ---"

## execute binary using mpirun on the allocated computation resource; the number of cores is $
mpirun ./build_cluster/hhg params/cl1_experiment_B.config

echo "--- END ---"
date
echo
echo "$(whoami) is leaving from $(hostname) ..."
echo
