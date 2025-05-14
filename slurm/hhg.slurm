#!/bin/bash

## For more information on the following SBATCH commands, please refer to the manual page (man sbatch) in the terminal
## or look up the slurm documentation under https://slurm.schedmed.com/sbatch.html

## Mandatory:
#SBATCH --job-name=hhg
#SBATCH --output=/home/althueser/phd/cpp/HigherHarmonics/output.txt
#SBATCH --time=168:00:00         ## maximum runtime; hours:minutes:seconds
#SBATCH --partition=ultralong       ## choose queue

#SBATCH --ntasks=120             ## sets number of total mpi processes
#SBATCH --tasks-per-node=40    ## mpi processes spawned per node
#SBATCH --cpus-per-task=1       ## logical cores used per mpi process: should be 1, except if you want to combine OpenMP with$

##SBATCH --mem=40gb               ## sets maximum allowed memory per node
#SBATCH --mem-per-cpu=1gb    ## sets maximum allowed memory per mpi process

#SBATCH --mail-user=joshua.althueser@tu-dortmund.de     ## replace mail by personal mail address
#SBATCH --mail-type=END ## most relevant options: NONE, BEGIN, END, FAIL

## Optional:
#SBATCH --hint=nomultithread    ## deactivate Hyperthreading (recommended); for Hyperthreading comment out this line
#SBATCH --constraint=CascadeLake       ## chose a specific feature, e.g., only nodes with Haswell-architecture
                                ## Feature-Output by "cat /etc/slurm/slurm.conf | grep Feature"
module purge
module load gcc/11.3.0-full
module load mpi/openmpi3-x86_64
module load boost/1.71.0
module load cmake/3.23.2
cd /home/althueser/phd/cpp/HigherHarmonics/        ## go to working directory

date
echo "--- START ---"

## execute binary using mpirun on the allocated computation resource; the number of cores is $
mpirun ./build_cluster/hhg params/cluster.config

echo "--- END ---"
date
echo
echo "$(whoami) is leaving from $(hostname) ..."
echo
