#!/bin/bash

## For more information on the following SBATCH commands, please refer to the manual page (man sbatch) in the terminal
## or look up the slurm documentation under https://slurm.schedmed.com/sbatch.html

## Mandatory:
#SBATCH --job-name=exp
#SBATCH --output=/home/althueser/phd/cpp/HigherHarmonics/output_exp.txt
#SBATCH --time=48:00:00         ## maximum runtime; hours:minutes:seconds
#SBATCH --partition=long       ## choose queue

#SBATCH --ntasks=160             ## sets number of total mpi processes
#SBATCH --tasks-per-node=40    ## mpi processes spawned per node
#SBATCH --cpus-per-task=1       ## logical cores used per mpi process: should be 1, except if you want to combine OpenMP with$

##SBATCH --mem=40gb               ## sets maximum allowed memory per node
#SBATCH --mem-per-cpu=128mb    ## sets maximum allowed memory per mpi process

#SBATCH --mail-user=joshua.althueser@tu-dortmund.de     ## replace mail by personal mail address
#SBATCH --mail-type=END ## most relevant options: NONE, BEGIN, END, FAIL

## Optional:
#SBATCH --hint=nomultithread    ## deactivate Hyperthreading (recommended); for Hyperthreading comment out this line
#SBATCH --constraint=CascadeLake       ## chose a specific feature, e.g., only nodes with Haswell-architecture
                                ## Feature-Output by "cat /etc/slurm/slurm.conf | grep Feature"
module purge
module load mpi
cd /home/althueser/phd/cpp/HigherHarmonics/        ## go to working directory

date
echo "--- START ---"

## execute binary using mpirun on the allocated computation resource; the number of cores is $
mpirun ./build_cluster/hhg params/cl1_experiment.config

echo "--- END ---"
date
echo
echo "$(whoami) is leaving from $(hostname) ..."
echo
