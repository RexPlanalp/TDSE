#!/usr/bin/bash
#SBATCH --job-name PES
#SBATCH -p jila
#SBATCH --nodes 1
#SBATCH --ntasks 16
#SBATCH --mem=128G
#SBATCH -o run.log 
#SBATCH -t 0-03:00:00


REPO_DIR="/users/becker/dopl4670/Research/TDSE_refactored/Analysis_Refactored"
                                                             

hostname
pwd

mpiexec -n $SLURM_NTASKS python $REPO_DIR/Parallel_Photospectrum.py >> results.log


