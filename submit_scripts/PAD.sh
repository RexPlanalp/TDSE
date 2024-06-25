#!/usr/bin/bash
#SBATCH --job-name PAD
#SBATCH -p jila
#SBATCH --nodes 1
#SBATCH --ntasks 32
#SBATCH --mem=128G
#SBATCH -o run.log 
#SBATCH -t 0-03:00:00


REPO_DIR="/users/becker/dopl4670/Research/TDSE/Analysis"
                                                             

hostname
pwd

mpiexec -n $SLURM_NTASKS python $REPO_DIR/Parallel_Angular.py >> results.log


