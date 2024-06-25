#!/usr/bin/bash
#SBATCH --job-name coeff
#SBATCH -p jila
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=32G
#SBATCH -o run.log 
#SBATCH -t 0-03:00:00


REPO_DIR="/users/becker/dopl4670/Research/TDSE/Analysis"
                                                             

hostname
pwd

mpiexec -n $SLURM_NTASKS python $REPO_DIR/Parallel_Coeff.py >> results.log


