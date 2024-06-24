#!/usr/bin/bash
#SBATCH --job-name TISE
#SBATCH -p jila
#SBATCH --nodes 1
#SBATCH --ntasks 32
#SBATCH --mem=64G
#SBATCH -o run.log 
#SBATCH -t 0-01:00:00

#SBATCH --exclude=node70
#SBATCH --exclude=node72


REPO_DIR="/users/becker/dopl4670/Research/TDSE_refactored/Linear_TDSE"
                                                               
hostname
pwd

mpiexec -n $SLURM_NTASKS python $REPO_DIR/tdse_main.py >> results.log




