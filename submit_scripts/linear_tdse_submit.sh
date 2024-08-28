#!/usr/bin/bash
#SBATCH --job-name LINEAR_TDSE
#SBATCH -p jila
#SBATCH --nodes 1
#SBATCH --ntasks 32
#SBATCH --mem=64G
#SBATCH -o run.log 
#SBATCH -t 0-08:00:00

#SBATCH --exclude=node70
#SBATCH --exclude=node72
#SBATCH --exclude=node73

#SBATCH --exclude=node80
#SBATCH --exclude=node81
#SBATCH --exclude=node82
#SBATCH --exclude=node83


REPO_DIR="/users/becker/dopl4670/Research/TDSE/Linear_TDSE"
                                                               
hostname
pwd

mpiexec -n $SLURM_NTASKS python $REPO_DIR/tdse_main.py >> results.log




