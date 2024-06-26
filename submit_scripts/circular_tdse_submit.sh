#!/usr/bin/bash
#SBATCH --job-name CIRCULAR_TDSE
#SBATCH -p jila
#SBATCH --nodes 1
#SBATCH --ntasks 32
#SBATCH --mem=64G
#SBATCH -o run.log 
#SBATCH -t 1-00:00:00
#SBATCH --exclude=node80
#SBATCH --exclude=node81
#SBATCH --exclude=node82
#SBATCH --exclude=node83


REPO_DIR="/users/becker/dopl4670/Research/TDSE/Circular_TDSE"
                                                               
hostname
pwd

mpiexec -n $SLURM_NTASKS python $REPO_DIR/tdse_main.py >> results.log




