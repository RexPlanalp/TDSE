#!/usr/bin/bash
#SBATCH --job-name TISE
#SBATCH -p jila
#SBATCH --nodes 1
#SBATCH --ntasks 32
#SBATCH --mem=32G
#SBATCH -o run.log 
#SBATCH -t 0-01:00:00

#SBATCH --exclude=node78
#SBATCH --exclude=node79
#SBATCH --exclude=node81
#SBATCH --exclude=node82
#SBATCH --exclude=node83

REPO_DIR="/users/becker/dopl4670/Research/TDSE/TISE"
                                                               
hostname
pwd

mpiexec -n $SLURM_NTASKS python $REPO_DIR/tise_main.py >> results.log




