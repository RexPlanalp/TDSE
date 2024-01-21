#!/usr/bin/bash
#SBATCH --job-name TDSE
#SBATCH -p jila
#SBATCH --nodes 1
#SBATCH --ntasks 32
#SBATCH --mem=144G
#SBATCH -o run.log 
#SBATCH -t 0-01:00:00


REPO_DIR="/users/becker/dopl4670/Research/TDSE/Solver"
                                                               
hostname
pwd

mpiexec -n $SLURM_NTASKS python $REPO_DIR/TDSE.py >> results.log




