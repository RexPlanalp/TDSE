#!/usr/bin/bash
#SBATCH --job-name TDSE
#SBATCH -p jila
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=32G
#SBATCH -o run.log 
#SBATCH -t 0-01:00:00


REPO_DIR="/users/becker/dopl4670/Research/TDSE/other/testing"
                                                               
hostname
pwd

mpiexec -n $SLURM_NTASKS python $REPO_DIR/multiprocessing_tests.py >> results.log




