#!/usr/bin/bash
#SBATCH --job-name TDSE
#SBATCH -p jila
#SBATCH --nodes 1
#SBATCH --ntasks 32
#SBATCH --mem=172G
#SBATCH -o run.log 
#SBATCH -t 1-00:00:00
#SBATCH --exclude=node80
#SBATCH --exclude=node83

REPO_DIR="/users/becker/dopl4670/Research/TDSE/Circular_Solver"
                                                               
hostname
pwd

mpiexec -n $SLURM_NTASKS python $REPO_DIR/TDSE.py >> results.log




