#!/usr/bin/bash
#SBATCH --job-name PAD
#SBATCH -p jila
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=64G
#SBATCH -o run.log 
#SBATCH -t 0-01:00:00


REPO_DIR="/users/becker/dopl4670/Research/TDSE/Circular_Analysis/Circular_Population"
                                                             

hostname
pwd

mpiexec -n $SLURM_NTASKS python $REPO_DIR/Population.py >> results.log


