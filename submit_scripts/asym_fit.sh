#!/usr/bin/bash
#SBATCH --job-name fit
#SBATCH -p jila
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=16G
#SBATCH -o run.log 
#SBATCH -t 0-00:30:00


REPO_DIR="/users/becker/dopl4670/Research/TDSE/Plotting"
                                                             

hostname
pwd

mpiexec -n $SLURM_NTASKS python $REPO_DIR/FitA_Plot.py >> results.log


