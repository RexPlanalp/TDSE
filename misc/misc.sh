#!/usr/bin/bash
#SBATCH --job-name misc
#SBATCH -p jila
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=32G
#SBATCH -o run.log 
#SBATCH -t 0-03:00:00


REPO_DIR="/users/becker/dopl4670/Research/TDSE/misc"
                                                             

hostname
pwd

mpiexec -n $SLURM_NTASKS python $REPO_DIR/Position_Space_PES.py >> results.log


