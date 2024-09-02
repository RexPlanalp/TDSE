#!/usr/bin/bash
#SBATCH --job-name PES
#SBATCH -p jila
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=32G
#SBATCH -o run.log 
#SBATCH -t 0-03:00:00

#SBATCH --exclude=node82
#SBATCH --exclude=node81
#SBATCH --exclude=node80
#SBATCH --exclude=node48


REPO_DIR="/users/becker/dopl4670/Research/TDSE/Plotting"
                                                             

hostname
pwd

mpiexec -n $SLURM_NTASKS python $REPO_DIR/Cont_PES.py >> results.log


