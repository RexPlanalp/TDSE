#!/usr/bin/bash
#SBATCH --job-name PAD
#SBATCH -p jila
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=32G
#SBATCH -o run.log 
#SBATCH -t 0-08:00:00


REPO_DIR="/users/becker/dopl4670/Research/TDSE/Circular/Circular_Spectrum"
                                                             

hostname
pwd

mpiexec -n $SLURM_NTASKS python $REPO_DIR/Photoenergy.py >> results.log


