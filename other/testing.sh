#!/usr/bin/bash
#SBATCH --job-name PES
#SBATCH -p jila
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=32G 
#SBATCH -o run.log 
#SBATCH -t 0-01:00:00


REPO_DIR="/users/becker/dopl4670/Research/TDSE/Circular/Circular_Analysis/Circular_EnergySpectrum/Circular_Population"
                                                             

hostname
pwd

mpiexec -n $SLURM_NTASKS python $REPO_DIR/Population.py >> results.log


