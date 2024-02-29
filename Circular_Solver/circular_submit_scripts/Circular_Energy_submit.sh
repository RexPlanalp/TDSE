#!/usr/bin/bash
#SBATCH --job-name PAD
#SBATCH -p jila
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=64G
#SBATCH -o run.log 
#SBATCH -t 0-04:00:00


REPO_DIR="/users/becker/dopl4670/Research/TDSE/Circular_Analysis/Circular_EnergySpectrum"
                                                             

hostname
pwd

mpiexec -n $SLURM_NTASKS python $REPO_DIR/run_photoenergy.py >> results.log


