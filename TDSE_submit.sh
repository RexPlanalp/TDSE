#!/usr/bin/bash
#SBATCH --job-name TDSE
#SBATCH -p photons
#SBATCH --nodes 1
#SBATCH --ntasks 16
#SBATCH --mem=32G 
#SBATCH -o run.log 
#SBATCH -t 0-01:00:00
#SBATCH --exclude=photon13

REPO_DIR="/home/becker/dopl4670/TDSE/Solver"
#module purge                                                                                 
#module load intel/2019.0                                                                     
#module load openmpi                                                                          
#module load hdf5                                                                           
#module load cmake                                                                            
#module load lapack                                                                           
#module load blas                                                                             
#module load intel-gcc/6.3.0                                                                  

hostname
pwd

mpiexec -n $SLURM_NTASKS python $REPO_DIR/TDSE.py >> results.log


