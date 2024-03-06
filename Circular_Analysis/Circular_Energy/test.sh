#!/usr/bin/bash
#SBATCH --job-name PAD
#SBATCH -p jila
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=64G
#SBATCH -o run.log 
#SBATCH -t 0-03:00:00
#SBATCH --exclude=node48

REPO_DIR="//data/becker/dopl4670/TDSE_Jobs/TEST_CIRCULAR"
                                                             

hostname
pwd

mpiexec -n $SLURM_NTASKS python $REPO_DIR/Photoenergy.py $1 $2 $3 >> photo_files/results$3.log


