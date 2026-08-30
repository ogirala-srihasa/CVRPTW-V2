#!/bin/bash
#SBATCH --job-name=cvrptw_par
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=04:00:00
#SBATCH --partition=small
#SBATCH --output=job.%J.out
#SBATCH --error=job.%J.err

cd $SLURM_SUBMIT_DIR
export OMP_NUM_THREADS=48
bash test.sh --parallel
