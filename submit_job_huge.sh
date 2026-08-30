#!/bin/bash
#SBATCH --job-name=cvrptw_huge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=12:00:00
#SBATCH --partition=small
#SBATCH --output=job_huge.%J.out
#SBATCH --error=job_huge.%J.err

cd $SLURM_SUBMIT_DIR
export OMP_NUM_THREADS=48
bash test_huge.sh --parallel
