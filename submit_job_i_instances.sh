#!/bin/bash
#SBATCH --job-name=cvrptw_i_instances
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=24:00:00
#SBATCH --partition=small
#SBATCH --output=job_i_instances.%J.out
#SBATCH --error=job_i_instances.%J.err

cd $SLURM_SUBMIT_DIR
export OMP_NUM_THREADS=48
bash test_i_instances.sh --parallel
