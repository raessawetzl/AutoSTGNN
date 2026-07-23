#!/bin/bash
#SBATCH --job-name=dehb_dcrnn
#SBATCH --output=/home/mnycha010/AutoSTGNN/results/dehb_%j.out
#SBATCH --error=/home/mnycha010/AutoSTGNN/results/dehb_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $SLURMD_NODENAME"
echo "Start time: $(date)"

module load python/miniconda3-py3.12

cd ~/AutoSTGNN/src

python dehb_dcrnn.py

echo "Done: $(date)"
