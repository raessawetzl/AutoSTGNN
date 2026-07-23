#!/bin/bash
#SBATCH --job-name=train_dcrnn
#SBATCH --output=/home/mnycha010/AutoSTGNN/results/train_dcrnn_%j.out
#SBATCH --error=/home/mnycha010/AutoSTGNN/results/train_dcrnn_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $SLURMD_NODENAME"
echo "Start time: $(date)"

module load python/miniconda3-py3.12

cd ~/AutoSTGNN/src

python train_dcrnn.py

echo "Done: $(date)"
