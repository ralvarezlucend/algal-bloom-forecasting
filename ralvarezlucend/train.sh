#!/bin/bash

#SBATCH --job-name="unet-regression"
#SBATCH --account=Education-EEMCS-Courses-CSE3000
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output="train%j.out"

# Activate conda environment.
source activate /scratch/${USER}/algal-bloom/envs/geo

# Run code.
srun python main.py
