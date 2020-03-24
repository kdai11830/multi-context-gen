#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-01:00
#SBATCH --mem=5000
#SBATCH -o /n/home13/kdai/dialog-output_%j.out
#SBATCH -e /n/home13/kdai/dialog-errors_%j.err

module load Anaconda3/5.0.1-fasrc01
source activate myenv
python ./dialog-extractor.py