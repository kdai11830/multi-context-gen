#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-04:00
#SBATCH --mem=60000
#SBATCH -o /n/home13/kdai/output_%j.out
#SBATCH -e /n/home13/kdai/errors_%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --gpu-freq=high

module load cuda/9.0-fasrc02
module load Anaconda3/5.0.1-fasrc01
source activate myenv
python ./train.py
