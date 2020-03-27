#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-04:00
#SBATCH --mem=60000
#SBATCH -o /n/home13/kdai/multi-context-gen/output_%j.out
#SBATCH -e /n/home13/kdai/multi-context-gen/errors_%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --gpu-freq=high

module load cuda/9.0-fasrc02
module load Anaconda3/5.0.1-fasrc01
source activate myenv
python ./encoder-agnostic-model/train.py -config ./encoder-agnostic-model/config/ac/transformer_ac_psa.yml -run_name psa -gpt2_params_path ./encoder-agnostic-model/gpt2/models/124M/ -gpt2_init_embanddec
