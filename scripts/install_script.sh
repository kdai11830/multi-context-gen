#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1

module load Anaconda3/5.0.1-fasrc01
module load cuda/9.0-fasrc02 cudnn/7.4.1.5_cuda9.0-fasrc01
conda create -n myenv python=3.6 numpy six wheel
source activate myenv
python
import nltk
nltk.download('punkt')
