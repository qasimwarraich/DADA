#!/bin/bash
#SBATCH  --output=log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

source /scratch_net/biwidl217/rokaushik/conda/bin/activate
conda activate pytcu10
python train.py --cfg ./configs_s2c/dada.yml
