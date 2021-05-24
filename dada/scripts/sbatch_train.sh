#!/bin/bash
#SBATCH  --output=baseline_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

source /itet-stor/qwarraich/net_scratch/conda/bin/activate
conda activate pytcuda10
python train.py --cfg ./configs_s2c/dada.yml --tensorboard
