#!/bin/bash

#SBATCH --job-name=ddpm-microscopy
#SBATCH --output=out_sbatch/%j.out
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

date

python3 run.py\
 --config config_train/restoration_mitochondria_edm2.json\
 --phase train --batch 8 --gpu 0

echo job finished
date