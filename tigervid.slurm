#!/bin/bash

#SBATCH -p gpu-8 --gres=gpu:1 --nodelist=n001
#SBATCH --mail-type=END
#SBATCH --mail-user=sheneman@uidaho.edu

hostname
nvidia-smi -L

module load python/3.8.1 openmpi/1.10.2 cuda/11.8

source ./venv/bin/activate 

python -u ./tigervid.py inputs/outdoor_reolink_01/a/ outputs --gpu --jobs 12 --nobar
