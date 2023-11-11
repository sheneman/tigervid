#!/bin/bash
 
#SBATCH -p gpu-long --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=sheneman@uidaho.edu

hostname
nvidia-smi -L

module load python/3.8.1 openmpi/1.10.2 cuda/11.8

source ./venv/bin/activate 

python -u ./tigervid.py inputs1 outputs --gpu --jobs 4 --progressbar None
