#!/bin/bash
 
#SBATCH -p gpu-long --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=sheneman@uidaho.edu

hostname
nvidia-smi -L

module load python/3.8.1 openmpi/1.10.2 cuda/11.2

source ./venv/bin/activate 

python ./tigervid.py inputs outputs --gpu --jobs 4
