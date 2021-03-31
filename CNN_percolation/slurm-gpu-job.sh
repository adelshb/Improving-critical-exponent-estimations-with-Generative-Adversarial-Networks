#!/bin/bash

#SBATCH --job-name="perc-training"
#SBATCH --output="%j.out" # job standard output file (%j replaced by job id)
#SBATCH --error="%j.err" # job standard error file (%j replaced by job id)

#SBATCH --time=48:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 8 processor core(s) per node 
#SBATCH --mem=5G   # maximum memory per node
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu    # gpu node(s)


#========================================================
# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

export HDF5_USE_FILE_LOCKING='FALSE'  # for exporting hd5 file in tf
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

conda activate tf-gpu # activate your environment

python3 main.py --epochs 1000 --n_configs_per_p 2000 --n_gpus 4
