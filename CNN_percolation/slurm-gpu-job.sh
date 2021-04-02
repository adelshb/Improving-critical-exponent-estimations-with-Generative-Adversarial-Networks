#!/bin/bash

#SBATCH --job-name="perc"
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

export HDF5_USE_FILE_LOCKING='FALSE'  # for exporting hd5 file
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

conda activate tf-gpu # activate your environment

python3 main.py             \
 --odir "saved-files"      \
 --L 128                    \
 --p_down 0.5               \
 --p_up 0.7                 \
 --p_increment 0.01         \
 --round_digit 2            \
 --epochs 100              \
 --n_configs_per_p 1000     \
 --n_gpus 1                 \
 --patience 10              \
 --test_size 0.2            \
 --random_state 42
