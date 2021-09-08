#!/bin/bash

#SBATCH --job-name="gan"
#SBATCH --time=1:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=2   # number of nodes
#SBATCH --ntasks-per-node=8   # 8 processor core(s) per node 
#SBATCH --mem=10G   # maximum memory per node
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu    # gpu node(s)

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

#LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

export HDF5_USE_FILE_LOCKING='FALSE'  # for exporting hd5 file
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

source .env/bin/activate # activate your environment

python src/GAN/train.py \
	--data_dir ./data/L_128/p_0.5928 \
	--batch_size 10 \
	--epochs 10 \
	--noise_dim 100 \
	--save_dir ./data/models/gan
