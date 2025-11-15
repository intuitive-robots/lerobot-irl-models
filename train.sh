#!/bin/bash
#SBATCH -p accelerated
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH -J train_flower
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

source ~/.bashrc
conda activate lerobot-irl-models

export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

# Start training
accelerate launch --main_process_port 29500 \
    src/train_flower3.py


