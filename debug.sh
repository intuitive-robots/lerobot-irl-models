#!/bin/bash
#SBATCH -p dev_accelerated
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:16:00  #40.000 steps should take ~6:45h
#SBATCH -J debug_flower
#SBATCH -o logs/debug/%x_%j.out
#SBATCH -e logs/debug/%x_%j.err

source ~/.bashrc
conda activate lerobot-irl-models

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

# Start training
python src/train_flower.py \
    wandb.enable=false \
    dataset.repo_id=test \
    dataset.dataset_path=/hkfs/work/workspace/scratch/usmrd-MemVLA/datasets/lerobot/test \
    train.batch_size=16\
    train.steps=1000 \
    train.save_freq=1000 \
    model.freeze_florence=false \
    model.freeze_vision_tower=false
