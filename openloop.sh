#!/bin/bash
#SBATCH -p dev_accelerated
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00  #40.000 steps should take ~6:45h
#SBATCH -J compare
#SBATCH -o logs/compare/%x_%j.out
#SBATCH -e logs/compare/%x_%j.err

source ~/.bashrc
conda activate lerobot-irl-models

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

# Start training
python src/utils/openloop.py \
    checkpoint_path=/home/hk-project-p0024638/usmrd/projects/lerobot-irl-models/output/train/flower/2025-12-03/11-00-42/model_outputs/checkpoints/last/pretrained_model/model.safetensors
    # checkpoint_path=/home/hk-project-p0024638/usmrd/projects/lerobot-irl-models/output/train/flower/2025-12-03/09-25-35/model_outputs/checkpoints/last/pretrained_model/model.safetensors
