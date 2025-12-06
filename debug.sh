#!/bin/bash
#SBATCH -p dev_accelerated
#SBATCH --gres=gpu:2 #1
#SBATCH --mem=32 #16G
#SBATCH --time=00:10:00  #40.000 steps should take ~6:45h
#SBATCH -J debug_flower
#SBATCH -o logs/debug/%x_%j.out
#SBATCH -e logs/debug/%x_%j.err


source ~/.bashrc
conda activate lerobot-irl-models

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

# --- CONFIGURATION FOR DEBUGGING MORE THAN 1 GPU RUNS ---
NUM_GPUS=2 #4 #check that this is correct
GLOBAL_BATCH_SIZE=32 #64
BATCH_PER_GPU=$(($GLOBAL_BATCH_SIZE / $NUM_GPUS))

# Get a unique port for this job to prevent collisions if multiple jobs run on the node
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "Master Port: $MASTER_PORT"
echo "Launching on GPUs: $SLURM_JOB_GPUS"

accelerate launch \
    --multi_gpu \
    --num_processes=$NUM_GPUS \
    --main_process_port $MASTER_PORT \
    src/train_flower.py \
    train.batch_size=$BATCH_PER_GPU \
    train.resume=True \
    train.checkpoint_path=/home/hk-project-p0024638/usmrd/projects/lerobot-irl-models/output/train/flower/2025-12-02/20-37-39/model_outputs/checkpoints/last/pretrained_model \
    model.scheduler.num_warmup_steps=1000 \
    model.scheduler.num_plateau_steps=15000 \
    model.scheduler.num_decay_steps=40000
    train_steps=500


# --- CONFIGURATION FOR DEBUGGING A SINGLE GPU RUN ---
# # Start training
# python src/train_flower.py \
#     wandb.enable=false \
#     dataset.repo_id=test \
#     dataset.dataset_path=/hkfs/work/workspace/scratch/usmrd-MemVLA/datasets/lerobot/test \
#     train.batch_size=16\
#     train.steps=1000 \
#     train.save_freq=1000 \
#     model.freeze_florence=false \
#     model.freeze_vision_tower=false
