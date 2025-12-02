#!/bin/bash
#SBATCH -p accelerated
#SBATCH --mem=64G        # Total CPU RAM
#SBATCH --gres=gpu:4     # 4 GPUs
#SBATCH --time=07:30:00
#SBATCH -J train_multi_gpu
#SBATCH -o logs/train_multi_gpu/%x_%j.out
#SBATCH -e logs/train_multi_gpu/%x_%j.err

source ~/.bashrc
conda activate lerobot-irl-models

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

# --- CONFIGURATION ---
NUM_GPUS=4 #4 #check that this is correct
GLOBAL_BATCH_SIZE=64 #64
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
    train.batch_size=$BATCH_PER_GPU
