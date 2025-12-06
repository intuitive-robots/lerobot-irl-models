#!/bin/bash
#SBATCH -p dev_accelerated
#SBATCH --gres=gpu:2 #1
#SBATCH --mem=32 #16G
#SBATCH --time=00:10:00  #40.000 steps should take ~6:45h
#SBATCH -J debug_flower
#SBATCH -o logs/debug/%x_%j.out
#SBATCH -e logs/debug/%x_%j.err

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================
source ~/.bashrc
conda activate lerobot-irl-models

# PyTorch/CUDA environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

# Set the master port for distributed training (DDP)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "Master Port: $MASTER_PORT"

# ==============================================================================
# BATCH SIZE CALCULATION
# ==============================================================================
# Split global batchsize defined in hydra config across all available GPUs
# IMPORTANT: make sure that $GLOBAL_BATCH_SIZE aligns with your training yaml
NUM_GPUS=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)
GLOBAL_BATCH_SIZE=64 # Total batch size across all GPUs (e.g., 64)
BATCH_PER_GPU=$(($GLOBAL_BATCH_SIZE / $NUM_GPUS))
echo "Number of GPUs detected: $NUM_GPUS"
echo "Batch size per GPU: $BATCH_PER_GPU"
echo "Launching on GPUs: $SLURM_JOB_GPUS"

# NOTE: $TMPDIR usage not implemented for debug script, avoid long trainings with many IO calls

# ==============================================================================
# EXECUTE TRAINING (Conditional Launch)
# ==============================================================================
if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU training using accelerate launch
    accelerate launch \
        --multi_gpu \
        --num_processes=$NUM_GPUS \
        --main_process_port $MASTER_PORT \
        src/train_flower.py \
        wandb.enable=false \
        train.batch_size=$BATCH_PER_GPU \
        repo_id=test \
        dataset_path=/hkfs/work/workspace/scratch/usmrd-MemVLA/datasets/lerobot/test \
        train.steps=1000 \
        train.save_freq=1000 \
        model.freeze_florence=false \
        model.freeze_vision_tower=false
else
    # Single-GPU (or CPU) training using standard python call
    python src/train_flower.py \
        dataset_path=$TMPDIR/$DATASET_ID \
        train.batch_size=$BATCH_PER_GPU
fi
