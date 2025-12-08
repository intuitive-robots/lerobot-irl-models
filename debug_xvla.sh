#!/bin/bash
#SBATCH -p dev_accelerated #accelerated
#SBATCH --mem=64G        # Total CPU RAM
#SBATCH --gres=gpu:2     # 4 GPUs
#SBATCH --time=00:10:00
# SBATCH --cpus-per-task=6
#SBATCH -J debug_xvla
#SBATCH -o logs/debug_xvla/%x_%j.out
#SBATCH -e logs/debug_xvla/%x_%j.err
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
GLOBAL_BATCH_SIZE=64 #64 # Total batch size across all GPUs (e.g., 64)
BATCH_PER_GPU=$(($GLOBAL_BATCH_SIZE / $NUM_GPUS))
echo "Number of GPUs detected: $NUM_GPUS"
echo "Batch size per GPU: $BATCH_PER_GPU"
echo "Launching on GPUs: $SLURM_JOB_GPUS"

now=$(date +%Y-%m-%d/%H-%M-%S)
OUTPUT_DIR="./output/debug/flower/${now}"
# ==============================================================================
# EXECUTE TRAINING (Conditional Launch)
# ==============================================================================
if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU training using accelerate launch
    accelerate launch \
        --multi_gpu \
        --num_processes=$NUM_GPUS \
        --main_process_port $MASTER_PORT \
        src/train_xvla.py \
        wandb.enable=True \
        train.steps=1000 \
        train.batch_size=$BATCH_PER_GPU \
        train.output_dir=$OUTPUT_DIR \
        policy.pretrained_path="/home/hk-project-p0024638/usmrd/projects/lerobot-irl-models/output/train/xvla/2025-12-07/17-56-59/model_outputs/checkpoints/020000/pretrained_model"
        # policy.action_mode="custom_joint8"  \
        # repo_id="test" \
        # dataset_path="/hkfs/work/workspace/scratch/usmrd-MemVLA/datasets/lerobot/test" \
else
    # Single-GPU (or CPU) training using standard python call
    python src/train_xvla.py \
        wandb.enable=False \
        train.steps=500 \
        train.batch_size=$BATCH_PER_GPU \
        train.output_dir=$OUTPUT_DIR
fi
