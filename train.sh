#!/bin/bash
#SBATCH -p accelerated #accelerated
#SBATCH --mem=64G        # Total CPU RAM
#SBATCH --gres=gpu:4     # 4 GPUs
#SBATCH --time=07:30:00
#SBATCH -J train_multi_gpu
#SBATCH -o logs/train_multi_gpu/%x_%j.out
#SBATCH -e logs/train_multi_gpu/%x_%j.err

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================
source ~/.bashrc
conda activate lerobot-irl-models

# PyTorch/CUDA environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export WANDB_LOG_MODEL=false #do not upload model to wandb (storage limitations)

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

# ==============================================================================
# DATA HANDLING: Extract Data to Local Scratch ($TMPDIR)
# ==============================================================================
# Reduce IO-Activity, see:  https://www.nhr.kit.edu/userdocs/horeka/filesystems/#tmpdir
DATASET_ID="pepper_only" #make sure this aligns with yaml config, right now its easiest to hardcode it here and in yaml file
PATH_TO_COMPRESSED_DATA="/hkfs/work/workspace/scratch/usmrd-MemVLA/datasets/lerobot/compressed_datasets/${DATASET_ID}.tar.gz"

# Check if $TMPDIR is available (cluster-specific check)
if [ -z "$TMPDIR" ]; then
    echo "ERROR: \$TMPDIR is not defined. Are you running on the cluster?"
    exit 1
fi

# Extract the data directly to $TMPDIR
tar -xf "$PATH_TO_COMPRESSED_DATA" -C "$TMPDIR"
echo "Succesfully extracted $PATH_TO_COMPRESSED_DATA to $TMPDIR/$DATASET_ID, setting this as datapath"

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
        dataset_path=$TMPDIR/$DATASET_ID \
        train.batch_size=$BATCH_PER_GPU
else
    # Single-GPU (or CPU) training using standard python call
    python src/train_flower.py \
        dataset_path=$TMPDIR/$DATASET_ID \
        train.batch_size=$BATCH_PER_GPU
fi


# To overwrite the arguments in the hydra yaml, e.g. for training from scratch or
# loading a specific checkpoint, we can set train.checkpoint_path = null or
# train.checkpoint_path=<path_to_checkpoint>
