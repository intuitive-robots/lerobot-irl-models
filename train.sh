#!/bin/bash
#SBATCH -p accelerated               # Partition
#SBATCH --gres=gpu:4                # Anzahl GPUs
#SBATCH --mem=64G                   # RAM
#SBATCH --time=6:00:00             # Laufzeit
#SBATCH -J train_flower        # Jobname
#SBATCH -o logs/%x_%j.out           # STDOUT-Log
#SBATCH -e logs/%x_%j.err           # STDERR-Log (optional)

source ~/.bashrc
conda activate lerobot-irl-models

# Starte Training
python src/train_flower.py
