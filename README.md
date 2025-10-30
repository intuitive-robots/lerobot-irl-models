# Python Project Template

```bash
conda create --name <NAME>
conda activate <NAME>

conda install black isort pre-commit -c conda-forge

pre-commit install
pre-commit run
```

# BESO LeRobot

A simple BESO policy implementation for LeRobot.

## References

- **Paper**: [Goal-Conditioned Imitation Learning using Score-based Diffusion Policies](https://arxiv.org/abs/2304.02532)
- **Original Repository**: [https://github.com/intuitive-robots/beso](https://github.com/intuitive-robots/beso/tree/main?tab=readme-ov-file)

## Usage

Train a BESO policy:

```bash
python train.py --data_dir /path/to/dataset
```

## Installation

```bash
pip install -e .
```

## Current Structure works for Lerobot 3.