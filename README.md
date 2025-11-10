# Implementation of IRL models in LeRobot

Evaluation on real robot:

1. Install Novometis with the commands from real robot and name the env lerobot-irl-models
2.
```bash
conda install black isort pre-commit -c conda-forge
pre-commit install
pre-commit run
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/test/cu124
pip install -r requirements.txt
pip insall lerobot
```

No evalutation on real robot:

```bash
conda create --name lerobot-irl-models python=3.10
conda activate lerobot-irl-models
conda install black isort pre-commit -c conda-forge
pre-commit install
pre-commit run
pip3 install torch --index-url https://download.pytorch.org/whl/test/cu124
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/test/cu124
pip install -r requirements.txt
pip insall lerobot
```
conda install -c conda-forge ffmpeg

# Models
## BESO LeRobot

A simple BESO policy implementation for LeRobot.

## References

- **Paper**: [Goal-Conditioned Imitation Learning using Score-based Diffusion Policies](https://arxiv.org/abs/2304.02532)
- **Original Repository**: [https://github.com/intuitive-robots/beso](https://github.com/intuitive-robots/beso/tree/main?tab=readme-ov-file)



# 2. Dann die anderen requirements installieren
pip install -r requirements.txt

#TODO
- [] Add PI_0 Model
- [] Clip Encoder for BESO
- [] Proper README
- [] Add Trainer and Agent System for only one train script
- [] Check Flower Processor for unnecessary parts and wrong implementations
