# Implementation of IRL models in LeRobot


```bash
conda create --name lerobot-irl-models python=3.10
conda activate lerobot-irl-models
pip install -r requirements.txt
```
conda install -c conda-forge ffmpeg

# Models
## BESO LeRobot

A simple BESO policy implementation for LeRobot.

## References

- **Paper**: [Goal-Conditioned Imitation Learning using Score-based Diffusion Policies](https://arxiv.org/abs/2304.02532)
- **Original Repository**: [https://github.com/intuitive-robots/beso](https://github.com/intuitive-robots/beso/tree/main?tab=readme-ov-file)

#TODO
- [] Add PI_0 Model
- [] Clip Encoder for BESO
- [] Proper README
- [] Add Trainer and Agent System for only one train script
- [] Check Flower Processor for unnecessary parts and wrong implementations
- [] Set for everything random seed