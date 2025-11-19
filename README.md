# Implementation of IRL models in LeRobot

# Models
## Flower Model
For Training Flower model: 
    1. Create conda env as described here: 
    2. Change in the config file your settings and train.sh 
    3. Run the training script: sbatch train.sh

For evaluation:
    0. Conda env see below
    1. Set in real_robot_sim the right parameters for the FrankaArm
    2. Add dataset stats to your config file
    3. Run eval_flower.py


## Pi0/0.5 Model
For Training Pi0 model: 
    1. Create conda env as described here: https://huggingface.co/docs/lerobot/en/pi0
    2. Change in the config file your settings and train.sh 
    3. Run the training script: sbatch train.sh

For evaluation:


## groot Model
For Training Pi0 model: 
    1. Create conda env as described here: https://huggingface.co/docs/lerobot/en/groot
    2. Change in the config file your settings and train.sh 
    3. Run the training script: sbatch train.sh

For evaluation:

# Instructions
Use pyav as video backend, because there are problems with torchcodec on Horeka.


Evaluation on real robot:

1. Install Novometis with the commands from real robot and name the env lerobot-irl-models
2.
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/test/cu124

pip install 'lerobot[all]'
pip install -r requirements.txt
```bash

3. Test if it works and install remaing dependencies



DO NOT USE:

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