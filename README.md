# Implementation of IRL models in LeRobot

# Models
## Flower Model


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