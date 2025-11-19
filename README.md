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

# Convert your dataset to LeRobot format
    1. Collect some data
        a) the parent folder should be named after the language goal
        b) multitask datasets should be all in one folder eg. path/to/dataset/"language_goal1" and path/to/dataset/"language_goal2"
    2. Chnage in convert_data_to_lerobot.py in main() the dataset_path to your dataset path and your specific settings
    3. Run the script: python convert_data_to_lerobot.py

# Models
## Flower Model

## Pi0/0.5 Model

## groot Model




# 2. Dann die anderen requirements installieren
pip install -r requirements.txt

