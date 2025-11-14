import logging
import os
import sys
from pathlib import Path

import hydra
import torch
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies import factory
from lerobot.scripts.lerobot_train import train as lerobot_train
from lerobot.utils.utils import init_logging

from policies.flower.flower_config import FlowerVLAConfig
from policies.flower.modeling_flower import FlowerVLAPolicy

os.environ["LEROBOT_VIDEO_BACKEND"] = "pyav"

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
log = logging.getLogger(__name__)


@hydra.main(
    config_path="../configs", config_name="train_flower_config", version_base="1.3"
)
def train(cfg):
    dataset_cfg = DatasetConfig(
        repo_id="my_dataset",
        root="/hkfs/work/workspace/scratch/uhtfz-flower/trickandtreat_lerobot",
        video_backend="pyav",
    )
    pretrained_config = FlowerVLAConfig(push_to_hub=False)
    train_cfg = TrainPipelineConfig(
        policy=pretrained_config,
        dataset=dataset_cfg,
        batch_size=64,
        steps=20000,
        save_freq=4000,
        seed=42,
        log_freq=1,
        wandb=get_wandb_config(),
    )

    policy = FlowerVLAPolicy(pretrained_config)
    checkpoint = torch.load(
        "/hkfs/work/workspace/scratch/uhtfz-flower/360000_model_weight.pt",
        map_location="cpu",
    )

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("agent."):
            new_key = "model." + new_key[6:]
        new_key = new_key.replace(".mlp.c_fc1.", ".mlp.fc1.")
        new_key = new_key.replace(".mlp.c_fc2.", ".mlp.fc2.")
        new_key = new_key.replace(".mlp.c_proj.", ".mlp.proj.")

        new_state_dict[new_key] = value

    state_dict = new_state_dict
    missing_keys, unexpected_keys = policy.load_state_dict(state_dict, strict=False)

    if missing_keys:
        log.warning(f"Missing keys when loading checkpoint: {missing_keys}")
    if unexpected_keys:
        log.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")

    log.info("Pretrained weights loaded successfully!")
    train_cfg.pretrained_policy = policy

    init_logging()
    lerobot_train(train_cfg)


def get_flower(typename: str, **kwargs):
    return FlowerVLAPolicy


def get_wandb_config():
    wandb_config = WandBConfig(
        enable=False,
        project="flower_lerobot",
        mode="disabled",
    )
    return wandb_config


if __name__ == "__main__":
    factory.get_policy_class = get_flower
    train()
