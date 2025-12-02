import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from wsgiref.handlers import CGIHandler

import hydra
import torch
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies import factory

from policies.flower.processor_flower import make_flower_pre_post_processors


def my_make_pre_post_processors(policy_cfg, pretrained_path=None, **kwargs):
    print(">>> Using custom make_pre_post_processors <<<")
    processors = make_flower_pre_post_processors(
        config=policy_cfg,
        dataset_stats=kwargs.get("dataset_stats"),
    )
    return processors


# _original_make_pre_post_processors = factory.make_pre_post_processors
factory.make_pre_post_processors = my_make_pre_post_processors

from lerobot.scripts.lerobot_train import train as lerobot_train
from lerobot.utils.utils import init_logging

from policies.flower.modeling_flower import FlowerVLAPolicy

os.environ["LEROBOT_VIDEO_BACKEND"] = "pyav"

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
log = logging.getLogger(__name__)


def load_flower_policy(pretrained_config, checkpoint_path):
    """
    This method loads the FLowerVLAPolicy class and handles its initialization from passed checkpoint
    that can either be from flower_vla_pret or checkpoints of previous flower versions trained with lerobot (resume = True)
    """
    policy = FlowerVLAPolicy(pretrained_config)
    if checkpoint_path:
        checkpoint = torch.load(
            checkpoint_path,
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
            state_dict = checkpoint  # NOTE: for flower we use this line

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

        log.info(f"Sucessfully loaded pretrained weights from {checkpoint_path}!")
    else:
        log.info(f"No pretrained weights provided, training from scratch")
    return policy


@hydra.main(
    config_path="../configs/flower", config_name="flower_config", version_base="1.3"
)
def train(cfg):
    dataset_cfg = DatasetConfig(
        repo_id=cfg.repo_id,
        root=cfg.dataset_path,
        video_backend="pyav",
    )
    pretrained_config = hydra.utils.instantiate(cfg.model, _convert_="all")
    pretrained_config.device = "cpu"  # set cpu for model loading, otherwise OOM Error
    policy = load_flower_policy(pretrained_config, cfg.checkpoint_path)
    pretrained_config.device = "cuda"

    train_cfg = TrainPipelineConfig(
        policy=pretrained_config,
        dataset=dataset_cfg,
        batch_size=cfg.train.batch_size,
        steps=cfg.train.steps,
        output_dir=Path(cfg.train.output_dir),
        job_name=cfg.train.job_name,
        save_freq=cfg.train.save_freq,
        seed=cfg.train.seed,
        log_freq=cfg.train.log_freq,
        num_workers=cfg.train.num_workers,
        wandb=WandBConfig(
            enable=cfg.wandb.enable,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            mode=cfg.wandb.mode,
        ),
    )

    train_cfg.pretrained_policy = policy

    init_logging()
    lerobot_train(train_cfg)


def get_flower(typename: str, **kwargs):
    return FlowerVLAPolicy


if __name__ == "__main__":
    factory.get_policy_class = get_flower
    train()
