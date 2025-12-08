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

from src.policies.flower.flower_config import FlowerVLAConfig
from src.policies.flower.flower_policymaker import make_flower_policy
from src.policies.flower.processor_flower import make_flower_pre_post_processors


def my_make_pre_post_processors(policy_cfg, pretrained_path=None, **kwargs):
    print(">>> Using custom make_pre_post_processors <<<")
    processors = make_flower_pre_post_processors(
        config=policy_cfg,
        dataset_stats=kwargs.get("dataset_stats"),
    )
    return processors


# _original_make_pre_post_processors = factory.make_pre_post_processors
factory.make_pre_post_processors = my_make_pre_post_processors
factory.make_policy = make_flower_policy  # monkey patch custom policy maker to pass pretrained non lerobot models

from lerobot.scripts.lerobot_train import train as lerobot_train
from lerobot.utils.utils import init_logging

from src.policies.flower.modeling_flower import FlowerVLAPolicy

os.environ["LEROBOT_VIDEO_BACKEND"] = "pyav"

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
log = logging.getLogger(__name__)


@hydra.main(
    config_path="../configs/flower", config_name="flower_config", version_base="1.3"
)
def train(cfg):
    dataset_cfg = DatasetConfig(
        repo_id=cfg.repo_id,
        root=cfg.dataset_path,
        video_backend="pyav",
    )

    # TODO: make sure that correct device types are set to avoid OOM errors when instantiating the policy
    # and passing it to the accelerator
    if cfg.train.resume:
        # When resuming: Need to check whether the optimizer and scheduler are loaded correctly, maybe we need TrainPipelineConfig for that
        pretrained_config = FlowerVLAConfig.from_pretrained(
            pretrained_name_or_path=cfg.checkpoint_path
        )
        pretrained_config.device = (
            "cpu"  # Currently there seems no better way of enforcing this
        )

    else:
        pretrained_config = hydra.utils.instantiate(cfg.model, _convert_="all")
    pretrained_config.device = "cuda"
    pretrained_config.pretrained_path = cfg.checkpoint_path

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

    init_logging()
    lerobot_train(train_cfg)


def get_flower(typename: str, **kwargs):
    return FlowerVLAPolicy


if __name__ == "__main__":
    factory.get_policy_class = get_flower
    train()
