"""
Due to this bug we need our own pytton script and can not directly use lerobot.scripts.lerobot_train.py as a command:
https://github.com/huggingface/lerobot/issues/2590
Currently lerobot has an issue with xvla that when we try to run the model without having the
config file locally, it is not able to correctly load the XVLAConfig because vision_config and text_config is not defined
-> wait for fix from lerobot and until then use the following code, that loads everything manually and
then runs the training :(
"""


import logging
import os
import sys
from pathlib import Path
from wsgiref.handlers import CGIHandler

import hydra
import torch
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.scripts.lerobot_train import train as lerobot_train
from lerobot.utils.utils import init_logging

# from src.policies.xvla.custom_action_space import CustomJointActionSpace #keep this

os.environ["LEROBOT_VIDEO_BACKEND"] = "pyav"

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
log = logging.getLogger(__name__)


@hydra.main(
    config_path="../configs/xvla", config_name="xvla_config", version_base="1.3"
)
def train(cfg):
    dataset_cfg = DatasetConfig(
        repo_id=cfg.repo_id,
        root=cfg.dataset_path,
        video_backend="pyav",
    )
    # This command uses the config.json from: https://huggingface.co/lerobot/xvla-base/blob/main/config.json
    pretrained_config = PreTrainedConfig.from_pretrained(
        pretrained_name_or_path=cfg.policy.pretrained_path,
        cli_overrides=[f"--{k}={str(v).lower()}" for k, v in cfg.policy.items()],
    )

    # Issue: when using the pretrained config, it already contains an empty camera key -> this does not fit with their logic,
    # therefore we need to force these input_features
    pretrained_config.input_features = {
        "observation.images.image": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 256, 256)
        ),
        "observation.images.image2": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 256, 256)
        ),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }

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
        wandb=WandBConfig(**cfg.wandb),
        rename_map={
            "observation.images.right_cam": "observation.images.image",
            "observation.images.wrist_cam": "observation.images.image2",
        },
    )

    init_logging()
    lerobot_train(train_cfg)


if __name__ == "__main__":
    train()
