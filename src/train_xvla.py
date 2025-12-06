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
from lerobot.policies import factory
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.scripts.lerobot_train import train as lerobot_train
from lerobot.utils.utils import init_logging

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
            "observation.images.left_cam": "observation.images.empty_camera_0",
            "observation.images.right_cam": "observation.image.image",
            "observation.images.wrist_cam": "observation.images.image2",
        },
    )

    init_logging()
    lerobot_train(train_cfg)


if __name__ == "__main__":
    train()


# python -m lerobot.scripts.lerobot_train \
#     --dataset.root=$TMPDIR/$DATASET_ID \
#     --dataset.repo_id=$DATASET_ID \
#     --dataset.video_backend="pyav" \
#     --output_dir=$OUTPUT_DIR \
#     --job_name=xvla_training \
#     --wandb.enable=False \
#     --wandb.project="xvla-training" \
#     --wandb.entity="usmrd" \
#     --wandb.mode="online" \
#     --policy.push_to_hub=False \
#     --policy.pretrained_path="lerobot/xvla-base" \
#     --steps=300 \
#     --policy.device=cuda \
#     --policy.freeze_vision_encoder=True \
#     --policy.freeze_language_encoder=True \
#     --policy.train_policy_transformer=True \
#     --policy.train_soft_prompts=True \
#     --policy.action_mode="joint" \
#     --policy.action_mode=auto
