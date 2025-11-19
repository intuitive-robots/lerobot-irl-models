import os
import sys
from pathlib import Path
import logging

import hydra
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies import factory
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.utils import init_logging
from lerobot.scripts.lerobot_train import train as lerobot_train
from lerobot.policies.utils import PolicyFeature
from lerobot.policies.utils import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
os.environ["LEROBOT_VIDEO_BACKEND"] = "pyav"

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
log = logging.getLogger(__name__)


@hydra.main(
    config_path="../configs", config_name="config", version_base="1.3"
)
def train(cfg):
    dataset_cfg = DatasetConfig(
        repo_id=cfg.repo_id,
        root=cfg.dataset_path,
        video_backend="pyav",
    )

    dataset = LeRobotDataset(
        dataset_cfg.repo_id,
        root=dataset_cfg.root
    )
    
    sample = dataset[0]
    
    img_shape = sample["observation.images.right_cam"].shape
    state_shape = sample["observation.state"].shape
    action_shape = sample["action"].shape
    
    print(f"Image shape: {img_shape}")
    print(f"State shape: {state_shape}")
    print(f"Action shape: {action_shape}")

    pi05_cfg = PI05Config(
        pretrained_path="lerobot/pi05_base",
        repo_id=cfg.repo_id,
        compile_model=cfg.train.compile_model,
        dtype=cfg.train.dtype,
        device=cfg.train.device,
        push_to_hub=cfg.train.push_to_hub,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        input_features={
            "observation.images.right_cam": PolicyFeature(FeatureType.VISUAL, img_shape),
            "observation.images.wrist_cam": PolicyFeature(FeatureType.VISUAL, img_shape),
            "observation.state": PolicyFeature(FeatureType.STATE, state_shape),
        },
        output_features={
            "action": PolicyFeature(FeatureType.ACTION, action_shape),
        },
    )

    train_cfg = TrainPipelineConfig(
        policy=pi05_cfg,
        dataset=dataset_cfg,
        output_dir=cfg.train.output_dir,
        job_name=cfg.train.job_name,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        steps=cfg.train.steps,
        save_freq=cfg.train.save_freq,
        seed=cfg.train.seed,
        log_freq=cfg.train.log_freq,
        wandb=WandBConfig(
        enable=cfg.wandb.enable,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        ),
    )

    init_logging()
    lerobot_train(train_cfg)

def get_pi05_policy(typename: str, **kwargs):
    return PI05Policy

if __name__ == "__main__":
    factory.get_policy_class = get_pi05_policy
    train()
