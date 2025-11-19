import os
import sys
from pathlib import Path
import logging

import hydra
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies import factory
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.modeling_groot import GrootPolicy
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
    config_path=None,
    config_name=None,
    version_base="1.3",
)
def train(_cfg):
    dataset_cfg = DatasetConfig(
        repo_id="your_dataset_id",
        root="/hkfs/work/workspace/scratch/uhtfz-flower/trickandtreat_lerobot",
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

    policy_cfg = GrootConfig(
        pretrained_path="lerobot/groot",  
        repo_id="your_repo_id",
        compile_model=True,
        dtype="bfloat16",
        device="cuda",
        push_to_hub=False,
        tune_diffusion_model=False, 
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
        policy=policy_cfg,
        dataset=dataset_cfg,
        output_dir="./outputs/groot_training",
        job_name="groot_training",  
        batch_size=32,
        steps=3000,
        save_freq=5000,     
        log_freq=100,       
        save_checkpoint=True,
        seed=42,
        wandb=get_wandb_config()
    )

    init_logging()
    lerobot_train(train_cfg)

def get_groot_policy(typename: str, **kwargs):
    return GrootPolicy

def get_wandb_config():
    return WandBConfig(
        enable=False,
        project="groot_lerobot",
        mode="disabled",
    )

if __name__ == "__main__":
    factory.get_policy_class = get_groot_policy
    train()