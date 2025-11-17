import os
import sys
from pathlib import Path
import logging

import hydra
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies import factory
from lerobot.policies.pi0.pi0_config import Pi0Config
from lerobot.policies.pi0.modeling_pi0 import Pi0Policy
from lerobot.utils.utils import init_logging
from lerobot.scripts.lerobot_train import train as lerobot_train

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
    # Dataset konfigurieren
    dataset_cfg = DatasetConfig(
        repo_id="my_dataset",
        root="/path/to/your/dataset",
        video_backend="pyav",
    )

    # Pi0-Config (aus LeRobot)
    pi0_cfg = Pi0Config(push_to_hub=False)

    # Trainingsparameter
    train_cfg = TrainPipelineConfig(
        policy=pi0_cfg,
        dataset=dataset_cfg,
        batch_size=32,
        steps=50000,
        save_freq=5000,
        seed=0,
        log_freq=100,
        wandb=get_wandb_config(),
    )

    # Policy instanziieren (wichtig für factory)
    _ = Pi0Policy(pi0_cfg)

    init_logging()
    lerobot_train(train_cfg)


def get_pi0_policy(typename: str, **kwargs):
    return Pi0Policy


def get_wandb_config():
    return WandBConfig(
        enable=False,
        project="pi0_lerobot",
        mode="disabled",
    )


if __name__ == "__main__":
    # Factory überschreiben, damit LeRobot Pi0 lädt
    factory.get_policy_class = get_pi0_policy
    train()
