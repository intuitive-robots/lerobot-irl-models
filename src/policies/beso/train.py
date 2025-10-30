
import argparse
import pathlib
from copy import deepcopy
from pprint import pprint
import numpy
from lerobot.configs.default import DatasetConfig,WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies import factory 
from policies.BESO.beso_config import BesoConfig
from lerobot.scripts.train import init_logging
from lerobot.scripts.train import train as lerobot_train


def train(data_dir="data"):
    print("\nStarting training...")
    dataset_cfg = DatasetConfig(repo_id="my_dataset", root=data_dir)
    default_kwargs = {
                "vision_backbone": "resnet34",
                # "pretrained_backbone_weights": "ResNet34_Weights.IMAGENET1K_V1",
                "crop_shape": (224, 224),
                "use_separate_rgb_encoder_per_camera": True,
                "down_dims": (128, 256, 512, 512),
                "kernel_size": 3,
                "n_groups": 8,
                "num_train_timesteps": 1000,
                "diffusion_step_embed_dim": 512,
                "prediction_type": "sample",
                # "n_obs_steps": n_obs_steps,
                "horizon": 32,
                "n_action_steps": 16,
                "spatial_softmax_num_keypoints": 32,
            }
    pretrained_config = BesoConfig(push_to_hub=False,**default_kwargs)
    cfg = TrainPipelineConfig(
        policy=pretrained_config,
        dataset=dataset_cfg,
        batch_size=192,
        steps=32000,
        save_freq=4000,
        log_freq=200,
        wandb=get_wandb_config()
    )

    init_logging()
    lerobot_train(cfg)
def get_beso(typename: str, **kwargs):
    from policies.BESO.modelling_beso import BesoPolicy
    return BesoPolicy
def get_wandb_config():
    wandb_config = WandBConfig(
        enable=True,
        project="beso_lerobot",
        entity="stepanfedunyak-karlsruhe-institute-of-technology",
        mode="online",
    )
    return wandb_config
def main():
    factory.get_policy_class = get_beso
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    args = parser.parse_args()
    train(data_dir=pathlib.Path(args.data_dir))

if __name__ == "__main__":
    main()