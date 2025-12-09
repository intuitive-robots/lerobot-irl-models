import logging
import multiprocessing as mp
import os
import random

# Set protobuf implementation to pure Python to avoid compatibility issues
# between polymetis (needs protobuf 3.x) and tensorflow-metadata (needs protobuf 4.x)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import hydra
import numpy as np
import torch
import wandb
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.utils.random_utils import set_seed  # before: from lerobot.utilt.utils
from lerobot.utils.utils import get_safe_torch_device, init_logging
from omegaconf import DictConfig, OmegaConf

from src.real_robot_env.utils.sanity_check import sanity_check_eval

# from src.real_robot_env.real_robot_sim import RealRobot

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers))
OmegaConf.register_new_resolver("mul", lambda *numbers: np.prod(numbers))
torch.cuda.empty_cache()


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(
    config_path="../configs", config_name="eval_config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    cfg.seed = 42  # TODO: remove, check seed
    set_seed_everywhere(cfg.seed)
    pretrained_path = "/home/hk-project-p0024638/usmrd/projects/lerobot-irl-models/output/train/xvla/2025-12-07/17-56-59/model_outputs/checkpoints/020000/pretrained_model"
    task_instruction = "Pick up the bell pepper and place it in the bowl."
    # -------------------------------------------------------------------------
    # Step 1: Load Policy Config, Overwrite Selected Values and Set Device & Seed
    # -------------------------------------------------------------------------
    # TODO: Here we directly load train_config.json from the checkpoint directory using TrainPipelineConfig
    # instead of config.json via PreTrainedConfig. We should ensure that both are the same
    train_cfg = TrainPipelineConfig.from_pretrained(
        pretrained_name_or_path=pretrained_path
    )  # , cli_overrides=cli_overrides)
    train_cfg.policy.pretrained_path = Path(pretrained_path)

    # TODO: fix the hardcoding of following values (e.g. root needs to be overwritten,
    # because the training config stores the datapath to the TMPDIR)
    train_cfg.dataset.root = (
        "/hkfs/work/workspace/scratch/usmrd-MemVLA/datasets/lerobot/pepper_only"
    )
    if any(["empty_camera" in key for key in train_cfg.policy.input_features]):
        train_cfg.policy.input_features = {
            "observation.images.image": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 256, 256)
            ),
            "observation.images.image2": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 256, 256)
            ),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        }
        train_cfg.policy.num_views = 2
        train_cfg.policy.empty_camera = 1

    # TODO: Think about whether not using image transforms is desired (i think so)
    train_cfg.dataset.image_transforms.enable = False
    train_cfg.policy.action_mode = "auto"
    logging.info(pformat(asdict(train_cfg)))

    device = get_safe_torch_device(train_cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(train_cfg.seed)  # TODO: check which seed function to use

    # TODO: Do not load entire dataset only to get metadata information
    log.info(f"Loading Dataset from {train_cfg.dataset.root}")
    dataset = make_dataset(train_cfg)

    # -------------------------------------------------------------------------
    log.info("Loading Policy...")
    # if isinstance(train_cfg.policy, FlowerVLAConfig):
    #     # Need to call FLOWER specific policy maker and preprocessor
    #     # factory.make_policy = make_flower_policy  # monkey patch custom policy maker to pass pretrained non lerobot models
    #     # factory.make_pre_post_processors = my_make_pre_post_processors
    #     # factory.get_policy_class = get_flower
    #     pass
    policy = make_policy(
        cfg=train_cfg.policy,
        env_cfg=None,
        ds_meta=dataset.meta,
        rename_map=train_cfg.rename_map,
    )
    policy.eval()
    policy.to(device)

    # -------------------------------------------------------------------------
    # Step 4: Make Processors (Identical to lerobot-eval)
    # -------------------------------------------------------------------------
    # This creates the exact normalization/un-normalization pipeline used in training.
    # TODO: check if overrides are really needed, they might be already covered by loading the
    # pretrained model -> getting the policy_postprocessor.json/ policy_preprocessor.json
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if train_cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {
                    **policy.config.input_features,
                    **policy.config.output_features,
                },
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": train_cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=train_cfg.policy,
        pretrained_path=train_cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    log.info("Initializing RealRobot environment...")

    # env_sim = RealRobot(device=cfg.device)

    log.info("Starting evaluation on real robot...")
    sanity_check_eval(
        policy, preprocessor, postprocessor, dataset
    )  # use this to run sanity check on eval
    # env_sim.test_agent(policy, task_instruction, preprocessor, postprocessor)

    log.info("Evaluation completed")
    wandb.finish()


if __name__ == "__main__":
    main()
