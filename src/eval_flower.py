import logging
import random
from pathlib import Path

import hydra
import numpy as np
import torch

# from real_robot_env.real_robot_sim import RealRobot
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies import factory
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import get_safe_torch_device
from omegaconf import DictConfig, OmegaConf

from src.policies.flower.flower_config import FlowerVLAConfig
from src.policies.flower.flower_policymaker import make_flower_policy
from src.policies.flower.modeling_flower import FlowerVLAPolicy
from src.real_robot_env.utils.sanity_check import sanity_check_eval
from src.train_flower import get_flower, my_make_pre_post_processors

# Set protobuf implementation to pure Python to avoid compatibility issues
# between polymetis (needs protobuf 3.x) and tensorflow-metadata (needs protobuf 4.x)
# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


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


def instantiate_policy(dataset_stats: dict = None):
    """Instantiate policy from Hydra config."""

    config = FlowerVLAConfig()
    if dataset_stats is not None:
        config._dataset_stats = dataset_stats
    agent = FlowerVLAPolicy(config, dataset_stats=dataset_stats)

    return agent


@hydra.main(
    config_path="../configs", config_name="eval_config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    torch.cuda.empty_cache()
    dataset_path = "/hkfs/work/workspace/scratch/usmrd-MemVLA/datasets/lerobot/pepper_only_initial"  # "/hkfs/work/workspace/scratch/usmrd-MemVLA"
    pretrained_path = "/home/hk-project-p0024638/usmrd/projects/lerobot-irl-models/output/train/flower/2025-12-03/11-00-42/model_outputs/checkpoints/last/pretrained_model"  # "/home/irl-admin/model_weights/xvla_12_07/020000"
    task_instruction = "Pick up the bell pepper and place it in the bowl."
    set_seed_everywhere(cfg.seed)

    # Use the exact configuration from training
    # TODO: if we want to read eval_config.yaml and overwrite the TrainPipelineConfig values,
    # we need to parse the yaml into cli_args format and pass it to from_pretrained
    train_cfg = TrainPipelineConfig.from_pretrained(
        pretrained_name_or_path=pretrained_path
    )  # , cli_args=cli_args)
    train_cfg.policy.pretrained_path = Path(pretrained_path)

    # NOTE: root needs to be overwritten, because during training, the TMPDIR path is used.
    train_cfg.dataset.root = (
        dataset_path
        # "/home/irl-admin/data_collection/lerobot/pepper_only"
    )
    train_cfg.dataset.image_transforms.enable = False  # no image transforms -> define this in yaml and overwrite directly using the cli_args
    device = get_safe_torch_device(train_cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # TODO: ingAvoid load entire dataset only to get metadata information
    log.info(f"Loading Dataset from {train_cfg.dataset.root}")
    dataset = make_dataset(train_cfg)

    # Make Policy excactly as during training
    log.info("Loading Policy...")
    if isinstance(train_cfg.policy, FlowerVLAConfig):
        # Need to call FLOWER specific policy maker and preprocessor
        factory.make_policy = make_flower_policy  # monkey patch custom policy maker to pass pretrained non lerobot models
        factory.make_pre_post_processors = my_make_pre_post_processors
        factory.get_policy_class = get_flower

    policy = make_policy(
        cfg=train_cfg.policy,
        env_cfg=None,
        ds_meta=dataset.meta,
        rename_map=train_cfg.rename_map,
    )
    policy.eval()
    policy.to(device)

    # Make pre and postprocessors excactly as during training
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


if __name__ == "__main__":
    main()
