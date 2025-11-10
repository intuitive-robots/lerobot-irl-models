import importlib
import logging
import os
import random
import sys
from pathlib import Path
from random import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# Force PyAV as video backend to avoid torchcodec FFmpeg issues
os.environ["LEROBOT_VIDEO_BACKEND"] = "pyav"

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies import factory

from src.policies.flower.flower_config import FlowerVLAConfig
from src.policies.flower.modeling_flower import FlowerVLAPolicy
from src.policies.flower.processor_flower import make_flower_pre_post_processors

log = logging.getLogger(__name__)


def get_flower_factory():
    """Get the policy factory function for Flower."""

    def get_flower(typename: str, **kwargs):
        return FlowerVLAPolicy

    return get_flower


def instantiate_policy_config(policy_cfg: DictConfig) -> FlowerVLAConfig:
    """Instantiate Flower policy configuration from Hydra config."""
    config_dict = OmegaConf.to_container(policy_cfg, resolve=True)
    # Remove _target_ if present
    config_dict.pop("_target_", None)
    return FlowerVLAConfig(**config_dict)


def train(cfg: DictConfig) -> None:
    log.info("Starting training for Flower...")
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Patch LeRobot to force PyAV backend
    try:
        from lerobot.datasets import video_utils

        original_decode = video_utils.decode_video_frames

        def patched_decode_video_frames(
            video_path, timestamps, tolerance_s, backend=None
        ):
            # Force PyAV backend regardless of what was requested
            return video_utils.decode_video_frames_pyav(
                video_path, timestamps, tolerance_s
            )

        video_utils.decode_video_frames = patched_decode_video_frames
        log.info("Successfully patched video backend to use PyAV")
    except Exception as e:
        log.warning(f"Could not patch video backend: {e}")

    # Set up the policy factory for Flower
    factory.get_policy_class = get_flower_factory()

    # Runtime registration of pre/post processors for Flower
    from lerobot.policies import factory as policy_factory

    # Keep original to fallback for other policy types
    original_make_pre_post = policy_factory.make_pre_post_processors

    def make_pre_post_processors_with_flower(*args, **kwargs):
        """Wrapper matching LeRobot's factory signature."""
        policy_cfg = kwargs.get("policy_cfg") or (args[0] if len(args) > 0 else None)
        dataset_stats = kwargs.get("dataset_stats") or (
            args[1] if len(args) > 1 else None
        )

        if getattr(policy_cfg, "type", None) == "flower":
            forwarded = dict(kwargs)
            forwarded.pop("policy_cfg", None)
            forwarded.pop("dataset_stats", None)
            return make_flower_pre_post_processors(
                policy_cfg, dataset_stats, **forwarded
            )
        return original_make_pre_post(*args, **kwargs)

    policy_factory.make_pre_post_processors = make_pre_post_processors_with_flower

    # Also override the symbol inside the training script module
    lerobot_train_module = importlib.import_module("lerobot.scripts.lerobot_train")
    setattr(
        lerobot_train_module,
        "make_pre_post_processors",
        make_pre_post_processors_with_flower,
    )

    policy_config = instantiate_policy_config(cfg.policy)
    dataset_path = Path(cfg.dataset.root)

    log.info(f"Loading local dataset from: {dataset_path}")

    meta_dir = dataset_path / "meta"
    info_file = meta_dir / "info.json"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found at: {dataset_path}\n")

    if not info_file.exists():
        raise FileNotFoundError(f"Dataset metadata not found at: {info_file}\n")

    dataset_cfg = DatasetConfig(
        repo_id=cfg.dataset.repo_id,
        root=str(dataset_path),
        video_backend="pyav",
    )

    wandb_cfg = WandBConfig(
        enable=cfg.wandb.enable,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
    )

    train_cfg = TrainPipelineConfig(
        policy=policy_config,
        dataset=dataset_cfg,
        batch_size=cfg.training.batch_size,
        steps=cfg.training.steps,
        save_freq=cfg.training.save_freq,
        log_freq=cfg.training.log_freq,
        wandb=wandb_cfg,
    )

    if hasattr(cfg, "pretrained_policy_path") and cfg.pretrained_policy_path:
        log.info(f"Loading pretrained weights from: {cfg.pretrained_policy_path}")

        policy = FlowerVLAPolicy(policy_config)
        checkpoint = torch.load(cfg.pretrained_policy_path, map_location="cpu")

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
            state_dict = checkpoint

        # Fix key naming: replace 'agent.' prefix with 'model.' prefix
        # and map MLP layer names (c_fc1 -> fc1, c_fc2 -> fc2, c_proj -> proj)
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

        log.info("Pretrained weights loaded successfully!")
        train_cfg.pretrained_policy = policy

    lerobot_train_module = importlib.import_module("lerobot.scripts.lerobot_train")
    lerobot_train_module.init_logging()
    lerobot_train_module.train(train_cfg)

    log.info("Training completed!")


@hydra.main(
    config_path="../configs", config_name="train_flower_config", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    """Main entry point for training with Hydra configuration."""
    # Set seed if specified
    if hasattr(cfg, "seed"):
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

    # Override data_dir if provided as command line argument
    if "data_dir" in cfg:
        cfg.dataset.root = cfg.data_dir

    # Start training
    train(cfg)


if __name__ == "__main__":
    main()
