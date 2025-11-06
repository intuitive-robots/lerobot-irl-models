import importlib
import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

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

    # Instantiate policy configuration
    policy_config = instantiate_policy_config(cfg.policy)

    # Set up dataset configuration for local dataset
    # For local datasets stored on disk, root should point to the dataset folder
    dataset_path = Path(cfg.dataset.root)
    log.info(f"Loading local dataset from: {dataset_path}")

    # Check if dataset exists and has required metadata
    meta_dir = dataset_path / "meta"
    info_file = meta_dir / "info.json"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found at: {dataset_path}\n"
            f"Please check your dataset.root in the config."
        )

    if not info_file.exists():
        raise FileNotFoundError(
            f"Dataset metadata not found at: {info_file}\n"
            f"Make sure your dataset is in LeRobot format with a meta/info.json file."
        )

    dataset_cfg = DatasetConfig(
        repo_id=cfg.dataset.repo_id,  # Dataset name (for logging/identification)
        root=str(dataset_path),  # Full path to dataset directory
    )

    # Set up W&B configuration
    wandb_cfg = WandBConfig(
        enable=cfg.wandb.enable,
        project=cfg.wandb.project.replace("${policy_name}", "flower"),
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode if cfg.wandb.enable else "disabled",
    )

    # Set up training pipeline configuration
    train_cfg = TrainPipelineConfig(
        policy=policy_config,
        dataset=dataset_cfg,
        batch_size=cfg.training.batch_size,
        steps=cfg.training.steps,
        save_freq=cfg.training.save_freq,
        log_freq=cfg.training.log_freq,
        wandb=wandb_cfg,
    )

    # Handle pretrained weights loading for fine-tuning
    if hasattr(cfg, "pretrained_policy_path") and cfg.pretrained_policy_path:
        log.info(f"Loading pretrained weights from: {cfg.pretrained_policy_path}")

        # Create policy instance and load weights
        policy = FlowerVLAPolicy(policy_config)

        # Load checkpoint
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
                # Assume checkpoint is the state dict itself
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Fix key naming: replace 'agent.' prefix with 'model.' prefix
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("agent."):
                # Replace 'agent.' with 'model.'
                new_key = "model." + key[6:]  # Remove 'agent.' and add 'model.'
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        state_dict = new_state_dict
        log.info(
            f"Mapped {len([k for k in state_dict.keys() if k.startswith('model.')])} keys from 'agent.' to 'model.' prefix"
        )

        # Load weights (non-strict to allow for fine-tuning with different architectures)
        missing_keys, unexpected_keys = policy.load_state_dict(state_dict, strict=False)

        if missing_keys:
            log.warning(f"Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            log.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")

        log.info("Pretrained weights loaded successfully!")

        # Store the loaded policy in train_cfg
        train_cfg.pretrained_policy = policy

    if hasattr(cfg, "resume_from_checkpoint") and cfg.resume_from_checkpoint:
        log.info(f"Resuming from checkpoint: {cfg.resume_from_checkpoint}")
        train_cfg.resume = True
        train_cfg.resume_path = cfg.resume_from_checkpoint

    # Initialize logging and start training
    # Import training module late (after patching) and run
    lerobot_train_module = importlib.import_module("lerobot.scripts.lerobot_train")
    lerobot_train_module.init_logging()
    lerobot_train_module.train(train_cfg)

    log.info("Training completed!")


@hydra.main(config_path="../configs", config_name="train_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main entry point for training with Hydra configuration."""
    # Set seed if specified
    if hasattr(cfg, "seed"):
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

    # Override data_dir if provided as command line argument
    if "data_dir" in cfg:
        cfg.dataset.root = cfg.data_dir

    # Start training
    train(cfg)


if __name__ == "__main__":
    main()
