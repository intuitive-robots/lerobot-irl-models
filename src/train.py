import logging
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies import factory
from lerobot.scripts.lerobot_train import init_logging
from lerobot.scripts.lerobot_train import train as lerobot_train

log = logging.getLogger(__name__)


def get_policy_name_from_config(policy_cfg: DictConfig) -> str:
    """Extract policy name from config target."""
    target = policy_cfg.get("_target_", "")
    if "beso" in target.lower():
        return "beso"
    elif "flower" in target.lower():
        return "flower"
    else:
        raise ValueError(f"Unknown policy type from target: {target}")


def get_policy_factory(policy_name: str):
    """Get the policy factory function for the specified policy."""
    if policy_name.lower() == "beso":

        def get_beso(typename: str, **kwargs):
            from src.policies.beso.modelling_beso import BesoPolicy

            return BesoPolicy

        return get_beso
    elif policy_name.lower() == "flower":

        def get_flower(typename: str, **kwargs):
            from src.policies.flower.modeling_flower import FlowerVLAPolicy

            return FlowerVLAPolicy

        return get_flower
    else:
        raise ValueError(f"Unknown policy: {policy_name}. Choose 'beso' or 'flower'.")


def instantiate_policy_config(policy_cfg: DictConfig) -> Any:
    """Instantiate policy configuration from Hydra config."""
    # Convert OmegaConf to dict and instantiate
    config_dict = OmegaConf.to_container(policy_cfg, resolve=True)

    # Get the target class
    target = config_dict.pop("_target_")

    # Import and instantiate the config class
    if "beso" in target.lower():
        from src.policies.beso.beso_config import BesoConfig

        return BesoConfig(**config_dict)
    elif "flower" in target.lower():
        from src.policies.flower.flower_config import FlowerVLAConfig

        return FlowerVLAConfig(**config_dict)
    else:
        raise ValueError(f"Unknown config target: {target}")


def train(cfg: DictConfig) -> None:
    policy_name = get_policy_name_from_config(cfg.policy)
    log.info(f"Starting training for {policy_name}...")
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set up the policy factory
    factory.get_policy_class = get_policy_factory(policy_name)

    # Instantiate policy configuration
    policy_config = instantiate_policy_config(cfg.policy)

    # Set up dataset configuration
    dataset_cfg = DatasetConfig(repo_id=cfg.dataset.repo_id, root=cfg.dataset.root)

    # Set up W&B configuration
    wandb_cfg = WandBConfig(
        enable=cfg.wandb.enable,
        project=cfg.wandb.project.replace("${policy_name}", policy_name),
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

    # Initialize logging and start training
    init_logging()
    lerobot_train(train_cfg)

    log.info("Training completed!")


@hydra.main(config_path="../../configs", config_name="train_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main entry point for training with Hydra configuration."""
    # Set seed if specified
    if hasattr(cfg, "seed"):
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

    # Override data_dir if provided as command line argument
    # This allows: python train.py data_dir=/path/to/data
    if "data_dir" in cfg:
        cfg.dataset.root = cfg.data_dir

    OmegaConf.set_struct(cfg, False)  # Allow adding new keys
    cfg.policy_name = get_policy_name_from_config(cfg.policy)
    OmegaConf.set_struct(cfg, True)  # Re-enable struct mode

    # Start training
    train(cfg)


if __name__ == "__main__":
    main()
