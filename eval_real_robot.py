import logging
import multiprocessing as mp
import os
import random

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

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


def get_policy_name_from_config(policy_cfg: DictConfig) -> str:
    """Extract policy name from config target."""
    target = policy_cfg.get("_target_", "")
    if "beso" in target.lower():
        return "beso"
    elif "flower" in target.lower():
        return "flower"
    else:
        raise ValueError(f"Unknown policy type from target: {target}")


def instantiate_policy(policy_cfg: DictConfig, dataset_stats: dict = None):
    """Instantiate policy from Hydra config."""
    policy_name = get_policy_name_from_config(policy_cfg)
    log.info(f"Instantiating {policy_name} policy...")

    # Convert OmegaConf to dict
    config_dict = OmegaConf.to_container(policy_cfg, resolve=True)
    target = config_dict.pop("_target_")

    if "beso" in target.lower():
        from src.policies.beso.beso_config import BesoConfig
        from src.policies.beso.modelling_beso import BesoPolicy

        config = BesoConfig(**config_dict)
        agent = BesoPolicy(config, dataset_stats=dataset_stats)
    elif "flower" in target.lower():
        from src.policies.flower.flower_config import FlowerVLAConfig
        from src.policies.flower.modeling_flower import FlowerVLAPolicy

        config = FlowerVLAConfig(**config_dict)
        agent = FlowerVLAPolicy(config)
    else:
        raise ValueError(f"Unknown policy type from target: {target}")

    return agent, policy_name


@hydra.main(config_path="configs", config_name="eval_config.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_seed_everywhere(cfg.seed)

    # init wandb logger and config from hydra path
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.get("group", "eval"),
        mode=cfg.wandb.get("mode", "disabled"),
        config=wandb.config,
    )

    # Instantiate agent from config
    dataset_stats = None
    if hasattr(cfg, "dataset_stats") and cfg.dataset_stats:
        # Load dataset stats if provided
        if isinstance(cfg.dataset_stats, str):
            dataset_stats = torch.load(cfg.dataset_stats)
        else:
            dataset_stats = OmegaConf.to_container(cfg.dataset_stats, resolve=True)

    agent, policy_name = instantiate_policy(cfg.policy, dataset_stats=dataset_stats)
    log.info(f"Successfully instantiated {policy_name} agent")

    # Load pretrained model if checkpoint path provided
    if hasattr(cfg, "checkpoint_path") and cfg.checkpoint_path:
        log.info(f"Loading pretrained model from {cfg.checkpoint_path}")
        checkpoint = torch.load(cfg.checkpoint_path, map_location=cfg.device)
        agent.load_state_dict(checkpoint)
        log.info("Model loaded successfully")

    # Move agent to device
    agent = agent.to(cfg.device)
    agent.eval()

    # Initialize environment and start evaluation
    # Import RealRobot here to avoid early import of polymetis/torchcontrol
    log.info("Initializing RealRobot environment...")
    from real_robot_sim import RealRobot

    env_sim = RealRobot(device=cfg.device)

    log.info("Starting evaluation on real robot...")
    env_sim.test_agent(agent)

    log.info("Evaluation completed")
    wandb.finish()


if __name__ == "__main__":
    main()
