"""Evaluation script for Pi0 policy on real robot."""

import logging
import os
import random

# Set protobuf implementation to pure Python to avoid compatibility issues
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import hydra
import numpy as np
import torch
import wandb
from lerobot.policies.factory import make_policy
from omegaconf import DictConfig, OmegaConf

from real_robot_env.real_robot_sim import RealRobot

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers))
OmegaConf.register_new_resolver("mul", lambda *numbers: np.prod(numbers))
torch.cuda.empty_cache()


def set_seed_everywhere(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def instantiate_pi0_policy(cfg: DictConfig):
    """Instantiate Pi0 policy from config."""
    log.info("Instantiating Pi0 policy...")

    # Prepare policy config
    policy_cfg = OmegaConf.to_container(cfg.policy, resolve=True)

    # Ensure type is set to pi0
    policy_cfg["type"] = "pi0"

    # Convert back to OmegaConf
    policy_cfg = OmegaConf.create(policy_cfg)

    # Create policy using LeRobot's factory
    # Note: make_policy expects a policy_cfg and optionally dataset_stats
    dataset_stats = None
    if hasattr(cfg, "dataset_stats") and cfg.dataset_stats:
        if isinstance(cfg.dataset_stats, str):
            dataset_stats = torch.load(cfg.dataset_stats)
        else:
            dataset_stats = OmegaConf.to_container(cfg.dataset_stats, resolve=True)

    policy = make_policy(policy_cfg, dataset_stats=dataset_stats)

    return policy


@hydra.main(
    config_path="../configs", config_name="eval_pi0_config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    set_seed_everywhere(cfg.seed)

    # Initialize wandb
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.get("group", "eval"),
        mode=cfg.wandb.get("mode", "disabled"),
        config=wandb.config,
    )

    # Instantiate Pi0 policy
    agent = instantiate_pi0_policy(cfg)
    log.info("Successfully instantiated Pi0 policy")

    # Load checkpoint if provided
    if hasattr(cfg, "checkpoint_path") and cfg.checkpoint_path:
        log.info(f"Loading pretrained model from {cfg.checkpoint_path}")
        checkpoint = torch.load(cfg.checkpoint_path, map_location=cfg.device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Preprocess keys if needed
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            # Handle different prefix conventions
            if key.startswith("agent."):
                new_key = "model." + key[6:]
            elif key.startswith("policy."):
                new_key = "model." + key[7:]
            elif not key.startswith("model.") and not key.startswith("_"):
                # If no standard prefix, try adding model.
                new_key = "model." + key

            new_state_dict[new_key] = value

        log.info(f"Preprocessed {len(new_state_dict)} keys from checkpoint")

        # Load with strict=False to allow partial loading
        missing_keys, unexpected_keys = agent.load_state_dict(
            new_state_dict, strict=False
        )

        if missing_keys:
            log.warning(f"Missing keys in checkpoint ({len(missing_keys)} total):")
            log.warning(f"  First few: {missing_keys[:5]}")
            log.warning("  → These parameters will use random initialization!")

        if unexpected_keys:
            log.warning(
                f"Unexpected keys in checkpoint ({len(unexpected_keys)} total):"
            )
            log.warning(f"  First few: {unexpected_keys[:5]}")
            log.warning("  → These parameters from checkpoint will be ignored!")

        if not missing_keys and not unexpected_keys:
            log.info("✅ All parameters loaded successfully!")
        else:
            log.info("⚠️  Model loaded with warnings (see above)")

    # Move agent to device
    agent = agent.to(cfg.device)
    agent.eval()

    # Initialize environment
    log.info("Initializing RealRobot environment...")
    env_sim = RealRobot(device=cfg.device)

    log.info("Starting evaluation on real robot...")
    env_sim.test_agent(agent)

    log.info("Evaluation completed")
    wandb.finish()


if __name__ == "__main__":
    main()
