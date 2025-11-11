import logging
import multiprocessing as mp
import os
import random

# Set protobuf implementation to pure Python to avoid compatibility issues
# between polymetis (needs protobuf 3.x) and tensorflow-metadata (needs protobuf 4.x)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from policies.flower.flower_config import FlowerVLAConfig
from policies.flower.modeling_flower import FlowerVLAPolicy
from real_robot_env.real_robot_sim import RealRobot

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

    config = FlowerVLAConfig(**config_dict)
    agent = FlowerVLAPolicy(config)

    return agent, policy_name


@hydra.main(
    config_path="../configs", config_name="eval_flower_config.yaml", version_base="1.3"
)
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

        # Check if it's a safetensors file
        if cfg.checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(cfg.checkpoint_path, device=str(cfg.device))
        else:
            # Load pickle format (.pt, .pth)
            checkpoint = torch.load(
                cfg.checkpoint_path, map_location=cfg.device, weights_only=False
            )

            # If checkpoint is a dict with 'model' or 'state_dict' key, extract it
            if isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

        # Fix key naming: remove 'agent.' prefix if present and replace with 'model.'
        # This handles checkpoints saved with different wrapper prefixes
        new_state_dict = {}
        for key, value in state_dict.items():
            # Remove common prefixes that might differ between training and inference
            new_key = key
            if key.startswith("agent."):
                new_key = "model." + key[6:]  # Remove 'agent.' and add 'model.'
            elif key.startswith("policy."):
                new_key = "model." + key[7:]  # Remove 'policy.' and add 'model.'
            elif not key.startswith("model."):
                # If no prefix, add 'model.'
                new_key = "model." + key

            # Map MLP layer names
            new_key = new_key.replace(".mlp.c_fc1.", ".mlp.fc1.")
            new_key = new_key.replace(".mlp.c_fc2.", ".mlp.fc2.")
            new_key = new_key.replace(".mlp.c_proj.", ".mlp.proj.")

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

    # Initialize environment and start evaluation
    # Import RealRobot here to avoid early import of polymetis/torchcontrol
    log.info("Initializing RealRobot environment...")

    env_sim = RealRobot(device=cfg.device)

    log.info("Starting evaluation on real robot...")
    env_sim.test_agent(agent)

    log.info("Evaluation completed")
    wandb.finish()


if __name__ == "__main__":
    main()
