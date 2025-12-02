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

def instantiate_policy(dataset_stats: dict = None):
    """Instantiate policy from Hydra config."""

    config = FlowerVLAConfig()
    if dataset_stats is not None:
        config._dataset_stats = dataset_stats
    agent = FlowerVLAPolicy(config, dataset_stats=dataset_stats)

    return agent


@hydra.main(
    config_path="../configs", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    set_seed_everywhere(cfg.seed)

    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.get("group", "eval"),
        mode=cfg.wandb.get("mode", "disabled"),
        config=wandb.config,
    )

    dataset_stats = None

    default_stats_path = "/home/multimodallearning/data_collected/flower-lerobot/trickandtreat/trickandtreat_lerobot/meta/stats.json"
    if os.path.exists(default_stats_path):
        log.info(f"Loading dataset stats from default path: {default_stats_path}")
        import json

        with open(default_stats_path, "r") as f:
            stats_json = json.load(f)

        log.info(f"Raw stats keys from JSON: {list(stats_json.keys())}")

        dataset_stats = {}
        for key, value in stats_json.items():
            if isinstance(value, dict) and "mean" in value and "std" in value:
                try:
                    dataset_stats[key] = {
                        "mean": torch.tensor(value["mean"], dtype=torch.float32),
                        "std": torch.tensor(value["std"], dtype=torch.float32),
                        "min": torch.tensor(value["min"], dtype=torch.float32),
                        "max": torch.tensor(value["max"], dtype=torch.float32),
                    }
                    log.info(
                        f"  ✓ Loaded stats for '{key}' - mean shape: {dataset_stats[key]['mean'].shape}"
                    )
                except Exception as e:
                    log.warning(f"  ✗ Failed to load stats for '{key}': {e}")
            else:
                log.debug(f"  - Skipping '{key}' (no mean/std or not a dict)")

        log.info(f"Final dataset_stats keys: {list(dataset_stats.keys())}")
    else:
        log.warning(
            f"No dataset stats provided and default path not found: {default_stats_path}"
        )

    agent = instantiate_policy(dataset_stats=dataset_stats)

    if hasattr(cfg, "checkpoint_path") and cfg.checkpoint_path:
        log.info(f"Loading pretrained model from {cfg.checkpoint_path}")

        if cfg.checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(cfg.checkpoint_path, device=str(cfg.device))
        else:
            checkpoint = torch.load(
                cfg.checkpoint_path, map_location=cfg.device, weights_only=False
            )

            if isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith("agent."):
                new_key = "model." + key[6:]
            elif key.startswith("policy."):
                new_key = "model." + key[7:]
            elif not key.startswith("model."):
                new_key = "model." + key

            new_key = new_key.replace(".mlp.c_fc1.", ".mlp.fc1.")
            new_key = new_key.replace(".mlp.c_fc2.", ".mlp.fc2.")
            new_key = new_key.replace(".mlp.c_proj.", ".mlp.proj.")

            new_state_dict[new_key] = value

        log.info(f"Preprocessed {len(new_state_dict)} keys from checkpoint")

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

    # Turn on RTC
    rtc_cfg = RTCConfig(
          enabled=cfg.rtc.enabled,
          execution_horizon=cfg.rtc.execution_horizon,
          max_guidance_weight=cfg.rtc.max_guidance_weight,
          prefix_attention_schedule=RTCAttentionSchedule[cfg.rtc.prefix_attention_schedule.upper()],
      )
    
    agent.config.rtc_config = rtc_cfg

    # Init RTC processort, as by default if RTC disabled in the config
    # The processor won't be created
    agent.init_rtc_processor()

    agent = agent.to(cfg.device)
    agent.eval()


    log.info("Initializing RealRobot environment...")
    env_sim = RealRobot(device=cfg.device)

    log.info("Starting evaluation on real robot...")
    env_sim.test_agent(agent, cfg, rtc_cfg)

    log.info("Evaluation completed")
    wandb.finish()


if __name__ == "__main__":
    main()
