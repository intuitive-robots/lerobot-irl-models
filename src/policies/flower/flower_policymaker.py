#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
import os
from typing import Any, TypedDict

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.envs.configs import EnvConfig
from lerobot.envs.utils import env_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.sac.reward_model.configuration_classifier import (
    RewardClassifierConfig,
)
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig
from lerobot.policies.utils import validate_visual_features_consistency
from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from typing_extensions import Unpack

from src.policies.flower.modeling_flower import FlowerVLAPolicy


def make_flower_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
    env_cfg: EnvConfig | None = None,
    rename_map: dict[str, str] | None = None,
) -> PreTrainedPolicy:
    """
    Instantiate a policy model.

    This factory function handles the logic of creating a policy, which requires
    determining the input and output feature shapes. These shapes can be derived
    either from a `LeRobotDatasetMetadata` object or an `EnvConfig` object. The function
    can either initialize a new policy from scratch or load a pretrained one.

    Args:
        cfg: The configuration for the policy to be created. If `cfg.pretrained_path` is
             set, the policy will be loaded with weights from that path.
        ds_meta: Dataset metadata used to infer feature shapes and types. Also provides
                 statistics for normalization layers.
        env_cfg: Environment configuration used to infer feature shapes and types.
                 One of `ds_meta` or `env_cfg` must be provided.
        rename_map: Optional mapping of dataset or environment feature keys to match
                 expected policy feature names (e.g., `"left"` → `"camera1"`).

    Returns:
        An instantiated and device-placed policy model.

    Raises:
        ValueError: If both or neither of `ds_meta` and `env_cfg` are provided.
        NotImplementedError: If attempting to use an unsupported policy-backend
                             combination (e.g., VQBeT with 'mps').
    """
    if bool(ds_meta) == bool(env_cfg):
        raise ValueError(
            "Either one of a dataset metadata or a sim env must be provided."
        )

    # NOTE: Currently, if you try to run vqbet with mps backend, you'll get this error.
    # TODO(aliberts, rcadene): Implement a check_backend_compatibility in policies?
    # NotImplementedError: The operator 'aten::unique_dim' is not currently implemented for the MPS device. If
    # you want this op to be added in priority during the prototype phase of this feature, please comment on
    # https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment
    # variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be
    # slower than running natively on MPS.
    if cfg.type == "vqbet" and cfg.device == "mps":
        raise NotImplementedError(
            "Current implementation of VQBeT does not support `mps` backend. "
            "Please use `cpu` or `cuda` backend."
        )

    policy_cls = FlowerVLAPolicy

    kwargs = {}
    if ds_meta is not None:
        features = dataset_to_policy_features(ds_meta.features)
    else:
        if not cfg.pretrained_path:
            logging.warning(
                "You are instantiating a policy from scratch and its features are parsed from an environment "
                "rather than a dataset. Normalization modules inside the policy will have infinite values "
                "by default without stats from a dataset."
            )
        if env_cfg is None:
            raise ValueError("env_cfg cannot be None when ds_meta is not provided")
        features = env_to_policy_features(env_cfg)

    if not cfg.output_features:
        cfg.output_features = {
            key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
        }
    if not cfg.input_features:
        cfg.input_features = {
            key: ft for key, ft in features.items() if key not in cfg.output_features
        }
    kwargs["config"] = cfg

    # Keep everything until here
    # TODO: Improve the code down below and make sure to leverage lerobots prebuilt options such as TrainConfig
    # where possible -> for now keep it as is and push ahead
    # Train from scratch
    if not cfg.pretrained_path:
        policy = FlowerVLAPolicy(cfg)  # policy_cls(**kwargs)
        logging.info(f"No pretrained weights provided, training from scratch")
    else:
        # TODO: Improve the handling of policy intialization from lerobot or non-lerobot checkpoints (especially decision logic)
        if os.path.isdir(cfg.pretrained_path):
            pass
            policy = FlowerVLAPolicy.from_pretrained(
                pretrained_name_or_path=cfg.pretrained_path, config=cfg
            )  # Load lerobot checkpoint
            # not sure if i need to set this: cfg.device = "cpu" #Currently there seems no better way of enforcing this
            # policy = FlowerVLAPolicy.from_pretrained(pretrained_name_or_path=cfg.checkpoint_path, config = cfg)

        else:
            # Load non lerobot checkpoint
            policy = FlowerVLAPolicy(cfg)
            checkpoint = torch.load(
                cfg.pretrained_path,
                map_location="cpu",
            )

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
                state_dict = checkpoint  # NOTE: for flower we use this line

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
            missing_keys, unexpected_keys = policy.load_state_dict(
                state_dict, strict=False
            )

            if missing_keys:
                logging.warning(f"Missing keys when loading checkpoint: {missing_keys}")
            if unexpected_keys:
                logging.warning(
                    f"Unexpected keys when loading checkpoint: {unexpected_keys}"
                )

            logging.info(
                f"Sucessfully loaded pretrained weights from {cfg.pretrained_path}"
            )

    # if cfg.pretrained_path:
    #     # Load a pretrained policy and override the config if needed (for example, if there are inference-time
    #     # hyperparameters that we want to vary).
    #     kwargs["pretrained_name_or_path"] = cfg.pretrained_path
    #     policy = policy_cls.from_pretrained(**kwargs)
    # else:
    #     # Make a fresh policy.
    #     policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, torch.nn.Module)

    # policy = torch.compile(policy, mode="reduce-overhead")

    if not rename_map:
        validate_visual_features_consistency(cfg, features)
        # TODO: (jadechoghari) - add a check_state(cfg, features) and check_action(cfg, features)

    return policy
