from dataclasses import dataclass, field
from typing import Dict, List

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("beast_vla")
class BeastVLAConfig(SmolVLAConfig):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.obs_modalities = "observation"
        self.goal_modalities = "task"
        self.target_modality = "action"
        self.lang_modalities = ["language_instruction"]
        self.img_modalities = ["image_primary"]
        
        # Define input and output features for normalization
        self.input_features = {
            "observation.images.right_cam": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 256, 256)
            ),
            "observation.images.wrist_cam": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 256, 256)
            ),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        }
        self.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
        }
        
        self.normalization_mapping = {
            FeatureType.STATE: NormalizationMode.MEAN_STD,
            FeatureType.ACTION: NormalizationMode.MEAN_STD,
        }
        
        # VLM configuration
        self.vlm_path: str = "microsoft/Florence-2-base"
        self.freeze_florence: bool = False
        self.freeze_vision_tower: bool = False
        self.freeze_embeddings_only: bool = False
        self.vlm_prompt_style: str = "default"
        self.token_dropout: float = 0.1
        self.cfg_dropout: float = 0.0
        self.cfg_lambda: float = 1.0
        # Action and observation configuration
        self.action_dim: int = 8
        self.act_window_size: int = 16
        self.chunk_size: int = 16
        self.multistep: int = 16
        self.lowdim_obs_dim: int = 16
        self.use_proprio: bool = False
        # Image configuration
        self.use_second_view: bool = True
        self.second_view_key: str = "image_secondary"
        # Beast Tokenizer configuration
        self.num_dof: int = 8
        # B-spline parameters
        self.gripper_zero_order: bool = False
        self.num_basis: int = 5
        self.degree_p: int = 4
        self.action_bins: int = 256
        self.update_w_bound: bool = True
        # Action output configuration
        self.return_act_chunk: bool = False
        # Additional features
        self.use_action_scale: bool = False
        self.use_early_cross_fusion: bool = True


    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=2e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=1000,
            num_decay_steps=400_000,
            peak_lr=2e-5,
            decay_lr=1e-5,
        )
