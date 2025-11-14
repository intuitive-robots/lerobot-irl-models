from dataclasses import dataclass, field
from typing import Dict, List

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("flower")
class FlowerVLAConfig(SmolVLAConfig):
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
        self.vlm_path: str = "microsoft/Florence-2-large"
        self.freeze_florence: bool = True
        self.freeze_vision_tower: bool = True
        self.freeze_embeddings_only: bool = True
        self.vlm_prompt_style: str = "default"
        self.token_dropout: float = 0.1
        self.cfg_dropout: float = 0.0
        self.cfg_lambda: float = 1.0
        # Action and observation configuration
        self.action_dim: int = 8
        self.act_window_size: int = 16
        self.chunk_size: int = 16
        self.multistep: int = 16
        self.num_sampling_steps: int = 4
        self.sampling_type: str = "uniform"
        self.lowdim_obs_dim: int = 16
        self.use_proprio: bool = True

        # Image configuration
        self.use_second_view: bool = True
        self.second_view_key: str = "image_secondary"

        # DiT architecture
        self.dit_dim: int = 1024
        self.n_heads: int = 16
        self.n_layers: int = 12
        self.attn_pdrop: float = 0.1
        self.resid_pdrop: float = 0.1
        self.mlp_pdrop: float = 0.1
        # Attention configuration
        self.use_cross_attn: bool = True
        self.use_causal_attention: bool = True
        self.use_adaln_cond: bool = False
        self.action_type_adaln: bool = True
        self.use_readout_token: bool = False

        # Positional encoding
        self.use_rope: bool = True
        self.use_nope: bool = False
        self.query_seq_len: int = 100
        self.rope_theta: float = 1000.0

        # Action output configuration
        self.return_act_chunk: bool = False
        # Additional features
        self.use_action_scale: bool = False
        self.use_early_cross_fusion: bool = True

    # def __post_init__(self):
    #     """Validate configuration after initialization"""
    #     valid_sampling_types = ["uniform", "ln", "pi_zero", "loglogistic", "stratified"]
    #     if self.sampling_type not in valid_sampling_types:
    #         raise ValueError(
    #             f"Invalid sampling_type: {self.sampling_type}. "
    #             f"Must be one of {valid_sampling_types}"
    #         )

    #     # Validate prompt style
    #     valid_prompt_styles = ["default", "feature_focused", "state_oriented"]
    #     if self.vlm_prompt_style not in valid_prompt_styles:
    #         raise ValueError(
    #             f"Invalid vlm_prompt_style: {self.vlm_prompt_style}. "
    #             f"Must be one of {valid_prompt_styles}"
    #         )

    #     # Validate dimensions
    #     if self.dit_dim % self.n_heads != 0:
    #         raise ValueError(
    #             f"dit_dim ({self.dit_dim}) must be divisible by "
    #             f"n_heads ({self.n_heads})"
    #         )

    #     # Validate action window and multistep
    #     if self.act_window_size <= 0:
    #         raise ValueError(
    #             f"act_window_size must be positive, got {self.act_window_size}"
    #         )

    #     if self.multistep <= 0:
    #         raise ValueError(f"multistep must be positive, got {self.multistep}")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=2e-5,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=1000,
            num_decay_steps=400_000,
            peak_lr=2e-5,
            decay_lr=1e-5,
        )
