from dataclasses import dataclass, field
from typing import Dict, List

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.configs.types import NormalizationMode, PolicyFeature

@PreTrainedConfig.register_subclass("flower")
@dataclass
class FlowerVLAConfig(PreTrainedConfig):
    obs_modalities = "observation"
    goal_modalities = "task"
    target_modality = "action"
    lang_modalities = ["language_instruction"]
    img_modalities = ["image_primary"]
    # VLM configuration
    vlm_path: str = "microsoft/Florence-2-large"
    freeze_florence: bool = True
    freeze_vision_tower: bool = True
    freeze_embeddings_only: bool = True
    vlm_prompt_style: str = "default"
    token_dropout: float = 0.1
    cfg_dropout: float = 0.0
    cfg_lambda: float = 1.0

    # Action and observation configuration
    action_dim: int = 8
    act_window_size: int = 16
    chunk_size: int = 16
    multistep: int = 16
    num_sampling_steps: int = 4
    sampling_type: str = "uniform"
    lowdim_obs_dim: int = 16
    use_proprio: bool = True

    # Image configuration
    use_second_view: bool = True
    second_view_key: str = "image_secondary"

    # DiT architecture
    dit_dim: int = 1024
    n_heads: int = 16
    n_layers: int = 12
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    mlp_pdrop: float = 0.1

    # Attention configuration
    use_cross_attn: bool = True
    use_causal_attention: bool = True
    use_adaln_cond: bool = False
    action_type_adaln: bool = True
    use_readout_token: bool = False

    # Positional encoding
    use_rope: bool = True
    use_nope: bool = False
    query_seq_len: int = 100
    rope_theta: float = 1000.0

    # Action output configuration
    return_act_chunk: bool = False

    # LeRobot compatibility
    input_shapes: Dict[str, List[int]] = field(default_factory=dict)
    output_shapes: Dict[str, List[int]] = field(default_factory=dict)
    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
    output_normalization_modes: Dict[str, str] = field(default_factory=dict)

    # Additional features
    use_action_scale: bool = False
    use_early_cross_fusion: bool = True

    def __post_init__(self):
        """Validate configuration after initialization"""
        valid_sampling_types = ["uniform", "ln", "pi_zero", "loglogistic", "stratified"]
        if self.sampling_type not in valid_sampling_types:
            raise ValueError(
                f"Invalid sampling_type: {self.sampling_type}. "
                f"Must be one of {valid_sampling_types}"
            )

        # Validate prompt style
        valid_prompt_styles = ["default", "feature_focused", "state_oriented"]
        if self.vlm_prompt_style not in valid_prompt_styles:
            raise ValueError(
                f"Invalid vlm_prompt_style: {self.vlm_prompt_style}. "
                f"Must be one of {valid_prompt_styles}"
            )

        # Validate dimensions
        if self.dit_dim % self.n_heads != 0:
            raise ValueError(
                f"dit_dim ({self.dit_dim}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )

        # Validate action window and multistep
        if self.act_window_size <= 0:
            raise ValueError(
                f"act_window_size must be positive, got {self.act_window_size}"
            )

        if self.multistep <= 0:
            raise ValueError(f"multistep must be positive, got {self.multistep}")

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        # Use act_window_size if chunk_size not set
        chunk = self.chunk_size if hasattr(self, "chunk_size") else self.act_window_size
        return list(range(chunk))

    @property
    def reward_delta_indices(self) -> None:
        return None

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

    def validate_features(self) -> None:
        if not self.input_shapes:
            return

        # Check for at least one image input
        has_image = any(
            key.startswith("observation.image") for key in self.input_shapes.keys()
        )
        if not has_image:
            raise ValueError(
                "At least one image observation is required. "
                "Expected keys like 'observation.image' in input_shapes"
            )

        # Validate image shapes (should be [C, H, W] or [H, W, C])
        for key, shape in self.input_shapes.items():
            if "image" in key and len(shape) != 3:
                raise ValueError(
                    f"Invalid image shape for {key}: {shape}. "
                    f"Expected 3D tensor (C, H, W) or (H, W, C)"
                )

        # Check proprioceptive observations if enabled
        if self.use_proprio:
            has_proprio = "observation.state" in self.input_shapes
            if not has_proprio:
                raise ValueError(
                    "use_proprio is True but 'observation.state' not found in input_shapes"
                )

            # Validate proprio dimension
            if has_proprio:
                proprio_shape = self.input_shapes["observation.state"]
                if len(proprio_shape) != 1:
                    raise ValueError(
                        f"Invalid state shape: {proprio_shape}. Expected 1D vector"
                    )
                actual_dim = proprio_shape[0]
                if actual_dim != self.lowdim_obs_dim:
                    raise ValueError(
                        f"State dimension mismatch: got {actual_dim}, "
                        f"expected {self.lowdim_obs_dim}"
                    )

        # Validate action output
        if self.output_shapes:
            if "action" not in self.output_shapes:
                raise ValueError("'action' not found in output_shapes")

            action_shape = self.output_shapes["action"]
            if len(action_shape) != 1:
                raise ValueError(
                    f"Invalid action shape: {action_shape}. Expected 1D vector"
                )

            expected_action_dim = self.action_dim
            actual_action_dim = action_shape[0]
            if actual_action_dim != expected_action_dim:
                raise ValueError(
                    f"Action dimension mismatch: got {actual_action_dim}, "
                    f"expected {expected_action_dim}"
                )
