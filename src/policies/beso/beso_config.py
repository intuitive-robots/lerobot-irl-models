from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


@PreTrainedConfig.register_subclass("beso")
class BesoConfig(DiffusionConfig):
    def __init__(
        self,
        # Diffusion specific parameters
        sigma_data: float = 0.5,
        sigma_max: float = 80.0,
        sigma_min: float = 1e-3,
        sampling_steps: int = 8,
        sampling_type: str = "ddim",
        sigma_sample_density_type: str = "loglogistic",
        # CLIP specific parameters
        use_clip_encoder: bool = False,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        clip_feature_dim: int = 512,
        freeze_clip: bool = True,
        # Language conditioning parameters
        use_language_conditioning: bool = False,
        language_feature_dim: int = 512,
        max_language_tokens: int = 77,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = 448
        # EDM-like scaling hyperparams
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sampling_steps = sampling_steps
        self.sampling_type = sampling_type
        self.sigma_sample_density_type = sigma_sample_density_type

        # CLIP parameters (either CLIP or ResNet, not both)
        self.use_clip_encoder = use_clip_encoder
        self.clip_model_name = clip_model_name
        self.clip_feature_dim = clip_feature_dim
        self.freeze_clip = freeze_clip

        # Language conditioning parameters
        self.use_language_conditioning = use_language_conditioning
        self.language_feature_dim = language_feature_dim
        self.max_language_tokens = max_language_tokens
