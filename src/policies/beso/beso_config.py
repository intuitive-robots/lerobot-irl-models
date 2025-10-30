from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.configs.policies import PreTrainedConfig

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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim=448
        # EDM-like scaling hyperparams
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sampling_steps = sampling_steps
        self.sampling_type = sampling_type
        self.sigma_sample_density_type = sigma_sample_density_type