import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor.normalize_processor import (
    NormalizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.utils.constants import ACTION
from timm.layers.mlp import Mlp
from torchdiffeq import odeint
from transformers import AutoModelForCausalLM, AutoProcessor

from .action_index import ActionIndex
from .flower_config import FlowerVLAConfig
from .transformers import (
    ActionSpaceEmbedderParameter,
    FlowBlock,
    FreqEmbedder,
    RmsNorm,
    SharedAdaLNController,
    TimestepEmbedder,
    ZeroEncoder,
    stateless_norm,
)

logger = logging.getLogger(__name__)


class FlowerVLAPolicy(PreTrainedPolicy):
    """
    FlowerVLA Policy for LeRobot.

    Combines Florence-2 VLM with Flow-based DiT for learning generalist manipulation policies.

    Key Features:
    - Multi-view image observations
    - Language goal conditioning
    - Rectified Flow loss
    - Action chunking (predicts multiple actions)
    - Multiple action spaces (eef_delta, joint_single, bimanual_nav)
    """

    name = "flower_vla"
    config_class = FlowerVLAConfig

    def __init__(
        self,
        config: FlowerVLAConfig,
        dataset_stats: dict[str, dict[str, torch.Tensor]]
        | None = None,  # NOTE: This is not passed by lerobot at initialization
    ):
        """
        Initialize FlowerVLA Policy.

        Args:
            config: FlowerVLAConfig instance with all hyperparameters
        """
        super().__init__(config)

        # If dataset_stats not provided, try to get from config
        if dataset_stats is None and hasattr(config, "_dataset_stats"):
            dataset_stats = config._dataset_stats
            logger.info("📊 Using dataset_stats from config")

        self.config = config
        self.normalize_inputs = NormalizerProcessorStep(
            config.input_features, config.normalization_mapping, dataset_stats
        )
        self.normalize_targets = NormalizerProcessorStep(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = UnnormalizerProcessorStep(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.model = FlowerModel(config)

        self.model.reset()

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass (training mode) - returns (loss, output_dict) for LeRobot training loop.

        Args:
            batch: Dictionary with observation and action data

        Returns:
            Tuple of (loss tensor, output dictionary with metrics)
        """
        result = self.model.forward(batch)
        return result["loss"], result["loss_dict"]

    def encode_observations(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Delegate to model."""
        return self.model.encode_observations(batch)

    def rf_loss(
        self, cond: dict, actions: torch.Tensor, dataset_idx: Any = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Delegate to model."""
        return self.model.rf_loss(cond, actions, dataset_idx)

    def sample_actions(
        self, z: torch.Tensor, cond: Dict[str, torch.Tensor], inference: bool = False
    ) -> torch.Tensor:
        """Delegate to model."""
        return self.model.sample_actions(z, cond, inference)

    def compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute loss for training (LeRobot protocol).

        Args:
            batch: Training batch

        Returns:
            - Loss tensor
            - Dictionary with loss statistics
        """
        result = self.forward(batch)
        return result["loss"], result["loss_dict"]

    def get_optim_params(self) -> dict:
        return self.parameters()

    @torch.no_grad()
    def predict_action_chunk(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict a chunk of actions given environment observations.

        Args:
            batch: Dictionary with observation data

        Returns:
            Action chunk [B, act_window_size, action_dim]
        """
        # Encode observations
        cond = self.model.encode_observations(batch)

        # Sample noise
        B = cond["features"].shape[0]
        noise = torch.randn(
            B,
            self.config.act_window_size,
            self.config.action_dim,
            device=self.model.device,
        )

        # Sample actions
        action_seq = self.model.sample_actions(noise, cond, inference=True)
        action_seq = self.unnormalize_outputs({ACTION: action_seq})[ACTION]
        return action_seq  # [B, T, action_dim]

    def reset(self) -> None:
        """Reset rollout state."""
        self.model.reset()

    @torch.no_grad()
    def select_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Select action for inference (LeRobot protocol).
        This method handles action chunking internally:
        - On first call (or every multistep): predicts new action chunk
        - Returns single action per timestep from the chunk

        Args:
            batch: Observation batch

        Returns:
            Selected action [B, action_dim]
        """
        if ACTION in batch:
            batch.pop(ACTION)
        batch = self.normalize_inputs(batch)
        # Check if we need to predict a new action chunk
        if (
            self.model.rollout_step_counter % self.config.multistep == 0
            or self.model.pred_action_seq is None
        ):
            # Predict new action chunk
            self.model.pred_action_seq = self.predict_action_chunk(batch)

        # Get current action from the chunk
        if self.config.return_act_chunk:
            # Return full chunk
            action = self.model.pred_action_seq
        else:
            # Return single action at current step
            # action = self.model.pred_action_seq
            action = self.model.pred_action_seq[:, self.model.rollout_step_counter, :]

        # Update counter
        self.model.rollout_step_counter += 1
        if self.model.rollout_step_counter >= self.config.multistep:
            self.model.rollout_step_counter = 0

        return action


class FlowerModel(nn.Module):
    """
    FlowerVLA Model combining Florence-2 VLM with Flow-based DiT.
    """

    def __init__(self, config: FlowerVLAConfig):
        super().__init__()
        # Core attributes
        self.device = torch.device(
            config.device
            if hasattr(config, "device") and config.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize configuration groups.
        self._init_modalities(
            config.target_modality,
            config.obs_modalities,
            config.goal_modalities,
            config.img_modalities,
            config.lang_modalities,
        )
        self._init_dimensions(
            config.dit_dim,
            config.n_heads,
            config.lowdim_obs_dim,
            config.action_dim,
            config.act_window_size,
            config.multistep,
            config.num_sampling_steps,
        )
        self._init_flags(
            config.first_view_key,
            config.second_view_key,
            config.use_second_view,
            config.use_causal_attention,
            config.use_cross_attn,
            config.use_adaln_cond,
            config.use_readout_token,
            config.use_rope,
            config.use_nope,
            config.vlm_prompt_style,
            config.token_dropout,
            config.action_type_adaln,
            config.sampling_type,
            config.use_proprio,
            config.return_act_chunk,
            config.cfg_dropout,
            config.cfg_lambda,
        )

        logger.info("Configuration (modalities, dimensions, flags) initialized.")
        # Initialize action space index
        self.action_space_index = ActionIndex()

        self._setup_vlm(
            config.vlm_path,
            config.freeze_vision_tower,
            config.freeze_florence,
            config.freeze_embeddings_only,
        )
        hidden_dim = self.vlm.config.text_config.d_model
        self.vlm_latent_dim = hidden_dim
        self.use_dopri5 = False
        self._setup_dit_components(
            config.dit_dim,
            config.n_heads,
            config.n_layers,
            config.action_dim,
            config.act_window_size,
            hidden_dim,
            config.attn_pdrop,
            config.resid_pdrop,
            config.mlp_pdrop,
            config.use_cross_attn,
            config.use_rope,
            config.use_nope,
            config.query_seq_len,
            config.rope_theta,
        )
        logger.info("VLM and DiT components set up.")

        # Initialize rollout state.
        self.rollout_step_counter = 0
        self.pred_action_seq = None

        # Ensure that all parameters and buffers are on the correct device.
        self.ensure_device_consistency()

    # === Initialization Helpers ===
    def _init_modalities(
        self,
        target_modality: str,
        obs_modalities: str,
        goal_modalities: str,
        img_modalities: List[str],
        lang_modalities: List[str],
    ) -> None:
        """Initializes modality-related attributes."""
        self.target_modality = target_modality
        self.obs_modalities = obs_modalities
        self.goal_modalities = goal_modalities
        self.img_modalities = img_modalities
        self.lang_modalities = lang_modalities

    def _init_dimensions(
        self,
        dit_dim: int,
        n_heads: int,
        lowdim_obs_dim: int,
        action_dim: int,
        act_window_size: int,
        multistep: int,
        num_sampling_steps: int,
    ) -> None:
        """Initializes dimension-related attributes and checks consistency."""
        if dit_dim % n_heads != 0:
            raise ValueError(
                f"dit_dim ({dit_dim}) must be divisible by n_heads ({n_heads})"
            )
        self.lowdim_obs_dim = lowdim_obs_dim
        self.action_dim = action_dim
        self.act_window_size = act_window_size
        self.multistep = multistep
        self.num_sampling_steps = num_sampling_steps
        self.dit_dim = dit_dim

    def _init_flags(
        self,
        first_view_key: str,
        second_view_key: str,
        use_second_view: bool,
        use_causal_attention: bool,
        use_cross_attn: bool,
        use_adaln_cond: bool,
        use_readout_token: bool,
        use_rope: bool,
        use_nope: bool,
        vlm_prompt_style: str,
        token_dropout: float,
        action_type_adaln: bool,
        sampling_type: str,
        use_proprio: bool,
        return_act_chunk: bool,
        cfg_dropout: float,
        cfg_lambda: float,
    ) -> None:
        """Initializes boolean flags and related parameters."""
        if vlm_prompt_style not in ["default", "feature_focused", "state_oriented"]:
            raise ValueError("Invalid VLM prompt style")
        if sampling_type not in [
            "ln",
            "pi_zero",
            "loglogistic",
            "uniform",
            "stratified",
        ]:
            raise ValueError(f"Invalid sampling type: {sampling_type}")
        self.first_view_key = first_view_key
        self.second_view_key = second_view_key
        self.use_second_view = use_second_view
        self.use_causal_attention = use_causal_attention
        self.use_cross_attn = use_cross_attn
        self.use_adaln_cond = use_adaln_cond
        self.use_readout_token = use_readout_token
        self.use_rope = use_rope
        self.use_nope = use_nope
        self.use_proprio = use_proprio
        self.return_act_chunk = return_act_chunk
        self.vlm_prompt_style = vlm_prompt_style
        self.token_dropout = token_dropout
        self.action_type_adaln = action_type_adaln
        self.sampling_type = sampling_type
        self.cfg_dropout = cfg_dropout
        self.cfg_lambda = cfg_lambda

    def _setup_vlm(
        self,
        vlm_path: str,
        freeze_vision_tower: bool,
        freeze_florence: bool,
        freeze_embeddings_only: bool,
    ) -> None:
        """
        Loads the pretrained VLM, sets up the processor/tokenizer, adds a prompt token,
        and optionally freezes parameters.
        """
        logger.info(f"Loading VLM from {vlm_path}")
        self.vlm = AutoModelForCausalLM.from_pretrained(
            vlm_path,
            trust_remote_code=True,
            attn_implementation="eager",  # Fix for Florence-2 SDPA issue
        )
        self.train_vlm = not freeze_florence

        if freeze_florence:
            for param in self.vlm.parameters():
                param.requires_grad = False
        elif freeze_embeddings_only:
            embedding_layer = self.vlm.get_input_embeddings()
            for param in embedding_layer.parameters():
                param.requires_grad = False
            if hasattr(self.vlm.language_model, "shared"):
                for param in self.vlm.language_model.shared.parameters():
                    param.requires_grad = False

        if not freeze_vision_tower:
            for param in self.vlm.vision_tower.parameters():
                param.requires_grad = True

        self.processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        if (
            self.tokenizer.pad_token is None
        ):  # from flower_vla_pret but not really needed here
            print("setting padding and eos token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.prompt_embeds = self._create_prompt_embed("<Flow>").to(self.device)
        del self.vlm.language_model.model.decoder, self.vlm.language_model.lm_head
        self.vlm_token_dropout = nn.Dropout(self.token_dropout)
        # self.vlm_latent_dim = self.vlm.config.text_config.d_model

    def _setup_dit_components(
        self,
        dit_dim: int,
        n_heads: int,
        n_layers: int,
        action_dim: int,
        act_window_size: int,
        hidden_dim: int,
        attn_pdrop: float,
        resid_pdrop: float,
        mlp_pdrop: float,
        use_cross_attn: bool,
        use_rope: bool,
        use_nope: bool,
        query_seq_len: int,
        rope_theta: float,
    ) -> None:
        """
        Sets up the Diffusion Transformer (DiT) components including action-specific
        encoders/decoders and shared conditioning components.
        """
        # Initialize module dictionaries
        self.action_encoders = nn.ModuleDict()
        self.action_decoders = nn.ModuleDict()
        if self.use_proprio:
            self.proprio_encoders = nn.ModuleDict()
        self.adaln = nn.ModuleDict() if self.action_type_adaln else None

        # Set up action-specific components
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            input_dim = self.action_space_index.get_action_dim(action_idx)

            # Action encoder/decoder
            self.action_encoders[action_name] = Mlp(
                in_features=input_dim,
                hidden_features=dit_dim,
                out_features=dit_dim,
                bias=True,
            )
            self.action_decoders[action_name] = nn.Linear(dit_dim, input_dim).to(
                self.device
            )

            # Action-specific AdaLN
            if self.action_type_adaln:
                self.adaln[action_name] = SharedAdaLNController(
                    dit_dim, global_conddim=dit_dim, use_cross_attn=use_cross_attn
                )

            # Proprioceptive encoders
            if self.use_proprio:
                if action_name == "bimanual_nav":
                    self.proprio_encoders[action_name] = Mlp(
                        input_dim, dit_dim, out_features=dit_dim, drop=0.2
                    ).to(self.device)
                else:
                    self.proprio_encoders[action_name] = ZeroEncoder(
                        self.dit_dim, device=self.device
                    )

        # Set up shared AdaLN if not using action-specific AdaLN
        if not self.action_type_adaln:
            self.adaln = SharedAdaLNController(
                dit_dim, global_conddim=dit_dim, use_cross_attn=use_cross_attn
            )

        # Set up shared conditioning components
        self.cond_linear = nn.Linear(hidden_dim, dit_dim, bias=False)
        self.t_embedder = TimestepEmbedder(dit_dim)
        self.cond_norm = RmsNorm(hidden_dim)
        self.frequency_embedder = FreqEmbedder(dit_dim)
        self.action_space_embedder = ActionSpaceEmbedderParameter(
            dit_dim, max_actions=len(self.action_space_index.action_spaces)
        )

        # Set up positional encoding if neither RoPE nor NoPE is used
        if not use_rope and not use_nope:
            self.positional_encoding = nn.Parameter(
                torch.randn(1, act_window_size, dit_dim) * 0.1
            )

        # Set up DiT blocks
        self.dit = nn.ModuleList(
            [
                FlowBlock(
                    dim=dit_dim,
                    heads=n_heads,
                    attn_pdrop=attn_pdrop,
                    resid_pdrop=resid_pdrop,
                    mlp_pdrop=mlp_pdrop,
                    use_cross_attn=use_cross_attn,
                    use_rope=use_rope,
                    query_seq_len=query_seq_len,
                    rope_theta=rope_theta,
                )
                for _ in range(n_layers)
            ]
        )

    def _verify_device_consistency(self) -> None:
        """Verifies that all parameters and buffers are on the expected device."""
        expected = self.device
        inconsistent = []
        for name, param in self.named_parameters():
            if param.device != expected:
                inconsistent.append(f"{name}: {param.device} (expected {expected})")
        for name, buf in self.named_buffers():
            if buf.device != expected:
                inconsistent.append(
                    f"{name} (buffer): {buf.device} (expected {expected})"
                )
        if inconsistent:
            logger.warning("Device consistency issues: " + "; ".join(inconsistent))

    def ensure_device_consistency(self) -> None:
        """Moves the entire model (and buffers) to the designated device."""
        self.to(self.device)
        self.vlm.to(self.device)
        if not self.use_rope and hasattr(self, "positional_encoding"):
            self.positional_encoding = self.positional_encoding.to(self.device)
        if self.use_readout_token and hasattr(self, "register_token"):
            self.register_token = self.register_token.to(self.device)
        self._verify_device_consistency()

    def _create_prompt_embed(self, prompt_text: str) -> nn.Parameter:
        """
        Creates a prompt embedding. Adds the prompt token to the tokenizer
        and returns its embedding (frozen).
        """
        self.tokenizer.add_special_tokens({"additional_special_tokens": [prompt_text]})
        self.vlm.resize_token_embeddings(len(self.tokenizer))
        prompt_token_id = self.tokenizer.convert_tokens_to_ids(prompt_text)
        prompt_embed = nn.Parameter(
            self.vlm.get_input_embeddings()(torch.tensor(prompt_token_id)),
            requires_grad=False,
        )
        return prompt_embed.unsqueeze(0).unsqueeze(0)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass (training mode).

        Args:
            batch: Dictionary with observation and action data

        Returns:
            Dictionary with loss and other outputs
        """
        obs_features = self.encode_observations(batch)
        dataset_idx = batch.get("task.dataset_index", None)
        action_loss, losses_dict = self.rf_loss(
            obs_features, batch[self.target_modality], dataset_idx
        )

        return {"loss": action_loss, "loss_dict": losses_dict}

    def encode_observations(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Encodes primary (and optional second view) image observations and text goals.
        Returns a dictionary with:
            - 'features': Encoder outputs.
            - 'frequency_embeds': Frequency embeddings.
            - 'action_space_embeds': Action space embeddings.
            - 'action_type': Action type indices.
            - 'proprio': Proprioception data (if available).
            - 'attention_mask': Attention mask.
        """
        device = self.device
        default_dtype = next(self.parameters()).dtype
        # Debug: print available keys
        image_tensor = batch[self.first_view_key]
        # Handle both 4D [B, C, H, W] and 5D [B, T, C, H, W] image tensors
        if len(image_tensor.shape) == 4:
            # Shape is [B, C, H, W], add temporal dimension
            B, C, H, W = image_tensor.shape
            T = 1
            image_tensor = image_tensor.unsqueeze(1)  # [B, 1, C, H, W]
        else:
            B, T, C, H, W = image_tensor.shape

        image_features = self.vlm._encode_image(
            image_tensor.view(-1, C, H, W).to(device).to(default_dtype)
        )
        image_features = image_features.view(B, T * image_features.shape[1], -1)

        if self.use_second_view and self.second_view_key in batch:
            image2_tensor = batch[self.second_view_key]

            # Handle both 4D and 5D for second view as well
            if len(image2_tensor.shape) == 4:
                image2_tensor = image2_tensor.unsqueeze(1)

            image2_features = self.vlm._encode_image(
                image2_tensor.view(-1, C, H, W).to(device).to(default_dtype)
            )
            image2_features = image2_features.view(B, T * image2_features.shape[1], -1)
            image_features = torch.cat([image_features, image2_features], dim=1)

        if "task" in batch:
            task_text = batch["task"]
            # SAFEGUARD: Ensure it is a list/tuple of strings matching batch size
            if isinstance(task_text, str):
                # Handle edge case where it might be a single string (e.g. batch size 1 or broadcasting)
                task_text = [task_text] * B
            elif isinstance(task_text, tuple):
                task_text = list(task_text)

            # Tokenize the text
            tokenized = self.tokenizer(
                task_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            text_embeds = self.vlm.get_input_embeddings()(
                tokenized["input_ids"].to(device)
            ).to(device)
            lang_attention_mask = tokenized["attention_mask"].to(device)
        else:
            dummy_text = [""] * B
            tokenized = self.tokenizer(
                dummy_text, return_tensors="pt", padding=True, max_length=128
            )
            text_embeds = self.vlm.get_input_embeddings()(
                tokenized["input_ids"].to(device)
            ).to(device)
            lang_attention_mask = tokenized["attention_mask"].to(device)

        # get the flow prompt for florence
        task_prompt = self.prompt_embeds.expand(B, -1, -1)
        merged_embeds = torch.cat(
            [
                task_prompt.to(image_features.device),
                image_features,
                text_embeds.to(image_features.device),
            ],
            dim=1,
        )

        # get attention mask from txt
        # lang_attention_mask was already created during tokenization above
        # define attention mask for image
        vis_attention_mask = torch.ones(
            image_features.shape[:2], device=image_features.device
        )
        prompt_mask = torch.zeros(B, 1, dtype=torch.bool, device=image_features.device)
        attention_mask = torch.cat(
            [prompt_mask, vis_attention_mask, lang_attention_mask], dim=1
        )

        features = self.vlm.get_encoder()(
            inputs_embeds=merged_embeds,
            attention_mask=attention_mask,
        ).last_hidden_state

        features = self.vlm_token_dropout(features)

        # add optinal cfg dropout
        if self.cfg_dropout > 0 and self.training:
            prompt_length = task_prompt.shape[1]
            image_length = image_features.shape[1]
            text_length = text_embeds.shape[1]  # assumed fixed length per example
            text_start = prompt_length + image_length
            text_end = (
                text_start + text_length
            )  # text features occupy features[:, text_start:text_end, :]
            # Create a dropout mask for the entire batch (per example)
            drop_mask = (
                (torch.rand(B, device=device) < self.cfg_dropout).float().view(B, 1, 1)
            )
            # Apply the mask only to the text portion of the features.
            features[:, text_start:text_end, :] = features[
                :, text_start:text_end, :
            ] * (1 - drop_mask)
        # TODO: think about moving some of these initializations to processor class
        return {
            "features": features,
            "frequency_embeds": self.frequency_embedder(
                batch.get(
                    "task.frequency",
                    torch.ones(B, 1, device=device, dtype=default_dtype)
                    * 15,  # TODO: fix hardcoding of frequency
                )
                .to(device)
                .to(default_dtype)
            ),
            "action_space_embeds": self.action_space_embedder(
                torch.full(
                    (B,),
                    fill_value=self.action_space_index.action_space_mapping.get(
                        ("JOINT_POS", "position", 1)
                    ),
                    dtype=torch.long,
                    device=device,
                )
                # TODO: fix hardcoding, read the robot_type, the control type and the number of arms directly from the dataset
            ),
            "action_type": torch.full(
                (B,),
                fill_value=self.action_space_index.action_space_mapping.get(
                    ("JOINT_POS", "position", 1)
                ),
                dtype=torch.long,
                device=device,
            ),
            "proprio": batch["observation.state"].to(device).to(default_dtype)
            if self.use_proprio and "observation.state" in batch
            else None,
            "attention_mask": attention_mask,
        }

    def encode_actions(
        self, z: torch.Tensor, action_type: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes actions for each sample based on its action type.
        Returns:
            - Encoded actions (latent representations).
            - A valid dimensions mask.
        """
        default_dtype = next(self.parameters()).dtype
        action_type = action_type.to(self.device)
        B = z.shape[0]
        encoded = torch.zeros(
            B, z.shape[1], self.dit_dim, device=self.device, dtype=default_dtype
        )
        valid_dims = torch.zeros_like(z, dtype=default_dtype)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = action_type == action_idx
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                valid_dims[mask, :, :adim] = 1
                encoded[mask] = self.action_encoders[action_name](z[mask, :, :adim])
        return encoded, valid_dims

    def decode_actions(
        self, z: torch.Tensor, action_type: torch.Tensor, valid_dims: torch.Tensor
    ) -> torch.Tensor:
        """
        Decodes latent representations into actual actions.
        Only the dimensions corresponding to valid action spaces are active.
        """
        default_dtype = next(self.parameters()).dtype
        B = z.shape[0]
        max_action_dim = self.action_dim
        decoded = torch.zeros(
            B, z.shape[1], max_action_dim, device=self.device, dtype=default_dtype
        )
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = action_type == action_idx
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                pred = self.action_decoders[action_name](z[mask])
                decoded[mask, :, :adim] = pred[..., :adim] * valid_dims[mask, :, :adim]
        return decoded

    def encode_proprio(
        self, proprio: torch.Tensor, action_type: torch.Tensor, output_shape
    ) -> torch.Tensor:
        """
        Encodes proprioceptive data based on action type.
        Returns a tensor with shape [batch, dit_dim].
        """
        batch_size, _ = output_shape
        dtype = next(self.parameters()).dtype

        if not self.use_proprio:
            return torch.zeros(batch_size, self.dit_dim, device=self.device)

        encoded = torch.zeros(batch_size, self.dit_dim, device=self.device, dtype=dtype)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = action_type == action_idx
            if mask.any():
                encoded[mask] = (
                    self.proprio_encoders[action_name](proprio[mask])
                    .squeeze(1)
                    .to(dtype)
                )

        return encoded

    def dit_forward(
        self, z: torch.Tensor, t: torch.Tensor, cond_dict: dict
    ) -> torch.Tensor:
        """
        Forward pass through the DiT blocks.
        Encodes actions, adds positional information, and applies conditioning.
        """
        B, t_seq, d = z.shape
        dtype = next(self.parameters()).dtype
        # Extract and process conditioning inputs
        cond = self.cond_linear(self.cond_norm(cond_dict["features"].to(dtype)))
        freq_embeds = cond_dict["frequency_embeds"].squeeze(1).to(dtype)
        action_type = cond_dict["action_type"].to(self.device)
        proprio = (
            cond_dict.get("proprio", torch.zeros_like(freq_embeds)).to(dtype)
            if self.use_proprio
            else torch.zeros_like(freq_embeds)
        )
        proprio_embeds = self.encode_proprio(
            proprio, action_type, freq_embeds.shape
        ).to(dtype)

        # Encode actions and positional information
        z, valid_dims = self.encode_actions(z, action_type)
        if not (self.use_rope or self.use_nope):
            z += self.positional_encoding

        # Apply CFG dropout on freq_embeds and proprio_embeds only
        if self.training and self.cfg_dropout > 0:
            # Create a binary dropout mask per example (shape: [B, 1])
            drop_mask = (
                (
                    torch.rand(freq_embeds.size(0), device=freq_embeds.device)
                    < self.cfg_dropout
                )
                .float()
                .unsqueeze(1)
            )
            # Zero out the embeddings for examples where dropout is active
            freq_embeds = freq_embeds * (1 - drop_mask)
            proprio_embeds = proprio_embeds * (1 - drop_mask)
        # Compute temporal embedding
        t_emb = sum(
            map(stateless_norm, [self.t_embedder(t), freq_embeds, proprio_embeds])
        )

        # Compute global conditioning
        if self.use_adaln_cond:
            global_cond = cond[:, 0, :] if self.use_readout_token else cond.mean(dim=1)
            global_cond += t_emb
        else:
            global_cond = t_emb

        context = cond if self.use_cross_attn else None

        # Compute AdaLN modulation
        global_adaln = (
            self.adaln(global_cond)
            if not self.action_type_adaln
            else self.action_specific_adaln(global_cond, action_type)
        )

        for layer in self.dit:
            z = layer(
                z,
                global_cond,
                context=context,
                custom_attn_mask=None,
                custom_cross_attn_mask=cond_dict["attention_mask"],
                is_causal=True,
                global_adaln=global_adaln,
            )

        # Decode actions
        return self.decode_actions(z, action_type, valid_dims)

    def action_specific_adaln(
        self, global_cond: torch.Tensor, action_type: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Computes action-specific AdaLN modulation signals.
        Returns a list of modulation tensors.
        """
        dtype = next(self.parameters()).dtype
        batch_size = global_cond.shape[0]
        num_chunks = 9 if self.use_cross_attn else 6

        mod_signals = [
            torch.zeros(batch_size, self.dit_dim, device=self.device, dtype=dtype)
            for _ in range(num_chunks)
        ]

        for action_idx in range(len(self.action_space_index.action_spaces)):
            mask = action_type == action_idx
            if mask.any():
                action_name = self.action_space_index.get_action_name(action_idx)
                action_mod = self.adaln[action_name](global_cond[mask])
                for i, signal in enumerate(action_mod):
                    mod_signals[i][mask] = signal

        return mod_signals

    # === Loss Functions ===
    def rf_loss(
        self, cond: dict, actions: torch.Tensor, dataset_idx: Any = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Computes the rectified flow loss.
        Interpolates between actions and noise, then computes MSE only over valid dimensions.
        """
        default_dtype = next(self.parameters()).dtype
        action_type = cond["action_type"]
        if len(actions.shape) == 4:
            actions = actions.squeeze(1)
        b = actions.size(0)
        device = actions.device
        actions = actions.to(default_dtype)

        # Sample time t based on the chosen distribution.
        if self.sampling_type == "pi_zero":
            alpha, beta = 1.5, 1.0
            t = (
                torch.distributions.Beta(alpha, beta)
                .sample((b,))
                .to(device)
                .clamp(max=0.999)
            )
        elif self.sampling_type == "ln":
            t = (
                torch.sigmoid(torch.randn((b,), device=device))
                .clamp(max=0.999)
                .to(default_dtype)
            )
        elif self.sampling_type == "uniform":
            eps = 1e-5
            t = (torch.rand(1, device=device) + torch.arange(b, device=device) / b) % (
                1 - eps
            )
            t = t.to(default_dtype)
        else:
            raise NotImplementedError(
                f"Sampling type {self.sampling_type} not implemented"
            )
        texp = t.view([b] + [1] * (actions.dim() - 1))
        z1 = torch.zeros_like(actions)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = action_type == action_idx
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                noise_slice = torch.randn(
                    (mask.sum(), actions.size(1), adim),
                    dtype=actions.dtype,
                    device=actions.device,
                )
                z1[mask, :, :adim] = noise_slice
        zt = (1 - texp) * actions + texp * z1
        vtheta = self.dit_forward(zt, t, cond)
        valid_mask = torch.zeros_like(actions, dtype=torch.bool)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = action_type == action_idx
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                mask_expanded = (
                    mask.view(-1, 1, 1).expand(-1, actions.size(1), adim).to(device)
                )
                valid_mask[mask, :, :adim] = mask_expanded[mask]
        diff = (z1 - actions) - vtheta
        valid_diff = diff[valid_mask]
        loss = (valid_diff**2).mean()
        losses_dict = {
            "diff_min": valid_diff.min().item(),
            "diff_max": valid_diff.max().item(),
            "diff_mean": valid_diff.mean().item(),
            "loss": loss.item(),
        }
        return loss, losses_dict

    # === Sampling Methods ===
    def sample_actions(
        self, z: torch.Tensor, cond: Dict[str, torch.Tensor], inference: bool = False
    ) -> torch.Tensor:
        """
        Samples actions from the DiT model.
        Chooses between an adaptive ODE solver and fixed-step Euler integration.
        """
        action_type = cond["action_type"]
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = action_type == action_idx
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                z[mask, :, adim:] = 0.0
        if hasattr(self, "use_dopri5") and self.use_dopri5:
            return self._sample_with_adaptive_solver(z, cond)
        else:
            return self._sample_with_fixed_steps(z, cond, inference)

    def _sample_with_adaptive_solver(
        self, z: torch.Tensor, cond: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Samples actions using an adaptive ODE solver (dopri5).
        """
        device = z.device
        action_type = cond["action_type"]

        def ode_func(t, z):
            b = z.size(0)
            t_tensor = t * torch.ones(b, device=device)
            with torch.no_grad():
                z = z.clone()
                for (
                    action_name,
                    action_idx,
                ) in self.action_space_index.action_spaces.items():
                    mask = action_type == action_idx
                    if mask.any():
                        adim = self.action_space_index.get_action_dim(action_idx)
                        z[mask, :, adim:] = 0.0
                v = self.dit_forward(z, t_tensor, cond)
                for (
                    action_name,
                    action_idx,
                ) in self.action_space_index.action_spaces.items():
                    mask = action_type == action_idx
                    if mask.any():
                        adim = self.action_space_index.get_action_dim(action_idx)
                        v[mask, :, adim:] = 0.0
            return v

        t_span = torch.tensor([1.0, 0.0], device=device)
        z = odeint(
            ode_func,
            z,
            t_span,
            method="dopri5",
            rtol=1e-4,
            atol=1e-4,
            options={
                "max_num_steps": max(self.num_sampling_steps * 2, 1000),
                "min_step": 1.0 / self.num_sampling_steps,
            },
        )[-1]
        return z.clamp(-1, 1)

    def _sample_with_fixed_steps(
        self, z: torch.Tensor, cond: Dict[str, torch.Tensor], inference: bool = False
    ) -> torch.Tensor:
        """
        Samples actions using fixed-step Euler integration.
        """
        steps = self.num_sampling_steps
        b = z.size(0)
        device = z.device
        action_type = cond["action_type"]
        dt = 1.0 / steps
        dt_tensor = torch.tensor([dt] * b, device=device).view(
            [b] + [1] * (z.dim() - 1)
        )

        # Only apply CFG during inference
        apply_cfg = inference and self.cfg_lambda != 1.0

        # Create null conditioning once outside the loop
        if apply_cfg:
            # Create a deep copy of the conditioning dictionary
            null_cond = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in cond.items()
            }

            # Clone features to avoid modifying original
            features = null_cond["features"].clone()

            # Calculate where text features begin based on your encode_observations method
            prompt_length = self.prompt_embeds.shape[1]
            image_length = 50 if self.use_second_view is False else 100

            # Zero out only the text portion (after prompt and image features)
            text_start = prompt_length + image_length
            features[:, text_start:, :] = 0.0

            # Update features in null_cond
            null_cond["features"] = features

        for i in range(steps, 0, -1):
            t_val = i / steps
            t_tensor = torch.full((b,), t_val, device=device)

            # Get conditional velocity
            vc = self.dit_forward(z, t_tensor, cond)

            # Apply CFG if needed
            if apply_cfg:
                vu = self.dit_forward(z, t_tensor, null_cond)
                vc = vu + self.cfg_lambda * (vc - vu)

            # Euler step
            z = z - dt_tensor * vc

            # Apply action masking
            for (
                action_name,
                action_idx,
            ) in self.action_space_index.action_spaces.items():
                mask = action_type == action_idx
                if mask.any():
                    adim = self.action_space_index.get_action_dim(action_idx)
                    z[mask, :, adim:] = 0.0

        return z.clamp(-1, 1)

    def reset(self) -> None:
        """
        Resets the rollout state.
        """
        self.rollout_step_counter = 0
        self.pred_action_seq = None
        self.eval()
