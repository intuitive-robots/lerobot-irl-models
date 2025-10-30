import logging
from typing import Dict, Tuple, List, Any

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForCausalLM
from timm.layers.mlp import Mlp
from lerobot.policies.pretrained import PreTrainedPolicy
from .flower_config import FlowerVLAConfig
from .action_index import ActionIndex
from .transformes import (
    TimestepEmbedder,
    SharedAdaLNController,
    RmsNorm,
    FreqEmbedder,
    ActionSpaceEmbedderParameter,
    ZeroEncoder,
    FlowBlock,
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
        # dataset_stats: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ):
        """
        Initialize FlowerVLA Policy.

        Args:
            config: FlowerVLAConfig instance with all hyperparameters
            dataset_stats: Optional normalization statistics (not used, kept for compatibility)
        """
        super().__init__(config)

        config.validate_features()
        self.config = config
        # self.dataset_stats = dataset_stats
        self.model = FlowerModel(config)

        # Expose important attributes from model for easy access
        self.action_dim = self.model.action_dim
        self.act_window_size = self.model.act_window_size
        self.return_act_chunk = self.model.return_act_chunk
        self.device = self.model.device

        self.reset()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass (training mode).

        Args:
            batch: Dictionary with observation and action data

        Returns:
            Dictionary with loss and other outputs
        """
        # Delegate to model
        return self.model.forward(batch)

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

    @torch.no_grad()
    def select_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Select action for inference (LeRobot protocol).

        Args:
            batch: Observation batch

        Returns:
            Selected action [B, action_dim]
        """
        # Encode observations
        cond = self.model.encode_observations(batch)

        # Sample noise
        B = cond["features"].shape[0]
        noise = torch.randn(
            B, self.act_window_size, self.action_dim, device=self.device
        )

        # Sample actions
        action_seq = self.model.sample_actions(noise, cond, inference=True)

        # Return first action
        if self.return_act_chunk:
            return action_seq  # [B, T, action_dim]
        else:
            return action_seq[:, 0, :]  # [B, action_dim]

    # ==================== Rollout Methods ====================

    def reset(self):
        """Reset rollout state."""
        self.rollout_step_counter = 0
        self.pred_action_seq = None
        self.eval()

    def get_optim_params(self) -> dict:
        return self.parameters()


class FlowerModel(nn.Module):
    """
    FlowerVLA Model combining Florence-2 VLM with Flow-based DiT.
    """

    def __init__(self, config: FlowerVLAConfig):
        super().__init__()
        self.config = config
        # Core attributes
        self.device = torch.device(
            config.device
            if hasattr(config, "device") and config.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.action_dim = config.action_dim
        self.act_window_size = config.act_window_size
        self.multistep = config.multistep
        self.num_sampling_steps = config.num_sampling_steps
        self.dit_dim = config.dit_dim

        # Modalities (required for data access)
        # These will be set by LeRobot's dataloader based on the config
        self.obs_modalities = "observation"
        self.goal_modalities = "task"
        self.target_modality = "action"
        self.lang_modalities = ["language_instruction"]
        self.second_view_key = config.second_view_key

        # Flags
        self.use_second_view = config.use_second_view
        self.use_cross_attn = config.use_cross_attn
        self.use_rope = config.use_rope
        self.use_nope = config.use_nope
        self.use_proprio = config.use_proprio
        self.return_act_chunk = config.return_act_chunk
        self.token_dropout = config.token_dropout
        self.cfg_dropout = config.cfg_dropout
        self.cfg_lambda = config.cfg_lambda
        self.sampling_type = config.sampling_type
        self.train_vlm = not config.freeze_florence
        self.use_adaln_cond = config.use_adaln_cond
        self.use_readout_token = config.use_readout_token
        self.action_type_adaln = config.action_type_adaln

        # Initialize action space index
        self.action_space_index = ActionIndex()

        # Setup VLM (Florence-2)
        self._setup_vlm(config)

        # Setup DiT components
        self._setup_dit_components(config)

        # Rollout state
        self.rollout_step_counter = 0
        self.pred_action_seq = None

    def _count_parameters(self) -> str:
        """Count trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"{trainable / 1e6:.1f}M/{total / 1e6:.1f}M trainable/total"

    def _setup_vlm(self, config: FlowerVLAConfig):
        """Setup Florence-2 VLM and tokenizer."""
        logger.info(f"Loading VLM from {config.vlm_path}")

        # Load VLM
        self.vlm = AutoModelForCausalLM.from_pretrained(
            config.vlm_path, trust_remote_code=True
        )

        # Freeze/unfreeze parameters
        if config.freeze_florence:
            for param in self.vlm.parameters():
                param.requires_grad = False
        elif config.freeze_embeddings_only:
            # Freeze only embeddings
            embedding_layer = self.vlm.get_input_embeddings()
            for param in embedding_layer.parameters():
                param.requires_grad = False
            if hasattr(self.vlm.language_model, "shared"):
                for param in self.vlm.language_model.shared.parameters():
                    param.requires_grad = False

        if not config.freeze_vision_tower:
            for param in self.vlm.vision_tower.parameters():
                param.requires_grad = True

        # Setup processor and tokenizer
        self.processor = AutoProcessor.from_pretrained(
            config.vlm_path, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer

        # Create special <Flow> prompt token
        self.prompt_embeds = self._create_prompt_embed("<Flow>").to(self.device)

        # Remove decoder (encoder-only mode)
        del self.vlm.language_model.model.decoder, self.vlm.language_model.lm_head

        # Token dropout
        self.vlm_token_dropout = nn.Dropout(self.token_dropout)

        # Store VLM hidden dim
        self.vlm_latent_dim = self.vlm.config.text_config.d_model

    def _setup_dit_components(self, config: FlowerVLAConfig):
        """Setup DiT (Diffusion Transformer) components."""
        hidden_dim = self.vlm_latent_dim

        # Initialize module dictionaries
        self.action_encoders = nn.ModuleDict()
        self.action_decoders = nn.ModuleDict()
        if self.use_proprio:
            self.proprio_encoders = nn.ModuleDict()
        self.adaln = nn.ModuleDict() if config.action_type_adaln else None

        # Setup action-specific components for each action space
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            input_dim = self.action_space_index.get_action_dim(action_idx)

            # Action encoder (input_dim -> dit_dim)
            self.action_encoders[action_name] = Mlp(
                in_features=input_dim,
                hidden_features=config.dit_dim,
                out_features=config.dit_dim,
                bias=True,
            )

            # Action decoder (dit_dim -> input_dim)
            self.action_decoders[action_name] = nn.Linear(config.dit_dim, input_dim)

            # Action-specific AdaLN
            if config.action_type_adaln:
                self.adaln[action_name] = SharedAdaLNController(
                    config.dit_dim,
                    global_conddim=config.dit_dim,
                    use_cross_attn=config.use_cross_attn,
                )

            # Proprioceptive encoders
            if self.use_proprio:
                if action_name == "bimanual_nav":
                    self.proprio_encoders[action_name] = Mlp(
                        input_dim, config.dit_dim, out_features=config.dit_dim, drop=0.2
                    )
                else:
                    self.proprio_encoders[action_name] = ZeroEncoder(
                        config.dit_dim, device=self.device
                    )

        # Setup shared AdaLN if not using action-specific
        if not config.action_type_adaln:
            self.adaln = SharedAdaLNController(
                config.dit_dim,
                global_conddim=config.dit_dim,
                use_cross_attn=config.use_cross_attn,
            )

        # Shared conditioning components
        self.cond_linear = nn.Linear(hidden_dim, config.dit_dim, bias=False)
        self.t_embedder = TimestepEmbedder(config.dit_dim)
        self.cond_norm = RmsNorm(hidden_dim)
        self.frequency_embedder = FreqEmbedder(config.dit_dim)
        self.action_space_embedder = ActionSpaceEmbedderParameter(
            config.dit_dim, max_actions=len(self.action_space_index.action_spaces)
        )

        # Positional encoding (if not using RoPE or NoPE)
        if not config.use_rope and not config.use_nope:
            self.positional_encoding = nn.Parameter(
                torch.randn(1, config.act_window_size, config.dit_dim) * 0.1
            )

        # DiT blocks
        self.dit = nn.ModuleList(
            [
                FlowBlock(
                    dim=config.dit_dim,
                    heads=config.n_heads,
                    attn_pdrop=config.attn_pdrop,
                    resid_pdrop=config.resid_pdrop,
                    mlp_pdrop=config.mlp_pdrop,
                    use_cross_attn=config.use_cross_attn,
                    use_rope=config.use_rope,
                    query_seq_len=config.query_seq_len,
                    rope_theta=config.rope_theta,
                )
                for _ in range(config.n_layers)
            ]
        )

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
        # Encode observations
        cond = self.encode_observations(batch)

        # Extract actions (support both formats)
        if "action" in batch:
            actions = batch["action"]
        else:
            actions = batch[self.target_modality]

        # Compute loss
        loss, loss_dict = self.rf_loss(cond, actions)

        return {"loss": loss, "loss_dict": loss_dict}

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
        image_tensor = batch[self.obs_modalities]["image_primary"]
        B, T, C, H, W = image_tensor.shape
        image_features = self.vlm._encode_image(
            image_tensor.view(-1, C, H, W).to(device).to(default_dtype)
        )
        image_features = image_features.view(B, T * image_features.shape[1], -1)

        if self.use_second_view and self.second_view_key in batch[self.obs_modalities]:
            image2_tensor = batch[self.obs_modalities][self.second_view_key]
            image2_features = self.vlm._encode_image(
                image2_tensor.view(-1, C, H, W).to(device).to(default_dtype)
            )
            image2_features = image2_features.view(B, T * image2_features.shape[1], -1)
            image_features = torch.cat([image_features, image2_features], dim=1)

        text_embeds = (
            self.vlm.get_input_embeddings()(
                batch[self.goal_modalities][self.lang_modalities[0]]["input_ids"].to(
                    device
                )
            )
            .to(device)
            .squeeze(1)
        )

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
        attention_mask = torch.ones(
            merged_embeds.shape[:2], device=merged_embeds.device
        )
        lang_attention_mask = (
            batch[self.goal_modalities][self.lang_modalities[0]]["attention_mask"]
            .to(device)
            .squeeze(1)
        )
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

        return {
            "features": features,
            "frequency_embeds": self.frequency_embedder(
                batch[self.goal_modalities]["frequency"].to(device).to(default_dtype)
            ),
            "action_space_embeds": self.action_space_embedder(
                batch[self.goal_modalities]["action_space_index"].to(device)
            ),
            "action_type": batch[self.goal_modalities]["action_space_index"],
            "proprio": batch[self.obs_modalities]["proprio"]
            .to(device)
            .to(default_dtype)
            if self.use_proprio and "proprio" in batch[self.obs_modalities]
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

    def sample_actions(
        self, z: torch.Tensor, cond: Dict[str, torch.Tensor], inference: bool = False
    ) -> torch.Tensor:
        """
        Samples actions from the DiT model.
        Chooses between an adaptive ODE solver and fixed-step Euler integration.
        """
        steps = self.num_sampling_steps if inference else 5
        b = z.size(0)
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
        Requires torchdiffeq package.
        """
        try:
            from torchdiffeq import odeint
        except ImportError:
            raise ImportError(
                "torchdiffeq is required for adaptive ODE solver. "
                "Install with: pip install torchdiffeq"
            )

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
        Supports Classifier-Free Guidance (CFG) during inference.
        """
        steps = self.num_sampling_steps if inference else 5
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

            # Calculate where text features begin based on encode_observations method
            prompt_length = self.prompt_embeds.shape[1]
            image_length = 50 if not self.use_second_view else 100

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
