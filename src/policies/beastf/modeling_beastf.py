import logging
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from lerobot.policies.pretrained import PreTrainedPolicy
from timm.layers.mlp import Mlp
from transformers import AutoModelForCausalLM, AutoProcessor

from .beastf_config import BeastVLAConfig
# Assuming beast.py is in .beast_tokenizer package or similar
from .beast_tokenizer.beast import BeastTokenizer
from .beastf_utils import create_bidirectional_mask, token_prediction_accuracy

from lerobot.processor.normalize_processor import (
    NormalizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.utils.constants import ACTION

logger = logging.getLogger(__name__)


class BeastVLAPolicy(PreTrainedPolicy):
    """
    BeastVLA Policy for LeRobot.
    """
    name = "beast_vla"
    config_class = BeastVLAConfig

    def __init__(
        self,
        config: BeastVLAConfig,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        super().__init__(config)
        if dataset_stats is None and hasattr(config, '_dataset_stats'):
            dataset_stats = config._dataset_stats
            
        config.validate_features()
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
        
        self.model = BeastFModel(config)
        self.model.reset()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        result = self.model.forward(batch)
        return result["loss"], result["loss_dict"]

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self.forward(batch)

    @torch.no_grad()
    def predict_action_chunk(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        cond = self.model.encode_observations(batch)
        action_seq = self.model.sample_actions(None, cond, inference=True)
        action_seq = self.unnormalize_outputs({ACTION: action_seq})[ACTION]
        return action_seq

    def reset(self) -> None:
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
    
    def get_optim_params(self) -> dict:
        return self.model.parameters()
    


class BeastFModel(nn.Module):
    def __init__(self, config: BeastVLAConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(
            config.device if hasattr(config, "device") and config.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # --- Init Helpers ---
        self._init_modalities(config)
        self._init_flags(config)

        # --- Setup VLM ---
        self._setup_vlm(
            config.vlm_path,
            config.freeze_vision_tower,
            config.freeze_florence,
            config.freeze_embeddings_only,
        )
        
        # --- Setup Tokenizer ---
        self._setup_action_tokenizer(config)

        self.rollout_step_counter = 0
        self.pred_action_seq = None
        self.ensure_device_consistency()

    def _init_modalities(self, config):
        self.target_modality = config.target_modality
        self.obs_modalities = config.obs_modalities
        self.img_modalities = config.img_modalities
        self.lang_modalities = config.lang_modalities

    def _init_flags(self, config):
        self.use_second_view = config.use_second_view
        self.vlm_prompt_style = config.vlm_prompt_style
        self.token_dropout = config.token_dropout
        self.use_proprio = config.use_proprio
        self.return_act_chunk = config.return_act_chunk
        self.second_view_key = config.second_view_key

    def _setup_vlm(self, vlm_path, freeze_vision, freeze_florence, freeze_embed):
        logger.info(f"Loading VLM from {vlm_path}")
        self.vlm = AutoModelForCausalLM.from_pretrained(
            vlm_path, trust_remote_code=True, attn_implementation="eager"
        )
        
        if freeze_florence:
            for param in self.vlm.parameters(): param.requires_grad = False
        elif freeze_embed:
            for param in self.vlm.get_input_embeddings().parameters(): param.requires_grad = False

        if not freeze_vision:
            for param in self.vlm.vision_tower.parameters(): param.requires_grad = True

        self.processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        
        self.prompt_embeds = self._create_prompt_embed("<Primitives>").to(self.device)

    def _setup_action_tokenizer(self, config: BeastVLAConfig) -> None:
        """
        Initializes BeastTokenizer and updates VLM vocabulary to include action tokens.
        """
        # 1. Initialize the provided BeastTokenizer
        # Note: config.action_bins should be in config (e.g., 256)
        self.action_bins = getattr(config, "action_bins", 256)
        
        self.action_tokenizer = BeastTokenizer(
            num_dof=config.num_dof,
            num_basis=config.num_basis,
            seq_len=config.act_window_size,
            vocab_size=self.action_bins,  # Beast divides range into this many bins
            degree_p=getattr(config, "degree_p", 4),
            device=self.device,
        )
        self.update_w_bound = config.update_w_bound

        # use <loc_0> as the starting token for action tokens 
        self.action_token_start_id = self.tokenizer.convert_tokens_to_ids("<loc_0>")
        logger.info(f"Action tokens start at ID: {self.action_token_start_id}")

    def _create_prompt_embed(self, prompt_text: str) -> nn.Parameter:
        self.tokenizer.add_special_tokens({"additional_special_tokens": [prompt_text]})
        self.vlm.resize_token_embeddings(len(self.tokenizer))
        prompt_token_id = self.tokenizer.convert_tokens_to_ids(prompt_text)
        # Create frozen embedding
        prompt_embed = nn.Parameter(
            self.vlm.get_input_embeddings()(torch.tensor(prompt_token_id)),
            requires_grad=False,
        )
        return prompt_embed.unsqueeze(0).unsqueeze(0)

    def ensure_device_consistency(self) -> None:
        self.to(self.device)
        self.vlm.to(self.device)
        self.action_tokenizer.to(self.device) # Ensure tokenizer buffers are on device

    def _bins_to_llm_ids(self, bin_ids: torch.Tensor) -> torch.Tensor:
        """Convert BeastTokenizer bins to VLM token IDs."""
        return bin_ids + self.action_token_start_id

    def _llm_ids_to_bins(self, llm_ids: torch.Tensor) -> torch.Tensor:
        """Convert VLM token IDs back to BeastTokenizer bins."""
        bins = llm_ids - self.action_token_start_id
        # Clamp to ensure validity during early training/sampling noise
        return torch.clamp(bins, 0, self.action_bins - 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 1. Encode Observations
        encoded = self.encode_observations(batch)
        features = encoded['features']
        encoder_attn_mask = encoded['attention_mask']

        # 2. Prepare Targets
        actions = batch[self.target_modality]

        ### test: visualize reconstructed errors
        # self.action_tokenizer.visualize_reconstruction_error_discrete(actions)
        
        # Encode: Continuous Actions -> Discrete Bins (0..255)
        # BeastTokenizer.encode_discrete returns just the tokens tensor
        action_bins = self.action_tokenizer.encode_discrete(
            actions, 
            update_bounds=self.update_w_bound
        )
        
        # Map Bins -> VLM Input IDs
        llm_label_ids = self._bins_to_llm_ids(action_bins)

        # 3. Prepare Decoder Input
        # Use a filler token (e.g., middle of action range) as prompt for decoder
        # This matches the "empty action token" logic from original code
        B, SeqLen = llm_label_ids.shape
        filler_bin = torch.full((B, SeqLen), self.action_bins // 2, dtype=torch.long, device=self.device)
        llm_input_ids = self._bins_to_llm_ids(filler_bin)

        # 4. Bidirectional Mask
        bidirectional_mask = create_bidirectional_mask(
            batch_size=B, seq_length=SeqLen, device=self.device
        )

        # 5. Forward
        decoder_outputs = self.vlm.get_decoder()(
            input_ids=llm_input_ids,
            encoder_hidden_states=features,
            encoder_attention_mask=encoder_attn_mask,
            attention_mask=bidirectional_mask,
            use_cache=False,
        )

        lm_logits = self.vlm.language_model.get_output_embeddings()(decoder_outputs[0])
        lm_logits = lm_logits + self.vlm.language_model.final_logits_bias.to(lm_logits.device)

        # 6. Loss
        loss_fct = nn.CrossEntropyLoss()
        # View: [Batch*Seq, VocabSize] vs [Batch*Seq]
        masked_lm_loss = loss_fct(
            lm_logits.view(-1, self.vlm.config.vocab_size),
            llm_label_ids.view(-1),
        )

        # 7. Metrics (Optional)
        with torch.no_grad():
            pred_ids = torch.argmax(lm_logits, dim=-1)
            pred_bins = self._llm_ids_to_bins(pred_ids)
            # Decode: Discrete Bins -> Continuous Actions
            recon_actions = self.action_tokenizer.decode_discrete(pred_bins)
            mse = F.mse_loss(recon_actions, actions).item()
            token_pred_acc = token_prediction_accuracy(pred_ids, llm_label_ids)

        return {
            "loss": masked_lm_loss,
            "loss_dict": {"ce_loss": masked_lm_loss.item(), "mse": mse, "token_acc": token_pred_acc},
        }

    def encode_observations(self, batch: Dict) -> Dict[str, torch.Tensor]:
        device = self.device
        default_dtype = next(self.parameters()).dtype
        
        # Image encoding logic (simplified for brevity, assuming LeRobot keys)
        img_key = self.img_modalities[0] if self.img_modalities else "observation.images.image"
        if img_key not in batch and "observation.images.right_cam" in batch:
            img_key = "observation.images.right_cam"
            
        image_tensor = batch[img_key]
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor.unsqueeze(1)
        B, T, C, H, W = image_tensor.shape
        
        image_features = self.vlm._encode_image(
            image_tensor.view(-1, C, H, W).to(device).to(default_dtype)
        )
        image_features = image_features.view(B, T * image_features.shape[1], -1)

        # Second view
        if self.use_second_view and self.second_view_key in batch:
            img2 = batch[self.second_view_key]
            if len(img2.shape) == 4: img2 = img2.unsqueeze(1)
            feat2 = self.vlm._encode_image(img2.view(-1, C, H, W).to(device).to(default_dtype))
            feat2 = feat2.view(B, T * feat2.shape[1], -1)
            image_features = torch.cat([image_features, feat2], dim=1)

        # Text encoding
        txt = batch.get("text", batch.get("task", [""] * B))
        if not isinstance(txt, list): txt = [txt] * B if isinstance(txt, str) else [""] * B
        
        # Use prompt style
        prompts = [
            f"<od>{t}</od>" if self.vlm_prompt_style != "default" else t 
            for t in txt
        ]
        
        tokens = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(device)
        
        text_embeds = self.vlm.get_input_embeddings()(tokens["input_ids"])
        
        # Combine
        task_prompt = self.prompt_embeds.expand(B, -1, -1)
        merged = torch.cat([task_prompt, image_features, text_embeds], dim=1)
        
        # Masks
        vis_mask = torch.ones(image_features.shape[:2], device=device)
        prompt_mask = torch.ones(B, 1, dtype=torch.long, device=device)
        attn_mask = torch.cat([prompt_mask, vis_mask, tokens["attention_mask"]], dim=1)

        features = self.vlm.get_encoder()(
            inputs_embeds=merged, attention_mask=attn_mask
        ).last_hidden_state
        
        return {"features": features, "attention_mask": attn_mask}

    def sample_actions(self, z, cond, inference=False):
        features = cond["features"]
        mask = cond["attention_mask"]
        B = features.shape[0]
        
        # 1. Construct Filler Input
        # We need (NumDOF * NumBasis) tokens
        seq_len_tokens = self.action_tokenizer.num_dof * self.action_tokenizer.num_basis
        filler_bin = torch.full((B, seq_len_tokens), self.action_bins // 2, dtype=torch.long, device=self.device)
        llm_input_ids = self._bins_to_llm_ids(filler_bin)
        
        # 2. Bidirectional Mask
        bidirectional_mask = create_bidirectional_mask(B, seq_len_tokens, self.device)
        
        # 3. Decode
        decoder_outputs = self.vlm.get_decoder()(
            input_ids=llm_input_ids,
            encoder_hidden_states=features,
            encoder_attention_mask=mask,
            attention_mask=bidirectional_mask,
            use_cache=False,
        )
        
        lm_logits = self.vlm.language_model.get_output_embeddings()(decoder_outputs[0])
        lm_logits = lm_logits + self.vlm.language_model.final_logits_bias.to(lm_logits.device)
        
        # 4. Reconstruct
        pred_ids = torch.argmax(lm_logits, dim=-1)
        pred_bins = self._llm_ids_to_bins(pred_ids)
        
        # Use init_pos relative reconstruction if needed (logic from original beast_florence)
        # beast.py decode_discrete accepts init_pos
        init_pos = None
        if self.pred_action_seq is not None and self.action_tokenizer.enforce_init_pos:
            # Use last action of previous chunk as start of next
            init_pos = self.pred_action_seq[:, -1, ...]
            
        actions = self.action_tokenizer.decode_discrete(pred_bins, init_pos=init_pos)
        
        return actions

    def reset(self):
        self.rollout_step_counter = 0
        self.pred_action_seq = None
        self.eval()