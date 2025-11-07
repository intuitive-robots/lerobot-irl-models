from collections import deque
from collections.abc import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from lerobot.processor.normalize_processor import (
    NormalizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from torch import Tensor, nn
from transformers import CLIPModel, CLIPProcessor

from .beso_config import BesoConfig
from .beso_transformer import Noise_Dec_only
from .utils import append_dims, get_sigmas_exponential, make_sample_density, sample_ddim


class BesoPolicy(PreTrainedPolicy):
    """
    Adapted from Lerobot implementation of Diffusion Policy.
    """

    config_class = BesoConfig
    name = "beso"

    def __init__(
        self,
        config: BesoConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to load_state_dict before the policy is used.
        """
        super().__init__(config)
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
        self.unnormalize_inputs = UnnormalizerProcessorStep(
            config.input_features, config.normalization_mapping, dataset_stats
        )
        self.step_counter = 0
        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None
        # self.
        self.diffusion = BesoModel(config)

    def get_optim_params(self) -> dict:
        return self.diffusion.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on env.reset()"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }

        if self.config.image_features:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(
                maxlen=self.config.n_obs_steps
            )

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        batch = {
            k: torch.stack(list(self._queues[k]), dim=1)
            for k in batch
            if k in self._queues
        }
        actions = self.diffusion.generate_actions(batch)
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        # in select_action(...)

        if ACTION in batch:
            batch.pop(ACTION)
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(
                batch
            )  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        # NOTE: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        self.step_counter += 1
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(
                batch
            )  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )

        batch = self.normalize_targets(batch)
        loss = self.diffusion.compute_loss(batch)
        # no output_dict so returning None
        return loss, None


class BesoModel(nn.Module):
    def __init__(self, config: BesoConfig):
        super().__init__()
        self.config = config
        self.sigma_data = config.sigma_data
        self.sigma_max = config.sigma_max
        self.sigma_min = config.sigma_min
        self.act_seq_len = config.horizon
        self.sampling_steps = config.sampling_steps
        # Build observation encoders (depending on which observations are provided).
        global_cond_dim = self.config.robot_state_feature.shape[0]

        # Use either CLIP or ResNet, not both
        if self.config.image_features:
            num_images = len(self.config.image_features)

            if self.config.use_clip_encoder:
                # Use CLIP encoder
                if self.config.use_separate_rgb_encoder_per_camera:
                    encoders = [BesoClipEncoder(config) for _ in range(num_images)]
                    self.rgb_encoder = nn.ModuleList(encoders)
                    global_cond_dim += encoders[0].feature_dim * num_images
                else:
                    self.rgb_encoder = BesoClipEncoder(config)
                    global_cond_dim += self.rgb_encoder.feature_dim * num_images
            else:
                # Use ResNet encoder
                if self.config.use_separate_rgb_encoder_per_camera:
                    encoders = [BesoRgbEncoder(config) for _ in range(num_images)]
                    self.rgb_encoder = nn.ModuleList(encoders)
                    global_cond_dim += encoders[0].feature_dim * num_images
                else:
                    self.rgb_encoder = BesoRgbEncoder(config)
                    global_cond_dim += self.rgb_encoder.feature_dim * num_images

        # Add language conditioning dimension if enabled
        if self.config.use_language_conditioning:
            global_cond_dim += self.config.language_feature_dim

        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        self.dit_backbone = Noise_Dec_only(
            state_dim=global_cond_dim * 2,
            action_dim=self.config.action_feature.shape[0],
            goal_dim=0,
            device="cuda",  # Default device, will be moved to correct device automatically by PyTorch
            goal_conditioned=False,
            embed_dim=config.embed_dim,
            embed_pdrob=0,
            goal_seq_len=0,
            obs_seq_len=self.config.n_obs_steps,
            action_seq_len=self.config.horizon,
            # linear_output=False,
            use_ada_conditioning=False,
            diffusion_type="beso",  # ddpm, beso or rf,
            use_pos_emb=False,
        )
        self.device = "cuda"

        # Print parameter counts
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        print("BESO Model Parameter Count:")
        print("=" * 40)

        if hasattr(self, "rgb_encoder"):
            encoder_type = "CLIP" if self.config.use_clip_encoder else "ResNet"
            if isinstance(self.rgb_encoder, nn.ModuleList):
                total_encoder_params = sum(
                    count_params(enc) for enc in self.rgb_encoder
                )
                single_encoder_params = count_params(self.rgb_encoder[0])
                print(f"{encoder_type} Encoders (total): {total_encoder_params:,}")
                print(f"Single {encoder_type} Encoder: {single_encoder_params:,}")
            else:
                encoder_params = count_params(self.rgb_encoder)
                print(f"{encoder_type} Encoder: {encoder_params:,}")

        backbone_params = count_params(self.dit_backbone)
        print(f"Transformer Backbone: {backbone_params:,}")

        total_params = count_params(self)
        print(f"Total Parameters: {total_params:,}")

        if self.config.use_language_conditioning:
            print(f"Language Conditioning: Enabled")

        print("=" * 40)

    # ========= inference  ============
    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Sample prior.
        actions = (
            torch.randn(
                size=(
                    batch_size,
                    self.config.horizon,
                    self.config.action_feature.shape[0],
                ),
                # size=(batch_size, 8, self.config.action_feature.shape[0]),
                dtype=dtype,
                device=device,
                generator=generator,
            )
            * self.sigma_max
        )
        input_state = global_cond
        sigmas = get_sigmas_exponential(
            self.config.sampling_steps,
            self.config.sigma_min,
            self.config.sigma_max,
            device,
        )
        actions = sample_ddim(self, input_state, actions, None, sigmas)

        return actions

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]

        # 1) state as-is (FIX)
        state_feats = torch.zeros_like(batch[OBS_STATE])  # (B, S, state_dim)

        global_cond_feats = [state_feats]

        # 2) images -> encoder -> concat cameras
        img_features = None
        if self.config.image_features:
            # Use single encoder (either CLIP or ResNet)
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(
                    batch["observation.images"], "b s n ... -> n (b s) ..."
                )
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(
                            self.rgb_encoder, images_per_camera, strict=True
                        )
                    ]
                )
                img_features = einops.rearrange(
                    img_features_list,
                    "(n b s) ... -> b s (n ...)",
                    b=batch_size,
                    s=n_obs_steps,
                )
            else:
                num_cameras = len(self.config.image_features)
                shared = self.rgb_encoder(
                    einops.rearrange(
                        batch["observation.images"], "b s n ... -> (b s n) ..."
                    )
                )
                img_features = einops.rearrange(
                    shared,
                    "(b s n) d -> b s (n d)",
                    b=batch_size,
                    s=n_obs_steps,
                    n=num_cameras,
                )
            global_cond_feats.append(img_features)

        # 3) language conditioning (if enabled and provided in batch)
        if self.config.use_language_conditioning and "language" in batch:
            # If using CLIP encoder, it can also encode language
            if self.config.use_clip_encoder and hasattr(
                self.rgb_encoder, "encode_text"
            ):
                language_features = self.rgb_encoder.encode_text(batch["language"])
            else:
                # Otherwise, language should be pre-encoded or use a separate encoder
                language_features = batch["language"]

            # Expand language features to match observation sequence length
            # (B, lang_dim) -> (B, S, lang_dim)
            if len(language_features.shape) == 2:
                language_features = language_features.unsqueeze(1).expand(
                    -1, n_obs_steps, -1
                )

            global_cond_feats.append(language_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        feats = torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1).unsqueeze(1)

        return feats

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps
        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)
        # run sampling
        actions = self.conditional_sample(batch_size, global_cond=global_cond)
        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        # Input validation.
        assert set(batch).issuperset({"observation.state", "action", "action_is_pad"})
        assert "observation.images" in batch or "observation.environment_state" in batch
        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # Forward diffusion.
        trajectory = batch["action"]

        # Sample noise to add to the trajectory.
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        sigmas = make_sample_density(
            self.config.sigma_sample_density_type,
            self.config.sigma_max,
            self.config.sigma_min,
        )(shape=(len(trajectory),), device=self.device).to(self.device)
        # ->

        c_skip, c_out, c_in = [
            append_dims(x, trajectory.ndim) for x in self.get_scalings(sigmas)
        ]
        noised_input = trajectory + noise * append_dims(sigmas, trajectory.ndim)
        model_output = self.dit_backbone(global_cond, noised_input * c_in, None, sigmas)
        target = (trajectory - c_skip * noised_input) / c_out

        loss = F.mse_loss(model_output, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)
        loss = loss.mean()
        return loss

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def forward(self, state, action, goal, sigma):
        c_skip, c_out, c_in = [
            append_dims(x, action.ndim) for x in self.get_scalings(sigma)
        ]
        return (
            self.dit_backbone(state, action * c_in, goal, sigma) * c_out
            + action * c_skip
        )


class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    (https://huggingface.co/papers/1509.06113). A minimal port of the robomimic implementation.

    At a high level, this takes 2D feature maps (from a convnet/ViT) and returns the "center of mass"
    of activations of each channel, i.e., keypoints in the image space for the policy to focus on.

    Example: take feature maps of size (512x10x12). We generate a grid of normalized coordinates (10x12x2):
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    This is achieved by applying channel-wise softmax over the activations (512x120) and computing the dot
    product with the coordinates (120x2) to get expected points of maximal activation (512x2).

    The example above results in 512 keypoints (corresponding to the 512 input channels). We can optionally
    provide num_kp != None to control the number of keypoints. This is achieved by a first applying a learnable
    linear mapping (in_channels, H, W) -> (num_kp, H, W).
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class BesoRgbEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, config: BesoConfig):
        super().__init__()
        # Set up optional preprocessing.
        if config.crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(
                    config.crop_shape
                )
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Set up backbone.
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16, num_channels=x.num_features
                ),
            )

        # Note: we have a check in the config class to make sure all images have the same shape.
        images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = (
            config.crop_shape if config.crop_shape is not None else images_shape[1:]
        )
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]
        self.pool = SpatialSoftmax(
            feature_map_shape, num_kp=config.spatial_softmax_num_keypoints
        )
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # If we're going to crop, first up/downscale to 256x256 so the crop has enough context.
        if self.do_crop:
            # Bilinear resize on a batch tensor; preserves value range [0,1].
            x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

            if self.training:  # noqa: SIM108
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)

        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer with non-linearity.
        x = self.relu(self.out(x))
        return x


class BesoClipEncoder(nn.Module):
    """Encodes RGB images and optionally text into feature vectors using CLIP.

    This encoder uses the full CLIP model which supports both vision and text encoding.
    """

    def __init__(self, config: BesoConfig):
        super().__init__()
        self.config = config

        # Load full CLIP model (supports both vision and text)
        self.clip_model = CLIPModel.from_pretrained(config.clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(config.clip_model_name)

        # Freeze CLIP weights if specified
        if config.freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # CLIP outputs config.clip_feature_dim dimensional features
        self.feature_dim = config.clip_feature_dim

        # Optional projection layer for vision features
        self.vision_projection = nn.Sequential(
            nn.Linear(config.clip_feature_dim, config.clip_feature_dim),
            nn.ReLU(),
            nn.Linear(config.clip_feature_dim, self.feature_dim),
        )

        # Optional projection layer for text features (if language conditioning is used)
        if config.use_language_conditioning:
            self.text_projection = nn.Sequential(
                nn.Linear(config.clip_feature_dim, config.language_feature_dim),
                nn.ReLU(),
                nn.Linear(config.language_feature_dim, config.language_feature_dim),
            )

        # Store image preprocessing parameters
        self.do_crop = config.crop_shape is not None
        if self.do_crop:
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(
                    config.crop_shape
                )
            else:
                self.maybe_random_crop = self.center_crop

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode images using CLIP vision encoder.

        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Optional cropping
        if self.do_crop:
            x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)

        # Resize to CLIP's expected input size (224x224 for most CLIP models)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # CLIP preprocessing: normalize with CLIP's mean and std
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(
            1, 3, 1, 1
        )
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(
            1, 3, 1, 1
        )
        x = (x - mean) / std

        # Forward through CLIP vision encoder
        vision_outputs = self.clip_model.vision_model(pixel_values=x)

        # Get the pooled output (CLS token representation)
        image_features = vision_outputs.pooler_output  # (B, feature_dim)

        # Apply projection to match expected feature dimension
        image_features = self.vision_projection(image_features)

        return image_features

    def encode_text(self, text_inputs: list[str] | Tensor) -> Tensor:
        """
        Encode text using CLIP text encoder.

        Args:
            text_inputs: Either a list of strings or tokenized text tensor.
        Returns:
            (B, language_feature_dim) text features.
        """
        if not self.config.use_language_conditioning:
            raise ValueError("Language conditioning is not enabled in config!")

        # Tokenize text if it's a list of strings
        if isinstance(text_inputs, list):
            # Use CLIP processor to tokenize text
            inputs = self.clip_processor(
                text=text_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_language_tokens,
            )
            input_ids = inputs["input_ids"].to(self.clip_model.device)
            attention_mask = inputs["attention_mask"].to(self.clip_model.device)
        else:
            # Assume it's already tokenized
            input_ids = text_inputs
            attention_mask = None

        # Forward through CLIP text encoder
        text_outputs = self.clip_model.text_model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Get the pooled output (EOS token representation)
        text_features = text_outputs.pooler_output  # (B, feature_dim)

        # Apply projection
        text_features = self.text_projection(text_features)

        return text_features


def _replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    Args:
        root_module: The module for which the submodules need to be replaced
        predicate: Takes a module as an argument and must return True if the that module is to be replaced.
        func: Takes a module as an argument and returns a new module to replace it with.
    Returns:
        The root module with its submodules replaced.
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    assert not any(
        predicate(m) for _, m in root_module.named_modules(remove_duplicate=True)
    )
    return root_module
