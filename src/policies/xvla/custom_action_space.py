##################################
# DO NOT USE IN CURRENT FORM
##################################

import torch
from lerobot.policies.xvla.action_hub import (
    BaseActionSpace,
    _ensure_indices_valid,
    register_action,
)
from torch import nn


@register_action("custom_joint8")
class CustomJointActionSpace(BaseActionSpace):
    """
    Adapts a 7-DoF Franka (7 joints + 1 gripper) to a fixed 20-dim model (xvla pretrianed model).

    Layout in 20-dim vector:
    [J1, J2, J3, J4, J5, J6, J7, Grip,  0,  0, ..., 0]
     |_______ JOINTS _______|    |      |__ PADDING __|
             Indices 0-6       Idx 7      Indices 8-19
    """

    dim_action = 20
    REAL_DIM = 8
    gripper_idx = 7
    GRIPPER_SCALE = 0.1
    JOINTS_SCALE = 1.0

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def _pad_to_model_dim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pad the 8-dim robot action to 20-dim for the model.
        Input:  [B, T, 8]
        Output: [B, T, 20]
        """
        if x is None:
            return None
        if x.size(-1) == self.dim_action:  # if already 20dim, do nothing
            return x

        if x.size(-1) == self.REAL_DIM:
            pad_shape = list(x.shape[:-1]) + [
                self.dim_action - self.REAL_DIM
            ]  # 12 zeros
            pad = x.new_zeros(pad_shape)
            return torch.cat([x, pad], dim=-1)

        if x.size(-1) != self.REAL_DIM:
            raise ValueError(
                f"Expected last dim to be {self.REAL_DIM} or {self.dim_action}, got {x.size(-1)}"
            )

    def _trim_to_real_dim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Slice the 20-dim model output back to 8-dim for the robot.
        Input:  [B, T, 20]
        Output: [B, T, 8]
        """
        return x[..., : self.REAL_DIM]

    def compute_loss(self, pred, target):
        """
        Compute loss ONLY on the valid dimensions (0-7).
        Ignore the padding dimensions (8-19).
        """
        target = self._pad_to_model_dim(target)
        assert (
            pred.shape == target.shape
        ), """Predicted and Ground Truth Action Shape does not equal"""

        # Get joint indices (first 7 values) from padded action vector
        joints_loss = (
            self.mse(pred[:, :, self.gripper_idx], target[:, :, self.gripper_idx])
            * self.JOINTS_SCALE
        )

        # 3. Gripper Loss (Classification / BCE)
        # ---------------------------------------------------------------------
        # THE FIX: Recover the binary state from the normalized target.
        # Since normalized data is centered at 0:
        # Values < 0  --> 0.0 (Open)
        # Values > 0  --> 1.0 (Closed)
        # TODO: Should be validated and fixed directly in processor pipeline (NormalizationProcessor)
        # ---------------------------------------------------------------------
        raw_gripper_target = target[..., self.gripper_idx]

        # Convert -1.2 to 0.0 and +1.2 to 1.0
        binary_gripper_target = (raw_gripper_target > 0).float()

        gripper_loss = (
            self.bce(pred[..., self.gripper_idx], binary_gripper_target)
            * self.GRIPPER_SCALE
        )

        return {
            "joints_loss": joints_loss,
            "gripper_loss": gripper_loss,
        }

    def preprocess(self, proprio, action, mode="train"):
        """Zero-out gripper channels in proprio/action."""
        proprio_m = self._pad_to_model_dim(proprio)
        action_m = self._pad_to_model_dim(action)

        # Zero out gripper using the class attribute
        proprio_m[..., self.gripper_idx] = 0.0
        action_m[..., self.gripper_idx] = 0.0
        return proprio_m, action_m

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid to gripper logits."""
        if action.size(-1) > self.gripper_idx:
            breakpoint()
            action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
            breakpoint()
        return self._trim_to_real_dim(action)
