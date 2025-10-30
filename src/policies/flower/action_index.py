from typing import List


class ActionIndex:
    """Registry for managing action spaces with robot type and control mode distinctions."""

    def __init__(self):
        # Define action spaces with their dimensions
        self.action_spaces = {
            "joint_single": 0,  # Single arm joint position control (type 0)
            "eef_delta": 1,  # Single arm end-effector velocity (type 1)
            "bimanual_nav": 2,  # Bimanual with navigation (type 2),
            # 'nav': 3,         # Navigation (type 3)
        }

        self.action_dims = {
            "joint_single": 8,  # Single arm joint position control (type 0)
            "eef_delta": 7,  # Single arm end-effector velocity (type 1)
            "bimanual_nav": 16,  # Bimanual with navigation (type 2),
            # 'nav': 2,         # Navigation (type 3)
        }

        # Create mapping from (robot_type, control_mode, num_arms) to action type
        self.action_space_mapping = {
            ("JOINT_POS", "position", 1): 0,  # end-effector pos-1-arm pos
            ("EEF_POS", "velocity", 1): 1,  # end-effector delta-1-arm
            (
                "JOINT_POS_BIMANUAL_NAV",
                "position",
                2,
            ): 2,  # joint-2-arm pos with navigation
            ("JOINT_POS_BIMANUAL", "position", 2): 2,  # joint-2-arm pos
            ("JOINT_POS_NAV", "position", 1): 0,  # joint-1-arm pos with navigation
            ("EEF_POS_NAV", "velocity", 1): 1,  # end-effector delta with navigation
            # ('NAV', 'position', 1): 3,  # navigation
        }

        # Map datasets to their (robot_type, control_mode, num_arms) configuration
        self.dataset_configs = {
            "bridge_dataset": ("DELTA_EEF", "velocity", 1),
            "kuka": ("JOINT_POS", "position", 1),
            "aloha_pen_uncap_diverse_dataset": ("JOINT_POS_BIMANUAL", "position", 2),
            # Add other dataset mappings...
        }

    def get_action_index(
        self, robot_type: str, control_mode: str, num_arms: int
    ) -> int:
        """Get action type index from robot configuration."""
        if num_arms not in [1, 2]:
            raise ValueError("num_arms must be either 1 or 2")

        index = self.action_space_mapping.get((robot_type, control_mode, num_arms))
        if index is None:
            raise ValueError(
                f"Unsupported combination: {(robot_type, control_mode, num_arms)}"
            )
        return index

    def get_action_dim(self, index: int) -> int:
        """Get action dimension for a given action type index."""
        dims = list(self.action_dims.values())
        return dims[index]

    def get_dataset_action_index(self, dataset_name: str) -> int:
        """Get action type index for a dataset."""
        config = self.dataset_configs.get(dataset_name)
        if config is None:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return self.get_action_index(*config)

    def get_max_action_dim(self) -> int:
        """Get maximum action dimension across all types."""
        return max(self.action_dims.values())

    def get_action_mask(self, action_type: int) -> List[bool]:
        """Get mask for which dimensions are active for this action type."""
        dim = self.get_action_dim(action_type)
        return [True] * dim + [False] * (self.get_max_action_dim() - dim)

    def get_action_name(self, action_idx: int) -> str:
        for name, idx in self.action_spaces.items():
            if idx == action_idx:
                return name
        raise ValueError(f"Invalid action index: {action_idx}")
