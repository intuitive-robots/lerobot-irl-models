import logging

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor.pipeline import PolicyProcessorPipeline

log = logging.getLogger(__name__)


def sanity_check_eval(
    agent: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline | None = None,
    postprocessor: PolicyProcessorPipeline | None = None,
    dataset: LeRobotDataset | None = None,
    max_episodes: int = 5,
) -> list[dict]:
    """
    Sanity check: Runs inference on the first few dataset episodes,
    calculates per-episode MSE and gripper accuracy, and logs the mean/std.
    The MSE should be similar to the one achieved with the openloop script.
    """
    num_eps = min(max_episodes, len(dataset))
    all_pred_actions = []
    all_gt_actions = []

    # Run Inference and Track Metrics
    for eps_idx in range(num_eps):
        log.info(f"Running inference on episode {eps_idx}")
        from_idx = dataset.meta.episodes[eps_idx]["dataset_from_index"]
        to_idx = dataset.meta.episodes[eps_idx]["dataset_to_index"]

        agent.reset()  # Crucial: Reset policy state (RNN/Diffusion) per episode

        # Iterate through time steps and run inference
        for idx in range(from_idx, to_idx):
            observation = dataset[idx]
            if preprocessor:
                observation = preprocessor(observation)
            with torch.inference_mode():
                pred_action_tensor = agent.select_action(observation)
            if postprocessor:
                pred_action_tensor = (
                    postprocessor(pred_action_tensor).squeeze(0).cpu()
                )  # no batch dim

            gt_action = dataset[idx]["action"][0, :].cpu()  # no batch dim
            all_pred_actions.append(pred_action_tensor)
            all_gt_actions.append(gt_action)

    if not all_pred_actions:
        log.warning("Dataset is empty or no episodes were processed.")
        return []

    pred_actions_all = torch.stack(all_pred_actions).numpy()
    gt_actions_all = torch.stack(all_gt_actions).numpy()
    diff = pred_actions_all - gt_actions_all
    overall_metrics = {
        "overall_mse": np.mean(diff**2),
        # Calculate per-dimension MSE
        "per_dim_mse": np.mean(diff**2, axis=0),
    }

    # Gripper-Specific Metrics (if applicable)
    if pred_actions_all.shape[1] > 6:
        pred_gripper = pred_actions_all[:, -1]
        gt_gripper = gt_actions_all[:, -1]

        # MSE
        gripper_mse = np.mean((pred_gripper - gt_gripper) ** 2)
        overall_metrics["gripper_mse"] = gripper_mse

        # Accuracy (Discrete)
        threshold = 0.0 if gt_gripper.min() < 0 else 0.5
        pred_discrete = (pred_gripper > threshold).astype(int)
        gt_discrete = (gt_gripper > threshold).astype(int)
        gripper_accuracy = np.mean(pred_discrete == gt_discrete)
        overall_metrics["gripper_accuracy"] = gripper_accuracy

    log.info(f"--- Policy Sanity Check Results over {num_eps} Episodes ---")
    for key, value in overall_metrics.items():
        if isinstance(value, np.ndarray):
            log.info(
                f"✨ {key} (Dimensions): {np.array2string(value, formatter={'float_kind':lambda x: f'{x:.6f}'})}"
            )
        else:
            log.info(f"✅ {key}: {value:.6f}")

    return overall_metrics
