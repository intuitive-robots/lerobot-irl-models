import logging
import os
import random
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision.transforms as transforms
import wandb
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from policies.flower.flower_config import FlowerVLAConfig
from policies.flower.modeling_flower import FlowerVLAPolicy

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def set_seed_everywhere(seed: int) -> None:
    """Set python, numpy, and torch (CPU & CUDA) seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_obs_to_device(obs_dict: dict, device: torch.device) -> dict:
    """Recursively move all tensors inside an observation dictionary to the target device."""
    for k, v in obs_dict.items():
        if isinstance(v, dict):
            move_obs_to_device(v, device)
        elif torch.is_tensor(v):
            obs_dict[k] = v.to(device, non_blocking=True)
    return obs_dict


def load_images_from_folder(folder_path: Path) -> list:
    """Load all images from a folder, sorted by frame number."""
    image_files = sorted(
        [f for f in folder_path.glob("*.png")], key=lambda x: int(x.stem)
    )
    return image_files


def prepare_observation(
    right_cam_path: Path,
    wrist_cam_path: Path,
    robot_state: torch.Tensor,
    transform,
    device: torch.device,
) -> dict:
    """Prepare observation dictionary from image paths, mimicking eval_flower.py structure."""
    # Load images
    right_img = Image.open(right_cam_path).convert("RGB")
    wrist_img = Image.open(wrist_cam_path).convert("RGB")

    # Apply transforms
    right_tensor = transform(right_img)
    wrist_tensor = transform(wrist_img)

    # Create observation dict with batch dimension and temporal dimension
    # Based on how eval_flower.py structures observations in real_robot_sim.py
    obs_dict = {
        "observation.images.right_cam": right_tensor.unsqueeze(0)
        .unsqueeze(0)
        .to(device),  # [1, 1, C, H, W]
        "observation.images.wrist_cam": wrist_tensor.unsqueeze(0)
        .unsqueeze(0)
        .to(device),  # [1, 1, C, H, W]
        "observation.state": robot_state.unsqueeze(0)
        .unsqueeze(0)
        .to(device),  # [1, 1, D]
        "task": "Grab the sweet and put it on the hand",
    }

    return obs_dict


# -----------------------------------------------------------------------------
# Main evaluation script (minimal changes from the original train script)
# -----------------------------------------------------------------------------


@hydra.main(
    config_path="../configs", config_name="eval_flower_config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    """Entry point. Loads a trained checkpoint, runs it on the evaluation dataset and
    saves comparison plots of predicted vs. ground‑truth actions.

    Only the absolute essentials have been changed compared to your original
    `finetuning.py` so that you can drop this file in and run it without
    touching the rest of the code base.
    """

    # ---------------------------------------------------------------------
    # 1. Reproducibility & logging
    # ---------------------------------------------------------------------
    set_seed_everywhere(cfg.seed)
    os.makedirs("plots", exist_ok=True)
    log.info("Plots will be saved in %s", os.path.abspath("plots"))

    # ---------------------------------------------------------------------
    # 2. Initialise (WandB in offline mode so it does not register a new run)
    # ---------------------------------------------------------------------
    wandb.init(mode="disabled")

    # ---------------------------------------------------------------------
    # 3. Build the agent *exactly* as during training and load the checkpoint
    # ---------------------------------------------------------------------
    config_dict = OmegaConf.to_container(cfg.policy, resolve=True)
    # Remove _target_ as it's not a parameter for FlowerVLAConfig
    config_dict.pop("_target_", None)
    config = FlowerVLAConfig(**config_dict)
    agent = FlowerVLAPolicy(config)

    # You can point these two Hydra overrides at runtime:
    #   python evaluate_model.py eval.ckpt_dir=/path/to/09-42-10 eval.ckpt_name=checkpoint_50000
    ckpt_dir: Path = Path(OmegaConf.select(cfg, "eval.ckpt_dir", default=""))
    ckpt_name: str = OmegaConf.select(cfg, "eval.ckpt_name", default="")

    from safetensors.torch import load_file

    state_dict = load_file(cfg.checkpoint_path, device=str(cfg.device))
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove common prefixes that might differ between training and inference
        new_key = key
        if key.startswith("agent."):
            new_key = "model." + key[6:]  # Remove 'agent.' and add 'model.'
        elif key.startswith("policy."):
            new_key = "model." + key[7:]  # Remove 'policy.' and add 'model.'
        elif not key.startswith("model."):
            # If no prefix, add 'model.'
            new_key = "model." + key

        # Map MLP layer names
        new_key = new_key.replace(".mlp.c_fc1.", ".mlp.fc1.")
        new_key = new_key.replace(".mlp.c_fc2.", ".mlp.fc2.")
        new_key = new_key.replace(".mlp.c_proj.", ".mlp.proj.")

        new_state_dict[new_key] = value

    log.info(f"Preprocessed {len(new_state_dict)} keys from checkpoint")

    # Load with strict=False to allow partial loading
    missing_keys, unexpected_keys = agent.load_state_dict(new_state_dict, strict=False)

    if missing_keys:
        log.warning(f"Missing keys in checkpoint ({len(missing_keys)} total):")
        log.warning(f"  First few: {missing_keys[:5]}")
        log.warning("  → These parameters will use random initialization!")

    if unexpected_keys:
        log.warning(f"Unexpected keys in checkpoint ({len(unexpected_keys)} total):")
        log.warning(f"  First few: {unexpected_keys[:5]}")
        log.warning("  → These parameters from checkpoint will be ignored!")

    if not missing_keys and not unexpected_keys:
        log.info("✅ All parameters loaded successfully!")
    else:
        log.info("⚠️  Model loaded with warnings (see above)")

    # Move agent to device
    agent = agent.to(cfg.device)
    agent.eval()

    agent.eval()
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------------
    # 4. Find all data directories and setup paths
    # ---------------------------------------------------------------------
    base_data_dir = Path("/hkfs/work/workspace/scratch/uhtfz-flower/trickandtreat_lerobot")
    data_dirs = sorted([d for d in base_data_dir.iterdir() if d.is_dir()])
    log.info(f"Found {len(data_dirs)} data directories to process")

    # Storage for metrics across all episodes
    all_episode_metrics = []

    # Setup image transform (adjust based on your model's requirements)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Adjust size as needed
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Get device from config
    device = torch.device(cfg.device)

    # ---------------------------------------------------------------------
    # 5. Process each episode
    # ---------------------------------------------------------------------
    for episode_idx, data_root in enumerate(data_dirs):
        log.info(f"\n{'='*80}")
        log.info(
            f"Processing episode {episode_idx + 1}/{len(data_dirs)}: {data_root.name}"
        )
        log.info(f"{'='*80}")

        try:
            # Setup paths for this episode
            right_cam_dir = data_root / "sensors" / "right_cam"
            wrist_cam_dir = data_root / "sensors" / "wrist_cam"
            gt_actions_path = data_root / "201 leader" / "joint_pos.pt"
            gt_gripper_path = data_root / "201 leader" / "gripper_state.pt"
            robot_states_path = data_root / "202 follower" / "joint_pos.pt"

            # Check if all required files exist
            if not all(
                [
                    right_cam_dir.exists(),
                    wrist_cam_dir.exists(),
                    gt_actions_path.exists(),
                    gt_gripper_path.exists(),
                    robot_states_path.exists(),
                ]
            ):
                log.warning(f"Skipping {data_root.name}: missing required files")
                continue

            # Load data for this episode
            gt_actions = torch.load(gt_actions_path)
            gt_gripper = torch.load(gt_gripper_path)
            robot_states = torch.load(robot_states_path)

            # Get image files
            right_cam_files = load_images_from_folder(right_cam_dir)
            wrist_cam_files = load_images_from_folder(wrist_cam_dir)

            # Ensure we have matching number of images, robot states and actions
            num_samples = min(
                len(right_cam_files),
                len(wrist_cam_files),
                robot_states.shape[0],
                gt_actions.shape[0],
                gt_gripper.shape[0],
            )

            if num_samples == 0:
                log.warning(f"Skipping {data_root.name}: no valid samples")
                continue

            log.info(f"Processing {num_samples} samples from this episode")

            # Run inference on all images in this episode
            predicted_actions = []
            predicted_gripper = []
            ground_truth_actions = []

            for idx in tqdm(range(num_samples), desc=f"Episode {episode_idx + 1}"):
                # Prepare observation with robot state
                obs_dict = prepare_observation(
                    right_cam_files[idx],
                    wrist_cam_files[idx],
                    robot_states[idx],  # Current robot state
                    transform,
                    device,
                )

                # Get prediction using select_action method (as in real_robot_sim.py)
                with torch.no_grad():
                    pred_action = agent.select_action(
                        obs_dict
                    )  # Shape: [1, D] where D=8 (7 joints + 1 gripper)

                    # Extract joint positions (first 7 dimensions)
                    pred_joint_pos = pred_action[:, :7]  # [1, 7]
                    pred_gripper_val = pred_action[:, 7:]  # [1, 1]

                    predicted_actions.append(pred_joint_pos.cpu())
                    predicted_gripper.append(pred_gripper_val.cpu())

                # Store ground truth (first 7 dimensions for joint positions)
                ground_truth_actions.append(
                    gt_actions[idx : idx + 1, :7]
                )  # Keep as [1, 7]

            # Stack all predictions and ground truth for this episode
            predicted_actions = torch.cat(predicted_actions, dim=0)  # [N, 7]
            predicted_gripper = torch.cat(predicted_gripper, dim=0)  # [N, 1]
            ground_truth_actions = torch.cat(ground_truth_actions, dim=0)  # [N, 7]
            ground_truth_gripper = gt_gripper[:num_samples]  # [N]

            # Compute metrics for this episode
            pred_np = predicted_actions.numpy()
            gt_np = ground_truth_actions.numpy()
            gripper_np = predicted_gripper.numpy().flatten()
            gripper_gt_np = ground_truth_gripper.numpy().flatten()

            # Overall MSE
            overall_mse = np.mean((pred_np - gt_np) ** 2)

            # Per-dimension MSE
            per_dim_mse = np.mean((pred_np - gt_np) ** 2, axis=0)

            # Gripper metrics
            pred_discrete = np.where(gripper_np > 0, 1, -1)
            gripper_accuracy = (pred_discrete == gripper_gt_np).mean()
            gripper_mse = np.mean((gripper_np - gripper_gt_np) ** 2)

            # Store metrics for this episode
            episode_metrics = {
                "episode_name": data_root.name,
                "num_samples": num_samples,
                "overall_mse": overall_mse,
                "per_dim_mse": per_dim_mse,
                "gripper_accuracy": gripper_accuracy,
                "gripper_mse": gripper_mse,
                "predictions": pred_np,
                "ground_truth": gt_np,
                "gripper_predictions": gripper_np,
                "gripper_ground_truth": gripper_gt_np,
            }
            all_episode_metrics.append(episode_metrics)

            log.info(f"Episode {data_root.name} - Overall MSE: {overall_mse:.6f}")
            log.info(
                f"Episode {data_root.name} - Gripper Accuracy: {gripper_accuracy*100:.2f}%"
            )

        except Exception as e:
            log.error(f"Error processing {data_root.name}: {str(e)}")
            continue

    # ---------------------------------------------------------------------
    # 6. Compute aggregate statistics across all episodes
    # ---------------------------------------------------------------------
    if not all_episode_metrics:
        log.error("No episodes were successfully processed!")
        return

    log.info(f"\n{'='*80}")
    log.info(f"AGGREGATE STATISTICS ACROSS {len(all_episode_metrics)} EPISODES")
    log.info(f"{'='*80}")

    # Compute mean and std of metrics
    overall_mses = [m["overall_mse"] for m in all_episode_metrics]
    gripper_accuracies = [m["gripper_accuracy"] for m in all_episode_metrics]
    gripper_mses = [m["gripper_mse"] for m in all_episode_metrics]

    # Per-dimension statistics
    num_dims = all_episode_metrics[0]["per_dim_mse"].shape[0]
    per_dim_mses_all = np.array([m["per_dim_mse"] for m in all_episode_metrics])

    log.info(f"\nOverall MSE: {np.mean(overall_mses):.6f} ± {np.std(overall_mses):.6f}")
    log.info(
        f"Gripper Accuracy: {np.mean(gripper_accuracies)*100:.2f}% ± {np.std(gripper_accuracies)*100:.2f}%"
    )
    log.info(f"Gripper MSE: {np.mean(gripper_mses):.6f} ± {np.std(gripper_mses):.6f}")

    log.info("\nPer-dimension MSE (mean ± std):")
    for dim in range(num_dims):
        dim_mses = per_dim_mses_all[:, dim]
        log.info(f"  Dimension {dim}: {np.mean(dim_mses):.6f} ± {np.std(dim_mses):.6f}")

    # ---------------------------------------------------------------------
    # 7. Save aggregate results
    # ---------------------------------------------------------------------
    results_path = Path("plots") / "all_predictions.pt"
    torch.save(
        {
            "episode_metrics": all_episode_metrics,
            "aggregate_stats": {
                "overall_mse_mean": np.mean(overall_mses),
                "overall_mse_std": np.std(overall_mses),
                "per_dim_mse_mean": np.mean(per_dim_mses_all, axis=0),
                "per_dim_mse_std": np.std(per_dim_mses_all, axis=0),
                "gripper_accuracy_mean": np.mean(gripper_accuracies),
                "gripper_accuracy_std": np.std(gripper_accuracies),
                "gripper_mse_mean": np.mean(gripper_mses),
                "gripper_mse_std": np.std(gripper_mses),
            },
        },
        results_path,
    )
    log.info(f"Saved aggregate results to {results_path}")

    # ---------------------------------------------------------------------
    # 8. Create aggregate visualizations
    # ---------------------------------------------------------------------

    # Plot 1: MSE distribution across episodes
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Overall MSE across episodes
    axes[0, 0].bar(range(len(overall_mses)), overall_mses, alpha=0.7, color="steelblue")
    axes[0, 0].axhline(
        y=np.mean(overall_mses),
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(overall_mses):.6f}",
    )
    axes[0, 0].set_title("Overall MSE per Episode")
    axes[0, 0].set_xlabel("Episode Index")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Gripper accuracy across episodes
    axes[0, 1].bar(
        range(len(gripper_accuracies)),
        [acc * 100 for acc in gripper_accuracies],
        alpha=0.7,
        color="green",
    )
    axes[0, 1].axhline(
        y=np.mean(gripper_accuracies) * 100,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(gripper_accuracies)*100:.2f}%",
    )
    axes[0, 1].set_title("Gripper Accuracy per Episode")
    axes[0, 1].set_xlabel("Episode Index")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_ylim([0, 105])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Per-dimension MSE boxplot
    axes[1, 0].boxplot(
        [per_dim_mses_all[:, i] for i in range(num_dims)],
        tick_labels=[f"Dim {i}" for i in range(num_dims)],
    )
    axes[1, 0].set_title("MSE Distribution per Dimension (across all episodes)")
    axes[1, 0].set_ylabel("MSE")
    axes[1, 0].grid(True, alpha=0.3)

    # Per-dimension mean MSE with error bars
    dim_means = np.mean(per_dim_mses_all, axis=0)
    dim_stds = np.std(per_dim_mses_all, axis=0)
    axes[1, 1].bar(
        range(num_dims), dim_means, yerr=dim_stds, alpha=0.7, color="coral", capsize=5
    )
    axes[1, 1].set_title("Mean MSE per Dimension with Std Dev")
    axes[1, 1].set_xlabel("Dimension")
    axes[1, 1].set_ylabel("MSE")
    axes[1, 1].set_xticks(range(num_dims))
    axes[1, 1].set_xticklabels([f"Dim {i}" for i in range(num_dims)])
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/aggregate_metrics_summary.png", dpi=150)
    plt.close()
    log.info("Saved aggregate_metrics_summary.png")

    # Plot 2: Concatenated predictions from a sample of episodes (first 5)
    sample_episodes = min(5, len(all_episode_metrics))
    fig, axes = plt.subplots(num_dims, 1, figsize=(20, 3 * num_dims))
    if num_dims == 1:
        axes = [axes]

    for dim in range(num_dims):
        offset = 0
        for ep_idx in range(sample_episodes):
            ep = all_episode_metrics[ep_idx]
            n_samples = ep["predictions"].shape[0]
            time_range = range(offset, offset + n_samples)

            # Plot ground truth and predictions for this episode
            axes[dim].plot(
                time_range,
                ep["ground_truth"][:, dim],
                linewidth=1.5,
                alpha=0.7,
                color="blue",
            )
            axes[dim].plot(
                time_range,
                ep["predictions"][:, dim],
                linewidth=1.5,
                alpha=0.7,
                linestyle="--",
                color="orange",
            )

            # Add vertical line to separate episodes
            if ep_idx < sample_episodes - 1:
                axes[dim].axvline(
                    x=offset + n_samples,
                    color="gray",
                    linestyle=":",
                    linewidth=2,
                    alpha=0.5,
                )

            offset += n_samples

        axes[dim].set_title(
            f"Dimension {dim} - Concatenated (First {sample_episodes} Episodes)"
        )
        axes[dim].set_xlabel("Time Step")
        axes[dim].set_ylabel("Joint Position")
        axes[dim].grid(True, alpha=0.3)

        # Create custom legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="blue", linewidth=2, label="Ground Truth"),
            Line2D(
                [0], [0], color="orange", linewidth=2, linestyle="--", label="Predicted"
            ),
        ]
        axes[dim].legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig("plots/concatenated_predictions.png", dpi=150)
    plt.close()
    log.info("Saved concatenated_predictions.png")

    # Plot 3: Average prediction curves (mean and std across episodes)
    # Find maximum episode length to align all episodes
    max_length = max([m["predictions"].shape[0] for m in all_episode_metrics])

    # Create arrays to store aligned predictions and ground truth
    aligned_predictions = np.full(
        (len(all_episode_metrics), max_length, num_dims), np.nan
    )
    aligned_ground_truth = np.full(
        (len(all_episode_metrics), max_length, num_dims), np.nan
    )

    for ep_idx, ep in enumerate(all_episode_metrics):
        n = ep["predictions"].shape[0]
        aligned_predictions[ep_idx, :n, :] = ep["predictions"]
        aligned_ground_truth[ep_idx, :n, :] = ep["ground_truth"]

    fig, axes = plt.subplots(num_dims, 1, figsize=(15, 3 * num_dims))
    if num_dims == 1:
        axes = [axes]

    for dim in range(num_dims):
        # Compute mean and std (ignoring NaN values)
        pred_mean = np.nanmean(aligned_predictions[:, :, dim], axis=0)
        pred_std = np.nanstd(aligned_predictions[:, :, dim], axis=0)
        gt_mean = np.nanmean(aligned_ground_truth[:, :, dim], axis=0)
        gt_std = np.nanstd(aligned_ground_truth[:, :, dim], axis=0)

        time_steps = np.arange(max_length)

        # Plot ground truth with confidence interval
        axes[dim].plot(time_steps, gt_mean, color="blue", linewidth=2, label="GT Mean")
        axes[dim].fill_between(
            time_steps,
            gt_mean - gt_std,
            gt_mean + gt_std,
            color="blue",
            alpha=0.2,
            label="GT Std",
        )

        # Plot predictions with confidence interval
        axes[dim].plot(
            time_steps,
            pred_mean,
            color="orange",
            linewidth=2,
            linestyle="--",
            label="Pred Mean",
        )
        axes[dim].fill_between(
            time_steps,
            pred_mean - pred_std,
            pred_mean + pred_std,
            color="orange",
            alpha=0.2,
            label="Pred Std",
        )

        axes[dim].set_title(
            f"Dimension {dim} - Average Trajectory (across {len(all_episode_metrics)} episodes)"
        )
        axes[dim].set_xlabel("Time Step")
        axes[dim].set_ylabel("Joint Position")
        axes[dim].legend()
        axes[dim].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/average_trajectories.png", dpi=150)
    plt.close()
    log.info("Saved average_trajectories.png")

    # Plot 4: Statistical summary table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("tight")
    ax.axis("off")

    # Create table data
    table_data = [
        ["Metric", "Mean", "Std Dev", "Min", "Max"],
        [
            "Overall MSE",
            f"{np.mean(overall_mses):.6f}",
            f"{np.std(overall_mses):.6f}",
            f"{np.min(overall_mses):.6f}",
            f"{np.max(overall_mses):.6f}",
        ],
        [
            "Gripper Acc (%)",
            f"{np.mean(gripper_accuracies)*100:.2f}",
            f"{np.std(gripper_accuracies)*100:.2f}",
            f"{np.min(gripper_accuracies)*100:.2f}",
            f"{np.max(gripper_accuracies)*100:.2f}",
        ],
        [
            "Gripper MSE",
            f"{np.mean(gripper_mses):.6f}",
            f"{np.std(gripper_mses):.6f}",
            f"{np.min(gripper_mses):.6f}",
            f"{np.max(gripper_mses):.6f}",
        ],
    ]

    # Add per-dimension data
    for dim in range(num_dims):
        dim_mses = per_dim_mses_all[:, dim]
        table_data.append(
            [
                f"Dim {dim} MSE",
                f"{np.mean(dim_mses):.6f}",
                f"{np.std(dim_mses):.6f}",
                f"{np.min(dim_mses):.6f}",
                f"{np.max(dim_mses):.6f}",
            ]
        )

    table = ax.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.2, 0.2, 0.2, 0.2, 0.2],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f0f0")

    plt.title(
        f"Statistical Summary Across {len(all_episode_metrics)} Episodes",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.savefig("plots/statistical_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved statistical_summary.png")

    # Plot 5: Episode-wise comparison heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(all_episode_metrics) * 0.4)))

    # Heatmap for per-dimension MSE
    sns.heatmap(
        per_dim_mses_all,
        annot=False,
        cmap="YlOrRd",
        ax=axes[0],
        xticklabels=[f"Dim {i}" for i in range(num_dims)],
        yticklabels=[f"Ep {i}" for i in range(len(all_episode_metrics))],
        cbar_kws={"label": "MSE"},
    )
    axes[0].set_title("Per-Dimension MSE Heatmap (by Episode)")
    axes[0].set_xlabel("Dimension")
    axes[0].set_ylabel("Episode")

    # Combined metrics heatmap
    combined_metrics = np.column_stack([overall_mses, gripper_accuracies, gripper_mses])

    # Normalize each column to [0, 1] for better visualization
    combined_metrics_norm = (combined_metrics - combined_metrics.min(axis=0)) / (
        combined_metrics.max(axis=0) - combined_metrics.min(axis=0) + 1e-8
    )

    sns.heatmap(
        combined_metrics_norm,
        annot=False,
        cmap="coolwarm",
        ax=axes[1],
        xticklabels=["Overall MSE", "Gripper Acc", "Gripper MSE"],
        yticklabels=[f"Ep {i}" for i in range(len(all_episode_metrics))],
        cbar_kws={"label": "Normalized Value"},
    )
    axes[1].set_title("Overall Metrics Heatmap (Normalized, by Episode)")
    axes[1].set_xlabel("Metric")
    axes[1].set_ylabel("Episode")

    plt.tight_layout()
    plt.savefig("plots/metrics_heatmap.png", dpi=150)
    plt.close()
    log.info("Saved metrics_heatmap.png")

    log.info(f"\n{'='*80}")
    log.info(f"EVALUATION COMPLETE")
    log.info(f"{'='*80}")
    log.info(f"Processed {len(all_episode_metrics)} episodes")
    log.info(
        f"Average Overall MSE: {np.mean(overall_mses):.6f} ± {np.std(overall_mses):.6f}"
    )
    log.info(
        f"Average Gripper Accuracy: {np.mean(gripper_accuracies)*100:.2f}% ± {np.std(gripper_accuracies)*100:.2f}%"
    )
    log.info(f"All plots saved in {os.path.abspath('plots')}")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
