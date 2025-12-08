#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
"""
LeRobot Open-Loop Evaluation Script.

This script combines the robust policy loading of `lerobot-eval` with the
open-loop logic of calculating MSE against a training dataset.

Usage:
    python eval_open_loop.py \
        --policy.path=outputs/train/my_policy/checkpoints/last/pretrained_model \
        --eval.n_episodes=10 \
        --policy.device=cuda
"""

import logging
import os
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from pprint import pformat

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# --- LeRobot Core Imports ---
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed  # before: from lerobot.utilt.utils
from lerobot.utils.utils import get_safe_torch_device, init_logging
from matplotlib.lines import Line2D
from termcolor import colored
from tqdm import tqdm

from src.policies.flower.flower_config import FlowerVLAConfig
from src.policies.xvla.custom_action_space import CustomJointActionSpace

# Initialize Logger
log = logging.getLogger(__name__)


def report_and_plot_metrics(all_episode_metrics):
    # Extract lists for aggregation
    overall_mses = [m["overall_mse"] for m in all_episode_metrics]
    gripper_accs = [m["gripper_accuracy"] for m in all_episode_metrics]
    gripper_mses = [m["gripper_mse"] for m in all_episode_metrics]
    per_dim_mses_all = np.array([m["per_dim_mse"] for m in all_episode_metrics])

    # Calculate stats
    stats = {
        "mse": (np.mean(overall_mses), np.std(overall_mses)),
        "grip_acc": (np.mean(gripper_accs), np.std(gripper_accs)),
        "grip_mse": (np.mean(gripper_mses), np.std(gripper_mses)),
        "dim_mean": np.mean(per_dim_mses_all, axis=0),
        "dim_std": np.std(per_dim_mses_all, axis=0),
    }
    num_dims = per_dim_mses_all.shape[1]

    # Log Aggregate Stats
    log.info(
        f"\n{'='*40}\nAGGREGATE STATISTICS ({len(all_episode_metrics)} EPISODES)\n{'='*40}"
    )
    log.info(f"Overall MSE:      {stats['mse'][0]:.6f} ± {stats['mse'][1]:.6f}")
    log.info(
        f"Gripper Accuracy: {stats['grip_acc'][0]*100:.2f}% ± {stats['grip_acc'][1]*100:.2f}%"
    )
    log.info(
        f"Gripper MSE:      {stats['grip_mse'][0]:.6f} ± {stats['grip_mse'][1]:.6f}"
    )

    # B. Save Results (.pt)
    save_path = f"plots/{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}"
    os.makedirs(save_path, exist_ok=True)

    torch.save(
        {"episode_metrics": all_episode_metrics, "aggregate_stats": stats},
        f"{save_path}/all_predictions.pt",
    )
    log.info(f"Saved results to {save_path}")

    # C. Generate Plots
    # ---------------------------------------------------------------------

    # Plot 1: Metrics Summary (Bar charts & Boxplots)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1.1 Overall MSE
    axes[0, 0].bar(range(len(overall_mses)), overall_mses, alpha=0.7, color="steelblue")
    axes[0, 0].axhline(
        y=stats["mse"][0],
        color="r",
        linestyle="--",
        label=f"Mean: {stats['mse'][0]:.6f}",
    )
    axes[0, 0].set_title("Overall MSE per Episode")
    axes[0, 0].legend()

    # 1.2 Gripper Accuracy
    axes[0, 1].bar(
        range(len(gripper_accs)),
        [a * 100 for a in gripper_accs],
        alpha=0.7,
        color="green",
    )
    axes[0, 1].axhline(
        y=stats["grip_acc"][0] * 100,
        color="r",
        linestyle="--",
        label=f"Mean: {stats['grip_acc'][0]*100:.2f}%",
    )
    axes[0, 1].set_title("Gripper Accuracy (%)")
    axes[0, 1].set_ylim([0, 105])
    axes[0, 1].legend()

    # 1.3 Dim MSE Boxplot
    axes[1, 0].boxplot(
        [per_dim_mses_all[:, i] for i in range(num_dims)],
        tick_labels=[f"D{i}" for i in range(num_dims)],
    )
    axes[1, 0].set_title("MSE Dist. per Dimension")

    # 1.4 Dim Mean Bar
    axes[1, 1].bar(
        range(num_dims),
        stats["dim_mean"],
        yerr=stats["dim_std"],
        alpha=0.7,
        color="coral",
        capsize=5,
    )
    axes[1, 1].set_xticks(range(num_dims))
    axes[1, 1].set_xticklabels([f"D{i}" for i in range(num_dims)])
    axes[1, 1].set_title("Mean MSE per Dimension")

    plt.tight_layout()
    plt.savefig(f"{save_path}/aggregate_metrics.png", dpi=150)
    plt.close()

    # Plot 2: Concatenated Predictions (Sample first 5)
    n_total = len(all_episode_metrics)
    n_sample = 10

    # Sample 10 random indices (sorted), or take all if fewer than 10
    if n_total <= n_sample:
        selected_indices = list(range(n_total))
    else:
        selected_indices = sorted(random.sample(range(n_total), n_sample))
    fig, axes = plt.subplots(num_dims, 1, figsize=(20, 3 * num_dims), squeeze=False)

    for dim in range(num_dims):
        offset = 0
        ax = axes[dim, 0]
        for j, i in enumerate(selected_indices):
            ep = all_episode_metrics[i]
            n = ep["predictions"].shape[0]
            rng = range(offset, offset + n)
            ax.plot(rng, ep["ground_truth"][:, dim], color="blue", alpha=0.6, lw=1.5)
            ax.plot(
                rng,
                ep["predictions"][:, dim],
                color="orange",
                alpha=0.7,
                ls="--",
                lw=1.5,
            )
            # Add Episode number text to plot
            y_min, y_max = ax.get_ylim()
            y_pos = y_min + (y_max - y_min) * 0.95
            ax.text(
                x=offset,  # X-position is the start of the episode
                y=y_pos,  # Y-position is near the top of the plot
                s=f"Ep: {i}",  # The text string (e.g., "Ep: 42")
                color="purple",
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="left",
            )

            # FIX: Check if the current sampled index (j) is NOT the last one
            if j < len(selected_indices) - 1:
                ax.axvline(x=offset + n, color="gray", ls=":", alpha=0.5)
            offset += n

        ax.set_title(
            f"Dim {dim} -  {len(selected_indices)} Randomly Sampled Episodes Concatenated"
        )
        ax.legend(
            [Line2D([0], [0], color="blue"), Line2D([0], [0], color="orange", ls="--")],
            ["GT", "Pred"],
        )

    plt.tight_layout()
    plt.savefig(f"{save_path}/concatenated_predictions.png", dpi=150)
    plt.close()

    # Plot 3: Average Trajectories (Mean ± Std)
    max_len = max(m["predictions"].shape[0] for m in all_episode_metrics)
    aligned_pred = np.full((len(all_episode_metrics), max_len, num_dims), np.nan)
    aligned_gt = np.full((len(all_episode_metrics), max_len, num_dims), np.nan)

    for i, m in enumerate(all_episode_metrics):
        n = m["predictions"].shape[0]
        aligned_pred[i, :n, :] = m["predictions"]
        aligned_gt[i, :n, :] = m["ground_truth"]

    fig, axes = plt.subplots(num_dims, 1, figsize=(15, 3 * num_dims), squeeze=False)
    t = np.arange(max_len)

    for dim in range(num_dims):
        p_mean, p_std = np.nanmean(aligned_pred[..., dim], 0), np.nanstd(
            aligned_pred[..., dim], 0
        )
        g_mean, g_std = np.nanmean(aligned_gt[..., dim], 0), np.nanstd(
            aligned_gt[..., dim], 0
        )

        ax = axes[dim, 0]
        ax.plot(t, g_mean, "b", label="GT Mean")
        ax.fill_between(t, g_mean - g_std, g_mean + g_std, color="b", alpha=0.2)
        ax.plot(t, p_mean, "orange", ls="--", label="Pred Mean")
        ax.fill_between(t, p_mean - p_std, p_mean + p_std, color="orange", alpha=0.2)
        ax.set_title(f"Dim {dim} - Average Trajectory over {n_total} Episodes")
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{save_path}/average_trajectories.png", dpi=150)
    plt.close()

    # Plot 4: Statistical Summary Table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")
    table_data = [["Metric", "Mean", "Std Dev", "Min", "Max"]]

    def add_row(name, data, scale=1.0):
        return [
            name,
            f"{np.mean(data)*scale:.6f}",
            f"{np.std(data)*scale:.6f}",
            f"{np.min(data)*scale:.6f}",
            f"{np.max(data)*scale:.6f}",
        ]

    table_data.append(add_row("Overall MSE", overall_mses))
    table_data.append(add_row("Gripper Acc (%)", gripper_accs, 100))
    table_data.append(add_row("Gripper MSE", gripper_mses))
    for d in range(num_dims):
        table_data.append(add_row(f"Dim {d} MSE", per_dim_mses_all[:, d]))

    tbl = ax.table(
        cellText=table_data, loc="center", cellLoc="center", colWidths=[0.2] * 5
    )
    tbl.scale(1, 2)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    # Style table
    for (i, j), cell in tbl.get_celld().items():
        if i == 0:
            cell.set_facecolor("#4CAF50")
            cell.set_text_props(weight="bold", color="white")
        elif i % 2 == 0:
            cell.set_facecolor("#f0f0f0")

    plt.title(
        f"Statistics Summary ({len(all_episode_metrics)} Episodes)",
        weight="bold",
        pad=20,
    )
    plt.savefig(f"{save_path}/statistical_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 5: Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(all_episode_metrics) * 0.4)))

    sns.heatmap(
        per_dim_mses_all,
        cmap="YlOrRd",
        ax=axes[0],
        xticklabels=[f"D{i}" for i in range(num_dims)],
        cbar_kws={"label": "MSE"},
    )
    axes[0].set_title("Per-Dimension MSE (by Episode)")
    axes[0].set_ylabel("Episode Index")

    # Normalized combined metrics
    comb = np.column_stack([overall_mses, gripper_accs, gripper_mses])
    comb_norm = (comb - comb.min(0)) / (comb.max(0) - comb.min(0) + 1e-8)

    sns.heatmap(
        comb_norm,
        cmap="coolwarm",
        ax=axes[1],
        xticklabels=["MSE", "Grip Acc", "Grip MSE"],
        cbar_kws={"label": "Norm. Value"},
    )
    axes[1].set_title("Metrics Heatmap (Normalized)")
    axes[1].set_ylabel("Episode Index")

    plt.tight_layout()
    plt.savefig(f"{save_path}/metrics_heatmap.png", dpi=150)
    plt.close()

    log.info(f"All plots saved in: {os.path.abspath(save_path)}")
    torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Main Evaluation Logic
# -----------------------------------------------------------------------------

# @parser.wrap()
def eval_open_loop():
    """
    Executes the Open Loop Evaluation.
    This replaces `eval_main` from the standard script but keeps the setup steps identical.
    """
    init_logging()
    eval_n_episodes = 10  # 30  # 10 #00_000
    pretrained_path = "/home/hk-project-p0024638/usmrd/projects/lerobot-irl-models/output/train/xvla/2025-12-07/17-56-59/model_outputs/checkpoints/020000/pretrained_model"

    # -------------------------------------------------------------------------
    # Step 1: Load Policy Config, Overwrite Selected Values and Set Device & Seed
    # -------------------------------------------------------------------------
    # TODO: Here we directly load train_config.json from the checkpoint directory using TrainPipelineConfig
    # instead of config.json via PreTrainedConfig. We should ensure that both are the same
    train_cfg = TrainPipelineConfig.from_pretrained(
        pretrained_name_or_path=pretrained_path
    )  # , cli_overrides=cli_overrides)
    train_cfg.policy.pretrained_path = Path(pretrained_path)

    # TODO: fix the hardcoding of following values (e.g. root needs to be overwritten,
    # because the training config stores the datapath to the TMPDIR)
    train_cfg.dataset.root = (
        "/hkfs/work/workspace/scratch/usmrd-MemVLA/datasets/lerobot/pepper_only"
    )
    if any(["empty_camera" in key for key in train_cfg.policy.input_features]):
        train_cfg.policy.input_features = {
            "observation.images.image": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 256, 256)
            ),
            "observation.images.image2": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 256, 256)
            ),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        }
        train_cfg.policy.num_views = 2
        train_cfg.policy.empty_camera = 1

    # TODO: Think about not using image transforms
    train_cfg.dataset.image_transforms.enable = False
    train_cfg.policy.action_mode = "auto"  # "custom_joint8"
    logging.info(pformat(asdict(train_cfg)))

    device = get_safe_torch_device(train_cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(train_cfg.seed)

    # -------------------------------------------------------------------------
    # Step 2: Load Dataset
    # -------------------------------------------------------------------------
    log.info(f"Loading Dataset from {train_cfg.dataset.root}")
    dataset = make_dataset(train_cfg)

    # -------------------------------------------------------------------------
    # Step 3: Make Policy (Identical to lerobot-eval)
    # -------------------------------------------------------------------------
    log.info("Loading Policy...")
    if isinstance(train_cfg.policy, FlowerVLAConfig):
        # Need to call FLOWER specific policy maker and preprocessor
        # factory.make_policy = make_flower_policy  # monkey patch custom policy maker to pass pretrained non lerobot models
        # factory.make_pre_post_processors = my_make_pre_post_processors
        # factory.get_policy_class = get_flower
        pass
    policy = make_policy(
        cfg=train_cfg.policy,
        env_cfg=None,
        ds_meta=dataset.meta,
        rename_map=train_cfg.rename_map,
    )
    policy.eval()
    policy.to(device)

    # -------------------------------------------------------------------------
    # Step 4: Make Processors (Identical to lerobot-eval)
    # -------------------------------------------------------------------------
    # This creates the exact normalization/un-normalization pipeline used in training.
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if train_cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {
                    **policy.config.input_features,
                    **policy.config.output_features,
                },
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": train_cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=train_cfg.policy,
        pretrained_path=train_cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    # -------------------------------------------------------------------------
    # Step 5: Evaluation Loop (The Rollout Replacement)
    # -------------------------------------------------------------------------
    # Limit number of episodes if requested
    num_episodes_to_eval = min(eval_n_episodes, dataset.num_episodes)
    all_episode_indices = list(range(dataset.num_episodes))
    sampled_episode_indices = random.sample(all_episode_indices, num_episodes_to_eval)
    log.info(f"Starting Open Loop Evaluation on {dataset.num_episodes} total episodes.")
    log.info(f"Randomly sampling and evaluating {num_episodes_to_eval} episodes.")
    all_episode_metrics = []

    # Iterate through episodes
    for episode_idx in tqdm(sampled_episode_indices, desc="Evaluating Episodes"):

        # Get start/end indices for this specific episode
        from_idx = dataset.meta.episodes[episode_idx]["dataset_from_index"]
        to_idx = dataset.meta.episodes[episode_idx]["dataset_to_index"]

        predicted_actions = []
        ground_truth_actions = []

        # IMPORTANT: Reset policy (for RNN/Diffusion state) at start of episode
        policy.reset()

        # Iterate through time steps in the episode
        for idx in range(from_idx, to_idx):
            observation = preprocessor(dataset[idx])
            with torch.inference_mode():
                action = policy.select_action(observation)
            action_postprocessed = postprocessor(action)

            # Extract result and remove batch dimension
            pred_action = action_postprocessed.squeeze(0).cpu()

            # --- E. Ground Truth ---
            # Get the raw action from the dataset (no batch dim)
            gt_action = dataset[idx]["action"][0, :]

            predicted_actions.append(pred_action)
            ground_truth_actions.append(gt_action)

        # Stack results for this episode
        pred_tensor = torch.stack(predicted_actions).numpy()
        gt_tensor = torch.stack(ground_truth_actions).numpy()

        # --- F. Calculate Metrics (User Logic) ---
        # MSE
        overall_mse = np.mean((pred_tensor - gt_tensor) ** 2)
        per_dim_mse = np.mean((pred_tensor - gt_tensor) ** 2, axis=0)

        # Gripper (Assuming last dimension is gripper)
        # Check if gripper exists (dim > 6 usually for 6DoF arms)
        gripper_accuracy = 0.0
        gripper_mse = 0.0

        if pred_tensor.shape[1] > 6:
            pred_gripper = pred_tensor[:, -1]
            gt_gripper = gt_tensor[:, -1]
            gripper_mse = np.mean((pred_gripper - gt_gripper) ** 2)

            # Discrete accuracy calculation (Thresholding)
            # Assuming -1 to 1 range or 0 to 1
            threshold = 0.0 if gt_gripper.min() < 0 else 0.5
            pred_discrete = np.where(pred_gripper > threshold, 1, 0)
            gt_discrete = np.where(gt_gripper > threshold, 1, 0)
            gripper_accuracy = (pred_discrete == gt_discrete).mean()

        metrics = {
            "episode_idx": episode_idx,
            "overall_mse": overall_mse,
            "per_dim_mse": per_dim_mse,
            "gripper_mse": gripper_mse,
            "gripper_accuracy": gripper_accuracy,
            "predictions": pred_tensor,
            "ground_truth": gt_tensor,
        }
        all_episode_metrics.append(metrics)
    # -------------------------------------------------------------------------
    # Step 6: Aggregate & Report
    # -------------------------------------------------------------------------
    if not all_episode_metrics:
        log.error("No episodes processed.")
    report_and_plot_metrics(all_episode_metrics)


if __name__ == "__main__":
    eval_open_loop()
