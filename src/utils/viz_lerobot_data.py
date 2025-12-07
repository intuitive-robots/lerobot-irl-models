"""
This script can be used to visualize lerobot data stored on the horeka cluster. As the cluster nodes do not
support GUI applications, the GUI output needs to be streamed to a local machine, see: https://www.nhr.kit.edu/userdocs/horeka/visualization/

* First Download TurboVNC on your local machine
* Log in to the cluster and run: start_vnc_desktop -n 1 --ppn 4 -t 00:20:00
* On your local machine start TurboVNC and connect using the information displayed by the start_vnc_desktop command
* Once you sucessfully connected, a remote desktop will open.
* On the remote desktop, run this script
"""

import random
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_dataset_viz import visualize_dataset


def main():
    # --- CONFIGURATION ---
    # Update this to your actual data path!
    DATA_ROOT = Path(
        "/hkfs/work/workspace/scratch/usmrd-MemVLA/datasets/lerobot/pepper_only_initial"
    )
    REPO_ID = "pepper_only_initial"

    # Set to None to visualize ALL episodes. Set to a number (e.g., 5) to sample N episodes.
    LIMIT_EPISODES = 15

    # Set a seed for reproducibility (optional)
    random.seed(42)
    # ---------------------

    # 1. Load metadata first to find out how many episodes exist
    print(f"Scanning dataset at {DATA_ROOT}...")
    # Setting episodes=None ensures we get metadata for all episodes
    meta_dataset = LeRobotDataset(repo_id=REPO_ID, root=DATA_ROOT, episodes=None)
    total_episodes = meta_dataset.num_episodes

    print(f"Found {total_episodes} total episodes.")

    # 2. Determine indices for visualization
    all_indices = list(range(total_episodes))

    if LIMIT_EPISODES is not None and total_episodes > LIMIT_EPISODES:
        print(f"Sampling {LIMIT_EPISODES} random episodes...")
        indices_to_visualize = random.sample(all_indices, LIMIT_EPISODES)
    else:
        # If limit is None or total is less than limit, visualize all
        print("Visualizing all available episodes sequentially.")
        indices_to_visualize = all_indices

    # Sort the list for cleaner output (optional)
    indices_to_visualize.sort()
    print(f"Visualizing indices: {indices_to_visualize}")

    # 3. Loop through the generated indices
    for episode_idx in indices_to_visualize:
        print(f"Processing Episode {episode_idx}...")

        # Initialize a lightweight dataset object just for this specific episode
        ds = LeRobotDataset(
            repo_id=REPO_ID,
            root=DATA_ROOT,
            episodes=[episode_idx],
            video_backend="pyav",
        )

        # 4. Visualize
        visualize_dataset(ds, episode_index=episode_idx, mode="local", save=False)

    print("Done! Check the 'Recordings' panel in the Rerun viewer.")


if __name__ == "__main__":
    main()
