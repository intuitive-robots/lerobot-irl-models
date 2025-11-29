#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert your current "Robocasa-like" folder structure into a LeRobotDataset.

Assumptions (override via CLI flags):
- Episodes are directories that contain:
    <EPISODE_DIR>/
      "201 leader"/joint_pos.pt
      "201 leader"/gripper_state.pt
      sensors/right_cam/*.png
      sensors/wrist_cam/*.png
      sensors/<tactile_name>/*.png   (optional, multiple)
- Camera images are RGB .png in BGR on disk (cv2), converted to RGB uint8.
- We store raw images (HWC, uint8). No normalization here; training can handle that.
- Action = [7 joint positions at t, 1 gripper] taken from t=1..T-1
  State  = 7 joint positions at t-1 (shape [7])

Output:
- LeRobotDataset with keys:
    observation.images.<cam_name>     (image/video)
    observation.images.<tactile_name> (image/video)  [optional]
    observation.state                 (float32, 7)
    action                            (float32, 8)
- One episode per discovered trajectory.
"""

import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

# ---------- Utils ----------


def _numeric_sort_key(p: Path) -> Tuple[int, str]:
    """Sort by numeric stem if possible, else by name."""
    try:
        return (int(re.sub(r"\D", "", p.stem)), p.name)
    except Exception:
        return (sys.maxsize, p.name)


def _read_png_folder(folder: Path, resize: Optional[Tuple[int, int]]) -> np.ndarray:
    """
    Read a folder of PNGs to an array [T, H, W, 3], dtype=uint8, RGB.
    If resize is provided, it is (width, height).
    """
    if not folder.exists():
        raise FileNotFoundError(f"Missing image folder: {folder}")
    files = sorted([p for p in folder.glob("*.png")], key=_numeric_sort_key)
    if not files:
        raise FileNotFoundError(f"No .png files in {folder}")

    frames = []
    for p in files:
        img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        if resize is not None:
            w, h = resize
            img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb[..., [2, 1, 0]]  # TODO: Remove after ZED cams have been fixed
        frames.append(img_rgb)
    if not frames:
        raise RuntimeError(f"Failed to load any images from {folder}")
    arr = np.stack(frames, axis=0)  # [T, H, W, 3]
    return arr.astype(np.uint8)


def _find_episode_dirs(root: Path, leader_subdir: str) -> List[Path]:
    """
    Find all episode directories containing required files.
    We consider a directory an episode if it contains:
      <episode>/<leader_subdir>/joint_pos.pt and gripper_state.pt
    """
    episodes = []
    for ep in root.rglob("*"):
        if not ep.is_dir():
            continue
        leader_dir = ep / leader_subdir
        if (leader_dir / "joint_pos.pt").exists() and (
            leader_dir / "gripper_state.pt"
        ).exists():
            episodes.append(ep)
    episodes = sorted(episodes)
    return episodes


def _compute_lengths_align(
    joint_pos: torch.Tensor,
    gripper: torch.Tensor,
    cam_frames: Dict[str, np.ndarray],
    tactile_frames: Dict[str, np.ndarray],
) -> int:
    """
    Compute aligned length T such that we can form (T-1) action/state pairs.
    All streams will be trimmed to T (and finally used as T-1).
    """
    lengths = [joint_pos.shape[0], gripper.shape[0]]
    lengths.extend([v.shape[0] for v in cam_frames.values()])
    lengths.extend([v.shape[0] for v in tactile_frames.values()])
    T = min(lengths)
    if T < 2:
        raise ValueError(f"Sequence too short after alignment: T={T}")
    return T


# ---------- Feature spec ----------


def _build_features_spec(
    image_shapes: Dict[str, Tuple[int, int, int]],
    tactile_shapes: Dict[str, Tuple[int, int, int]],
    use_videos: bool,
) -> Dict[str, dict]:
    """
    Build the LeRobot feature spec dictionary.
    image_shapes/tactile_shapes: dict cam_name -> (H, W, C) but LeRobot expects (C, H, W)
    """
    dtype_choice = "video" if use_videos else "image"

    features = {}
    for k, (h, w, c) in image_shapes.items():
        features[f"observation.images.{k.replace('zed_', '')}"] = {
            "dtype": dtype_choice,
            "shape": (c, h, w),
            "names": ["channel", "height", "width"],
        }
    for k, (h, w, c) in tactile_shapes.items():
        features[f"observation.images.{k}"] = {
            "dtype": dtype_choice,
            "shape": (c, h, w),
            "names": ["channel", "height", "width"],
        }

    # State: 7 joints
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": [f"motor_{i}" for i in range(7)]},
    }
    # Action: 7 joints + 1 gripper
    features["action"] = {
        "dtype": "float32",
        "shape": (8,),
        "names": {"motors": [f"motor_{i}" for i in range(7)] + ["gripper"]},
    }
    return features


# ---------- Core conversion ----------


def probe_first_valid_episode(
    episodes: List[Path],
    cams: List[str],
    tactile_names: List[str],
    leader_subdir: str,
    sensors_dirname: str,
    resize: Optional[Tuple[int, int]],
) -> Tuple[Dict[str, Tuple[int, int, int]], Dict[str, Tuple[int, int, int]]]:
    """
    Load the first episode just to discover image shapes for features.
    Returns dicts cam->(H,W,3), tactile->(H,W,3)
    """
    for ep in episodes:
        try:
            cam_shapes = {}
            for cam in cams:
                arr = _read_png_folder(ep / sensors_dirname / cam, resize)
                cam_shapes[cam] = tuple(arr.shape[1:4])  # (H,W,3)
            tact_shapes = {}
            for tname in tactile_names:
                arr = _read_png_folder(ep / sensors_dirname / tname, resize)
                tact_shapes[tname] = tuple(arr.shape[1:4])
            return cam_shapes, tact_shapes
        except Exception:
            continue
    raise RuntimeError(
        "Could not probe any episode for image shapes. Check paths/flags."
    )


def save_episode_to_lerobot(
    lerobot_ds: LeRobotDataset,
    ep_dir: Path,
    cams: List[str],
    tactile_names: List[str],
    leader_subdir: str,
    sensors_dirname: str,
    resize: Optional[Tuple[int, int]],
    task_instruction_mapping: Dict[str, str],  # str = "parent",  # "parent" or "name"
):
    """
    Read one episode and write it into the LeRobot dataset.
    """
    leader = ep_dir / leader_subdir
    sensors = ep_dir / sensors_dirname

    joint_pos = torch.load(leader / "joint_pos.pt")
    gripper = torch.load(leader / "gripper_state.pt")

    if joint_pos.ndim != 2 or joint_pos.shape[1] != 7:
        raise ValueError(
            f"Expected joint_pos shape [T,7], got {tuple(joint_pos.shape)} in {leader}"
        )
    if gripper.ndim == 1:
        gripper = gripper[:, None]

    # Load image streams
    cam_frames: Dict[str, np.ndarray] = {}
    for cam in cams:
        cam_frames[cam] = _read_png_folder(sensors / cam, resize)

    tactile_frames: Dict[str, np.ndarray] = {}
    for tname in tactile_names:
        tdir = sensors / tname
        if tdir.exists():
            tactile_frames[tname] = _read_png_folder(tdir, resize)

    # Align all streams to common T
    T = _compute_lengths_align(joint_pos, gripper, cam_frames, tactile_frames)

    # Build state/action with T-1 steps
    joint_np = joint_pos.detach().cpu().numpy().astype(np.float32)[:T]  # [T,7]
    grip_np = gripper.detach().cpu().numpy().astype(np.float32)[:T, :1]  # [T,1]
    state_np = joint_np[:-1]  # [T-1,7]
    action_np = np.concatenate([joint_np[1:], grip_np[1:]], axis=1)  # [T-1,8]

    # Trim image streams to T-1 frames to match action length
    cam_trimmed = {k: v[: T - 1] for k, v in cam_frames.items()}  # [T-1,H,W,3]
    tactile_trimmed = {k: v[: T - 1] for k, v in tactile_frames.items()}  # [T-1,H,W,3]

    task_instr = task_instruction_mapping[ep_dir.parent.name]
    # Stream frames
    for i in range(T - 1):
        image_dict = {}
        for cam, arr in cam_trimmed.items():
            # Convert from (H, W, C) to (C, H, W) for LeRobot
            image_dict[f"observation.images.{cam.replace('zed_','')}"] = np.transpose(
                arr[i], (2, 0, 1)
            )
        for tname, arr in tactile_trimmed.items():
            # Convert from (H, W, C) to (C, H, W) for LeRobot
            image_dict[f"observation.images.{tname}"] = np.transpose(arr[i], (2, 0, 1))

        lerobot_ds.add_frame(
            {
                **image_dict,
                "observation.state": state_np[i],
                "action": action_np[i],
                "task": task_instr,
            },
        )
    lerobot_ds.save_episode()


def create_lerobot_dataset(
    raw_dir: Path,
    local_dir: Optional[Path],
    repo_id: Optional[str],
    push_to_hub: bool,
    robot_type: Optional[str],
    fps: Optional[int],
    use_videos: bool,
    image_writer_process: int,
    image_writer_threads: int,
    keep_images: bool,  # kept for parity; LeRobot handles storage internally
    cams: List[str],
    tactile_names: List[str],
    leader_subdir: str,
    sensors_dirname: str,
    resize_w: Optional[int],
    resize_h: Optional[int],
    task_instruction_mapping: Dict[str, str],
):
    raw_dir = raw_dir.resolve()
    if local_dir is None:
        local_dir = Path(HF_LEROBOT_HOME)
    dataset_name = raw_dir.name
    out_root = local_dir.resolve()
    if out_root.exists():
        shutil.rmtree(out_root)

    episodes = _find_episode_dirs(raw_dir, leader_subdir)
    if not episodes:
        raise RuntimeError(
            f"No episodes found in {raw_dir} with leader subdir '{leader_subdir}'"
        )

    resize = None
    if resize_w is not None and resize_h is not None:
        resize = (resize_w, resize_h)

    # Probe shapes for features
    cam_shapes, tactile_shapes = probe_first_valid_episode(
        episodes, cams, tactile_names, leader_subdir, sensors_dirname, resize
    )

    # FPS & robot type defaults
    if fps is None:
        fps = 10
    if robot_type is None:
        robot_type = "unknown"

    # Build feature spec
    features = _build_features_spec(cam_shapes, tactile_shapes, use_videos)

    # Create target LeRobot dataset
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type.lower().replace(" ", "_").replace("-", "_"),
        root=out_root,
        fps=int(fps),
        use_videos=use_videos,
        features=features,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_process,
    )

    # Convert each episode
    failures = 0
    for ep in episodes:
        try:
            save_episode_to_lerobot(
                ds,
                ep_dir=ep,
                cams=cams,
                tactile_names=tactile_names,
                leader_subdir=leader_subdir,
                sensors_dirname=sensors_dirname,
                resize=resize,
                task_instruction_mapping=task_instruction_mapping,
            )
            print(f"[OK]   episode {ep}")
        except Exception as e:
            failures += 1
            print(f"[WARN] skipping episode {ep}: {e}")

    print(f"Done. Episodes written: {len(episodes)-failures} / {len(episodes)}")
    print(f"LeRobot dataset root: {out_root}")

    if push_to_hub:
        assert repo_id is not None, "repo_id required to push_to_hub"
        tags = ["LeRobot", dataset_name]
        if tactile_names:
            tags.append("tactile")
        if robot_type != "unknown":
            tags.append(robot_type)
        ds.push_to_hub(
            tags=tags,
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print("Pushed to Hub.")


# ---------- CLI ----------


def main():
    # Configuration - modify these variables as needed
    raw_dir = Path("/home/multimodallearning/data_collected/flower/trickandtreat")
    local_dir = Path(
        "/home/multimodallearning/data_collected/flower-lerobot/trickandtreat"
    )
    repo_id = None  # HF repo id (e.g. user/dataset). Required if push_to_hub=True
    push_to_hub = False
    robot_type = "franka_panda"
    fps = 20
    use_videos = True
    image_writer_process = 5
    image_writer_threads = 10
    keep_images = False

    # Structure configuration
    leader_subdir = "Gello leader"
    sensors_dirname = "sensors"
    cams = ["zed_right_cam", "zed_left_cam", "zed_wrist_cam"]
    tactile_names = []  # Optional tactile folder names
    resize_w = 256
    resize_h = 256

    # The script looks up the parent dir name of an episode and matches the key
    # in the following directory to identify the correct task instruction
    task_instruction_mapping = {
        "pepper_only": "Pick up the bell pepper and place it in the bowl."
    }

    create_lerobot_dataset(
        raw_dir=raw_dir,
        local_dir=local_dir,
        repo_id=repo_id,
        push_to_hub=push_to_hub,
        robot_type=robot_type,
        fps=fps,
        use_videos=use_videos,
        image_writer_process=image_writer_process,
        image_writer_threads=image_writer_threads,
        keep_images=keep_images,
        cams=cams,
        tactile_names=tactile_names,
        leader_subdir=leader_subdir,
        sensors_dirname=sensors_dirname,
        resize_w=resize_w,
        resize_h=resize_h,
        task_instruction_mapping=task_instruction_mapping,
    )


if __name__ == "__main__":
    main()
