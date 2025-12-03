import logging
import time
from pathlib import Path
import math
import sys

import cv2
import einops
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import traceback

from real_robot_env.base_sim import BaseSim
from real_robot_env.real_robot_env import RealRobotEnv
from real_robot_env.robot.hardware_audio import AudioInterface
from real_robot_env.robot.hardware_depthai import DAICameraType, DepthAI
from real_robot_env.robot.hardware_franka import ControlType, FrankaArm
from real_robot_env.robot.hardware_frankahand import FrankaHand
from real_robot_env.robot.utils.keyboard import KeyManager
from threading import Event, Lock, Thread
from dataclasses import dataclass, field
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.rl.process import ProcessSignalHandler
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.utils.hub import HubMixin
from lerobot.utils.constants import ACTION

DELTA_T = 0.034

logger = logging.getLogger(__name__)



class EnvWrapper:
    def __init__(self, env: RealRobotEnv):
        self.env = env
        self.lock = Lock()

    def get_observation(self):
        with self.lock:
            return self.env.get_observation()

    def step(self, action):
        with self.lock:
            return self.env.step(action)

    def reset(self):
        with self.lock:
            return self.env.reset()

    def image_to_tensor(self, image, device):
        # This doesn't need lock - pure computation
        # Move this method here from wherever it currently is
        rgb = torch.from_numpy(image.copy()).float().permute(2, 0, 1) / 255.0
        rgb = einops.rearrange(rgb, "c h w -> 1 1 c h w").to(device)

        return rgb
    


def get_actions(
    agent,
    env_wrapper: EnvWrapper,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: DictConfig,
):
    """Thread function to request action chunks from the policy.

    Args:
        policy: The policy instance (SmolVLA, Pi0, etc.)
        robot: The robot instance for getting observations
        robot_observation_processor: Processor for raw robot observations
        action_queue: Queue to put new action chunks
        shutdown_event: Event to signal shutdown
        cfg: Demo configuration
    """
    try:
        logger.info("[GET_ACTIONS] Starting get actions thread")

        latency_tracker = LatencyTracker()  # Track latency of action chunks
        fps = cfg.eval.fps
        time_per_chunk = 1.0 / fps

        get_actions_threshold = cfg.eval.action_queue_size_to_get_new_actions

        if not cfg.rtc.enabled:
            get_actions_threshold = 0

        while not shutdown_event.is_set():
            if action_queue.qsize() <= get_actions_threshold:
                current_time = time.perf_counter()

                action_index_before_inference = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()

                inference_latency = latency_tracker.max()
                inference_delay = math.ceil(inference_latency / time_per_chunk)

                # Get current observation (introduced a new method)
                obs = env_wrapper.get_observation()

                robot_states = torch.tensor(obs["robot_arm"].joint_pos)
                robot_states = robot_states.unsqueeze(0).unsqueeze(0)
                right_cam = obs["right_cam"]["rgb"]
                wrist_cam = obs["wrist_cam"]["rgb"]

                right_cam = env_wrapper.image_to_tensor(right_cam, cfg.device) #?
                wrist_cam = env_wrapper.image_to_tensor(wrist_cam, cfg.device) #?

                obs_dict = {
                    # "front_cam_image": front_cam,
                    "observation.images.right_cam": right_cam,
                    "observation.images.wrist_cam": wrist_cam,
                    "task": "pick_up_blue_cube",
                    "observation.state": robot_states,
                }
                #cv2.imwrite(f"debug_frame_{self.i}.jpg", obs["right_cam"]["rgb"])

                # Normalize inputs before predict_action_chunk (prev, it was done as a part of select_action, but we do not use that method anymore)
                obs_dict_normalized = agent.normalize_inputs(obs_dict)

                # Generate actions WITH RTC
                actions = agent.predict_action_chunk(
                    obs_dict_normalized,  # ← Use normalized version
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                )

                original_actions = actions.squeeze(0).clone()  # Normalized (for RTC)

                # ✅ Unnormalize for robot
                postprocessed_actions = agent.unnormalize_outputs({ACTION: actions})[ACTION]
                postprocessed_actions = postprocessed_actions.squeeze(0)

                new_latency = time.perf_counter() - current_time
                new_delay = math.ceil(new_latency / time_per_chunk)
                latency_tracker.add(new_latency)

                if cfg.eval.action_queue_size_to_get_new_actions < cfg.rtc.execution_horizon + new_delay:
                    logger.warning(
                        "[GET_ACTIONS] cfg.action_queue_size_to_get_new_actions Too small, It should be higher than inference delay + execution horizon."
                    )

                action_queue.merge(
                    original_actions, postprocessed_actions, new_delay, action_index_before_inference
                )
            else:
                # Small sleep to prevent busy waiting
                time.sleep(0.1)

        logger.info("[GET_ACTIONS] get actions thread shutting down")
    except Exception as e:
        logger.error(f"[GET_ACTIONS] Fatal exception in get_actions thread: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def actor_control(
    env_wrapper: EnvWrapper,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: DictConfig,
):
    """Thread function to execute actions on the robot.

    Args:
        robot: The robot instance
        action_queue: Queue to get actions from
        shutdown_event: Event to signal shutdown
        cfg: Demo configuration
    """
    try:
        logger.info("[ACTOR] Starting actor thread")

        action_count = 0
        action_interval = 1.0 / cfg.eval.fps  #####

        while not shutdown_event.is_set():
            start_time = time.perf_counter()

            # Try to get an action from the queue with timeout
            pred_action = action_queue.get() # pred_action = agent.select_action(obs_dict)

            if pred_action is not None:
                pred_action = pred_action.cpu().numpy() # pred_action = agent.select_action(obs_dict).cpu().numpy()
                pred_joint_pos = pred_action[:7]
                pred_gripper_command = pred_action[-1]
                pred_gripper_command = 1 if pred_gripper_command > 0 else -1

                action = {
                    "robot_arm": pred_joint_pos,
                    "robot_hand": pred_gripper_command,
                }

                env_wrapper.step(action) # I do not really need to get new obs or whatever

                action_count += 1

            dt_s = time.perf_counter() - start_time
            time.sleep(max(0, (action_interval - dt_s) - 0.001))

        logger.info(f"[ACTOR] Actor thread shutting down. Total actions executed: {action_count}")
    except Exception as e:
        logger.error(f"[ACTOR] Fatal exception in actor_control thread: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


class RealRobot(BaseSim):
    def __init__(self, device: str):
        super().__init__(seed=-1, device=device)

        self.p4 = FrankaArm(
            name="202_robot",
            ip_address="141.3.53.63",
            port=50053,
            control_type=ControlType.HYBRID_JOINT_IMPEDANCE_CONTROL,
            hz=100,
        )
        assert self.p4.connect(), f"Connection to {self.p4.name} failed"

        self.p4_hand = FrankaHand(
            name="202_gripper", ip_address="141.3.53.63", port=50054
        )
        assert self.p4_hand.connect(), f"Connection to {self.p4_hand.name} failed"

        self.i = 0

    def test_agent(self, agent, cfg: DictConfig, rtc_cfg: RTCConfig):
        self.cam0 = DepthAI(  # right cam
            device_id="1844301051D9B50F00",
            name="right_cam",  # named orb due to other code dependencies
            height=224,
            width=224,
            camera_type=DAICameraType.OAK_D,
        )
        #############Wrist camera
        self.cam1 = DepthAI(  # wrist cam
            device_id="1944301061BB782700",
            name="wrist_cam",  # named orb due to other code dependencies
            height=224,
            width=224,
            camera_type=DAICameraType.OAK_D_SR,
        )

        env = RealRobotEnv(
            robot_arm=self.p4,
            robot_hand=self.p4_hand,
            discrete_devices=[self.cam0, self.cam1],
        )

        #lang_emb = torch.zeros(1, 1, 512).float().to(self.device)

        logger.info("Starting trained model evaluation on real robot")

        km = KeyManager()

        while km.key != "q":
            print("Press 's' to start a new evaluation, or 'q' to quit")
            km.pool()

            while km.key not in ["s", "q"]:
                km.pool()

            #vis = False

            signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
            shutdown_event = signal_handler.shutdown_event

            if km.key == "s":

                env_wrapper = EnvWrapper(env)
                print()

                agent.reset()
                env.reset()

                print("Starting evaluation. Press 'd' to stop current evaluation")

                # Create action queue for communication between threads
                action_queue = ActionQueue(rtc_cfg)
                print("Action queue created succesfully")

                # Start chunk requester thread
                get_actions_thread = Thread(
                    target=get_actions,
                    args=(agent, env_wrapper, action_queue, shutdown_event, cfg),
                    daemon=True,
                    name="GetActions",
                )

                get_actions_thread.start()
                logger.info("Started get actions thread")

                # Start action executor thread
                actor_thread = Thread(
                    target=actor_control,
                    args=(env_wrapper, action_queue, shutdown_event, cfg),
                    daemon=True,
                    name="Actor",
                )
                actor_thread.start()
                logger.info("Started actor thread")
                
                km.pool()
                while km.key != "d":
                    km.pool()
                    if time.time() % 10 < 0.1:  # Every ~10 seconds
                        logger.info(f"[MAIN] Action queue size: {action_queue.qsize()}")
                    time.sleep(DELTA_T)

                shutdown_event.set()

                if get_actions_thread and get_actions_thread.is_alive():
                    logger.info("Waiting for get_actions thread...")
                    get_actions_thread.join(timeout=5.0)

                if actor_thread and actor_thread.is_alive():
                    logger.info("Waiting for actor thread...")
                    actor_thread.join(timeout=5.0)
                
                if agent.model.rtc_processor and agent.model.rtc_processor.is_debug_enabled():
                    debug_steps = agent.model.rtc_processor.get_all_debug_steps()
                    logger.info(f"Collected {len(debug_steps)} debug steps")

                    # Save debug data
                    import pickle
                    with open("rtc_debug.pkl", "wb") as f:
                        pickle.dump([step.to_dict() for step in debug_steps], f)
                    logger.info("Debug data saved to rtc_debug.pkl")


                print()
                logger.info("Evaluation done. Resetting robots")
                env.reset()

        print()
        logger.info("Quitting evaluation")

        km.close()
        env.close()