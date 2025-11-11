import logging
import time
from pathlib import Path

import einops
import numpy as np
import torch

from real_robot_env.base_sim import BaseSim
from real_robot_env.real_robot_env import RealRobotEnv
from real_robot_env.robot.hardware_audio import AudioInterface
from real_robot_env.robot.hardware_depthai import DAICameraType, DepthAI
from real_robot_env.robot.hardware_franka import ControlType, FrankaArm
from real_robot_env.robot.hardware_frankahand import FrankaHand
from real_robot_env.robot.utils.keyboard import KeyManager

DELTA_T = 0.034

logger = logging.getLogger(__name__)


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

    def image_to_tensor(self, image):

        rgb = torch.from_numpy(image.copy()).float().permute(2, 0, 1) / 255.0
        rgb = einops.rearrange(rgb, "c h w -> 1 1 c h w").to(self.device)

        return rgb

    def test_agent(self, agent):
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

        lang_emb = torch.zeros(1, 1, 512).float().to(self.device)

        logger.info("Starting trained model evaluation on real robot")

        km = KeyManager()

        while km.key != "q":
            print("Press 's' to start a new evaluation, or 'q' to quit")
            km.pool()

            while km.key not in ["s", "q"]:
                km.pool()
            vis = False
            if km.key == "s":
                print()

                agent.reset()
                obs, _ = env.reset()

                print("Starting evaluation. Press 'd' to stop current evaluation")

                km.pool()
                while km.key != "d":
                    km.pool()
                    robot_states = torch.tensor(obs["robot_arm"].joint_pos)
                    robot_states = robot_states.unsqueeze(0).unsqueeze(0)
                    right_cam = obs["right_cam"]["rgb"]
                    wrist_cam = obs["wrist_cam"]["rgb"]

                    right_cam = self.image_to_tensor(right_cam)
                    wrist_cam = self.image_to_tensor(wrist_cam)

                    obs_dict = {
                        # "front_cam_image": front_cam,
                        "observation.images.right_cam": right_cam,
                        "observation.images.wrist_cam": wrist_cam,
                        "task": "Grab the sweet and put it on the hand",
                        "observation.state": robot_states,
                    }

                    pred_action = agent.select_action(obs_dict).cpu().numpy()
                    pred_joint_pos = pred_action[0, :7]
                    pred_gripper_command = pred_action[0, -1]
                    pred_gripper_command = 1 if pred_gripper_command > 0 else -1

                    action = {
                        "robot_arm": pred_joint_pos,
                        "robot_hand": pred_gripper_command,
                    }

                    obs, *_ = env.step(action)

                    time.sleep(DELTA_T)

                print()
                logger.info("Evaluation done. Resetting robots")

                env.reset()

        print()
        logger.info("Quitting evaluation")

        km.close()
        env.close()
