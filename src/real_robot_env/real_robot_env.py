import gymnasium as gym

from real_robot_env.robot.hardware_devices import ContinuousDevice, DiscreteDevice
from real_robot_env.robot.hardware_robot import RobotArm, RobotHand


class RealRobotEnv(gym.Env):
    def __init__(
        self,
        robot_arm: RobotArm,
        robot_hand: RobotHand,
        discrete_devices: list[DiscreteDevice] = [],
        continuous_devices: list[ContinuousDevice] = [],
    ):
        self.robot_arm = robot_arm
        self.robot_hand = robot_hand
        self.discrete_devices = discrete_devices
        self.continuous_devices = continuous_devices

        assert robot_arm.connect(), f"Connection to {robot_arm.name} failed"
        assert robot_hand.connect(), f"Connection to {robot_hand.name} failed"

        for device in discrete_devices:
            assert device.connect(), f"Connection to {device.name} failed"

        for device in continuous_devices:
            assert device.connect(), f"Connection to {device.name} failed"

    def step(self, action: dict):
        self.robot_arm.go_to_within_limits(goal=action["robot_arm"])
        self.robot_hand.apply_commands(width=action["robot_hand"])

        obs = self._get_obs()
        info = self._get_info()

        return obs, 0, False, False, info, False

    def reset(self):
        for device in self.continuous_devices:
            device.stop_recording()

        self.robot_arm.reset()
        self.robot_hand.reset()

        for device in self.continuous_devices:
            device.start_recording()

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def close(self):
        for device in self.continuous_devices:
            device.stop_recording()

        if not self.robot_arm.close():
            print(f"Failed to close {self.robot_arm.name}")

        if not self.robot_hand.close():
            print(f"Failed to close {self.robot_hand.name}")

        for device in self.discrete_devices:
            if not device.close():
                print(f"Failed to close {device.name}")

        for device in self.continuous_devices:
            if not device.close():
                print(f"Failed to close {device.name}")

    def _get_obs(self):
        obs_dict = {}

        obs_dict["robot_arm"] = self.robot_arm.get_state()
        obs_dict["robot_hand"] = self.robot_hand.get_sensors()

        for device in self.discrete_devices:
            obs_dict[device.name] = device.get_sensors()

        for device in self.continuous_devices:
            obs_dict[device.name] = device.get_state()

        return obs_dict

    def _get_info(self):
        return {}
