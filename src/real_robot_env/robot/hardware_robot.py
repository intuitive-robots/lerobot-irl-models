import time
from abc import ABC, abstractmethod
from typing import NamedTuple

import torch


class ArmState(NamedTuple):
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    ee_pos: torch.Tensor
    ee_vel: torch.Tensor


class RobotArm(ABC):
    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def close(self) -> bool:
        pass

    def reconnect(self) -> bool:
        print("Attempting re-connection")
        self.connect()

        number_of_tries = 0
        while not self.okay() and number_of_tries < 3:
            self.connect()
            time.sleep(2)

        if self.okay():
            print("Reconnection successful")
            return True
        else:
            print("!! Reconnection unsuccessful !!")
            return False

    @abstractmethod
    def okay(self) -> bool:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def apply_commands(self, *args, **kwargs):
        pass

    @abstractmethod
    def go_to_within_limits(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_state(self) -> ArmState:
        pass


class RobotHand(ABC):
    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def close(self) -> bool:
        pass

    @abstractmethod
    def okay(self) -> bool:
        pass

    def reconnect(self) -> bool:
        print("Attempting re-connection")
        self.connect()

        number_of_tries = 0
        while not self.okay() and number_of_tries < 3:
            self.connect()
            time.sleep(2)

        if self.okay():
            print("Reconnection successful")
            return True
        else:
            print("!! Reconnection unsuccessful !!")
            return False

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def apply_commands(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_sensors(self):
        pass
