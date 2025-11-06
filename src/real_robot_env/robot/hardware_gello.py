import pickle
from collections import namedtuple

import zmq

from real_robot_env.robot.hardware_robot import RobotArm, RobotHand

DEFAULT_ROBOT_PORT = 6000


class AbstractGello:
    def __init__(
        self, name: str, host: str = "127.0.0.1", port: int = DEFAULT_ROBOT_PORT
    ):
        self.name = name
        self.host = host
        self.port = port
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)

    def connect(self) -> bool:
        print(f"Connecting to {self.name}: ", end="")
        try:
            self._socket.connect(f"tcp://{self.host}:{self.port}")
            print("Success")
            return True
        except Exception as e:
            print("Failed with exception: ", e)
            return False

    def close(self):
        self._socket.disconnect(f"tcp://{self.host}:{self.port}")
        return True

    def okay(self):
        return True

    def apply_commands(self):
        # At the moment Gello can't be controled
        return 0

    def reset(self):
        # At the moment Gello can't be controled
        return 0


class GelloArm(AbstractGello, RobotArm):
    def get_state(self):
        if self._socket.closed:
            raise Exception(f"Not connected to {self.name}")

        request = {"method": "get_joint_state"}
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())

        return result

    def go_to_within_limits(self, *args, **kwargs):
        # At the moment Gello can't be controled
        raise NotImplementedError


class GelloGripper(AbstractGello, RobotHand):
    def __init__(self, name, host="127.0.0.1", port=DEFAULT_ROBOT_PORT):
        super().__init__(name, host, port)
        self.max_width = 1
        self.min_width = 0

    def get_sensors(self):
        if self._socket.closed:
            raise Exception(f"Not connected to {self.name}")

        request = {"method": "get_gripper_state"}
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())

        return self._convert_closed_to_width(result)

    def _convert_closed_to_width(self, closed: float):
        # Gello outputs gripper VALUES between 0 and 1 where 0 is open and 1 is closed
        # Polymetis / FrankaHand outputs the gripper's width between min_width and max_width

        return self.max_width - closed
