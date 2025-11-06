import argparse
import concurrent
import time

import numpy as np
from polymetis import GripperInterface

from real_robot_env.robot.hardware_robot import RobotHand


class FrankaHand(RobotHand):
    def __init__(self, name, ip_address, port=50052, **kwargs):
        self.name = name
        self.ip_address = ip_address
        self.port = port
        self.robot = None
        self.max_width = 0.0
        self.min_width = 0.0

        self.within_grasp_action = False

        self.pool = concurrent.futures.ThreadPoolExecutor(1)

    def connect(self):
        """Establish hardware connection"""
        connection = False
        # Initialize self.robot interface
        print("FrankaHand:> Connecting to {}: ".format(self.name), end="")
        try:
            self.robot = GripperInterface(ip_address=self.ip_address, port=self.port)
            print("Success")
        except Exception as e:
            self.robot = None  # declare dead
            print("Failed with exception: ", e)
            return connection

        print("FrankaHand:> Testing {} connection: ".format(self.name), end="")
        if self.okay():
            print("Okay")
            # get max_width based on polymetis version
            if self.robot.metadata:
                self.max_width = self.robot.metadata.max_width
            elif self.robot.get_state().max_width:
                self.max_width = self.robot.get_state().max_width
            else:
                self.max_width = 0.085
            connection = True

            # self.reset()
        else:
            print("Not ready. Please retry connection")

        return connection

    def okay(self):
        """Return hardware health"""
        okay = False
        if self.robot:
            try:
                state = self.robot.get_state()
                delay = time.time() - (
                    state.timestamp.seconds + 1e-9 * state.timestamp.nanos
                )
                assert delay < 5, "Acquired state is stale by {} seconds".format(delay)
                okay = True
            except:
                self.robot = None  # declare dead
                okay = False
        return okay

    def close(self):
        """Close hardware connection"""
        if self.robot:
            print("FrankaHand:> Resetting robot before close: ", end="")
            try:
                self.reset()
                print("FrankaHand:> Success: ", end="")
            except:
                print("FrankaHand:> Failed. Exiting : ", end="")
            self.robot = None
            print("Connection closed")
        return True

    def reset(self, width=None, **kwargs):
        """Reset hardware"""
        # if not width:
        #     width = self.max_width
        # self.apply_commands(width=width, **kwargs)
        self.apply_commands(-1)
        time.sleep(2)
        self.apply_commands(1)
        time.sleep(2)

    def get_sensors(self):
        """Get hardware sensors"""
        try:
            curr_state = self.robot.get_state()
        except:
            print("FrankaHand:> Failed to get current sensors: ", end="")
            self.reconnect()
            return self.get_sensors()
        return np.array([curr_state.width])

    def apply_commands(self, width: float, speed: float = 0.1, force: float = 0.1):
        # assert width>=0.0 and width<=self.max_width, "Gripper desired width ({}) is out of bound (0,{})".format(width, self.max_width)
        # print("Gripper: {}".format(width))
        if width < 0:
            self.grasp(speed, force)
        else:
            self.open(speed, force)

        return 0

    def grasp(self, speed, force, blocking: bool = False):
        # don't send grasp if we are currently in one already
        if self.within_grasp_action:
            return

        state = self.robot.get_state()
        if state.is_moving:
            return

        # # don't issue grasp if not fully open
        # if state.width < (self.max_width * (4/5)):
        #     return

        if not state.is_grasped:
            if blocking:
                self.grasp_helper(speed, force)
            else:
                self.pool.submit(self.grasp_helper, speed, force)

    def grasp_helper(self, speed, force):
        self.within_grasp_action = True
        # print('send_grasp')

        self.robot.grasp(speed, force)

        state = self.robot.get_state()
        while not state.is_grasped:
            state = self.robot.get_state()
            time.sleep(0.1)
        while state.is_moving:
            state = self.robot.get_state()
            time.sleep(0.1)

        self.within_grasp_action = False

    def open(self, speed, force, blocking: bool = False):
        state = self.robot.get_state()
        if state.is_moving:
            return

        if state.is_grasped:
            # print('send open')
            # print(state)
            if blocking:
                self.robot.goto(self.max_width, speed, force)
            else:
                self.pool.submit(self.robot.goto, self.max_width, speed, force)


# Get inputs from user
def get_args():
    parser = argparse.ArgumentParser(description="Polymetis based gripper client")

    parser.add_argument(
        "-i",
        "--server_ip",
        type=str,
        help="IP address or hostname of the franka server",
        default="localhost",
    )  # 172.16.0.1

    return parser.parse_args()


if __name__ == "__main__":

    # args = get_args()

    # user inputs
    time_to_go = 2.0 * np.pi
    m = 0.5  # magnitude of sine wave (rad)
    T = 2.0  # period of sine wave
    hz = 50  # update frequency

    # Initialize robot
    rbq = FrankaHand(name="Demo_frankahand", ip_address="141.3.53.154")

    # connect to robot
    status = rbq.connect()
    assert status, "Can't connect to FrankaHand"

    # reset using the user controller
    rbq.reset()

    # Close gripper
    des_width = -1.0
    rbq.apply_commands(width=des_width)
    time.sleep(2)
    curr_width = rbq.get_sensors()
    print(
        "FrankaHand:> Testing gripper close: Desired:{}, Achieved:{}".format(
            des_width, curr_width
        )
    )

    # Open gripper
    des_width = 1.0
    rbq.apply_commands(width=des_width)
    time.sleep(2)
    curr_width = rbq.get_sensors()
    print(
        "FrankaHand:> Testing gripper Open: Desired:{}, Achieved:{}".format(
            des_width, curr_width
        )
    )

    # # Contineous control
    # for i in range(20):
    #     ctrl = -1.0 if i%2==0 else 1.0
    #     print(ctrl)
    #     rbq.apply_commands(width=ctrl)
    #     time.sleep(1)
    #     rbq.apply_commands(width=ctrl)
    #     time.sleep(2)
    #     # time.sleep(1 / hz)

    # Drive gripper using keyboard
    if True:
        from utils.keyboard import KeyManager

        km = KeyManager()
        km.pool()
        sen = None
        print("Press 'q' to stop listening")
        while sen != "q":
            km.pool()
            sen = km.key
            # state = rbq.robot.get_state()
            # print(f'grasp: {state.is_grasped}, moving: {state.is_moving}')
            if sen is not None:
                print(sen, flush=True)
                if sen == "up":
                    rbq.apply_commands(width=1, speed=0.1, force=0.1)
                elif sen == "down":
                    rbq.apply_commands(width=-1, speed=0.1, force=0.1)
            time.sleep(0.01)

    # close connection
    rbq.close()
    print("FrankaHand:> Demo Finished")
