# Original Author: Pankhuri Vanjani
import datetime
import time
from pathlib import Path

import cv2
from digit_interface.digit import Digit
from digit_interface.digit_handler import DigitHandler

from real_robot_env.robot.hardware_devices import DiscreteDevice


class TactileDigit(DiscreteDevice):
    """
    Discretedevice Wrapper that implements the DIGIT tactile sensor.``
    """

    def __init__(self, device_id, name=None, start_frame_latency=0):
        super().__init__(
            device_id, name if name else f"Digit_{device_id}", start_frame_latency
        )
        self.formats = [".png"]
        self.handler = DigitHandler()
        self.sensor = Digit(device_id)
        self.timestamps: list[float] = []

    def _setup_connect(self) -> None:
        # Ensure sensor is available and connect
        if hasattr(self.handler, "list_digits"):
            devs = self.handler.list_digits()
            serials = [d.get("serial") for d in devs]
        elif hasattr(self.handler, "get_serials"):
            serials = self.handler.get_serials()
        else:
            raise RuntimeError(
                "Cannot enumerate DIGIT devices: no known handler method"
            )
        if self.device_id not in serials:
            raise RuntimeError(
                f"DIGIT {self.device_id} not found (available: {serials})"
            )
        self.sensor.connect()

    def get_sensors(self) -> dict:
        # Capture a single frame and timestamp
        frame = self.sensor.get_frame()
        timestamp = time.time()
        return {"time": timestamp, "rgb": frame}

    def close(self) -> bool:
        self.sensor.disconnect()
        return True

    def store_last_frame(self, directory: Path, filename: str = None) -> None:
        # Save the latest tactile frame as PNG
        """
        Store the last DIGIT frame to a given directory with the filename.
        Parameters:
        ----------
        - `directory` (Path): Target directory
        - `filename` (str): Image file name (without extension). If none is given, the frame is stored with
                            the current timestamp as title.
        """
        data = self.get_sensors()
        self.timestamps.append(data["time"])
        directory.mkdir(parents=True, exist_ok=True)
        if filename is None:
            timestamp = datetime.datetime.fromtimestamp(data["time"])
            filename = str(directory / timestamp.isoformat()) + self.formats[0]
        else:
            filename = str(directory / f"{filename}") + self.formats[0]
        cv2.imwrite(filename, cv2.cvtColor(data["rgb"], cv2.COLOR_RGB2BGR))

        # frame = self._get_sensors()
        # filepath = directory / f"{filename}.png"
        # cv2.imwrite(str(filepath), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # @staticmethod
    # def get_devices(amount=-1, height: int = 480, width: int = 640, **kwargs) -> list['RealSense']:
    #     # Discover connected DIGIT sensors
    #     handler = DigitHandler()
    #     serials = handler.get_serials()
    #     devices = []
    #     for i, serial in enumerate(serials):
    #         if 0 <= amount == i:
    #             break
    #         devices.append(Digit(serial, **kwargs))
    #     return devices

    @staticmethod
    def get_devices(amount: int = -1, **kwargs) -> list["TactileDigit"]:
        # Discover connected DIGIT sensors
        handler = DigitHandler()
        # Enumerate via list_digits if available
        if hasattr(handler, "list_digits"):
            devs = handler.list_digits()
            serials = [d.get("serial") for d in devs]
        elif hasattr(handler, "get_serials"):
            serials = handler.get_serials()
        else:
            raise RuntimeError("Cannot list DIGIT devices: no known handler method")
        devices: list[TactileDigit] = []
        for i, serial in enumerate(serials):
            if 0 <= amount == i:
                break
            devices.append(TactileDigit(serial, **kwargs))
        return devices
