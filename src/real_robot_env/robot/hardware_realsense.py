# Original Author: Marcel Ruehle
import time
from collections import OrderedDict

import numpy as np
import pyrealsense2 as rs

from real_robot_env.robot.hardware_cameras import DiscreteCamera


class RealSense(DiscreteCamera):
    """
    Wrapper that implements boilerplate code for RealSense cameras.
    The recording fails, when using different dimensions than 480x640, so these are now hardcoded to capture the frames.

    Warning: This script is a bit buggy. Sometimes, this error appears (sometimes it doesn't):

    ``File "/home/kkuryshev/audio-pipeline/real_robot/real_robot_env/robot/hardware_realsense.py", line 48, in __get_frames`` \n
      ``return self.align.process(frameset)``
    ``RuntimeError: Error occured during execution of the processing block! See the log for more info``
    """

    RECORDING_HEIGHT = 480
    RECORDING_WIDTH = 640

    def __init__(
        self,
        device_id,
        name=None,
        height=480,
        width=640,
        fps=30,
        warm_start=30,
        start_frame_latency=0,
    ):
        super().__init__(
            device_id,
            name if name else f"RealSense_{device_id}",
            height,
            width,
            start_frame_latency,
        )
        self.fps = fps
        self.warm_start = warm_start
        self.pipe = None

    def _setup_connect(self):
        self.pipe = rs.pipeline()
        config = rs.config()

        config.enable_device(self.device_id)
        config.enable_stream(
            rs.stream.depth,
            self.RECORDING_WIDTH,
            self.RECORDING_HEIGHT,
            rs.format.z16,
            self.fps,
        )
        config.enable_stream(
            rs.stream.color,
            self.RECORDING_WIDTH,
            self.RECORDING_HEIGHT,
            rs.format.rgb8,
            self.fps,
        )
        self.profile = self.pipe.start(config)
        self.align = rs.align(rs.stream.color)

        for _ in range(self.warm_start):
            self.__get_frames()

    def get_intrinsics_dict(self):
        stream = self.profile.get_streams()[1]
        intrinsics = stream.as_video_stream_profile().get_intrinsics()
        param_dict = dict(
            [
                (p, getattr(intrinsics, p))
                for p in dir(intrinsics)
                if not p.startswith("__")
            ]
        )
        param_dict["model"] = param_dict["model"].name
        return param_dict

    def __get_frames(self):
        if self.pipe is None:
            raise RuntimeError("Please connect first.")
        frameset = self.pipe.wait_for_frames()
        return self.align.process(frameset)

    def get_rgbd(self):
        """
        returns color image as np.ndarray [h, w, 3] with RGB[0-255] and depth as np.ndarray [h, w] in millimeters
        """
        frameset = self.__get_frames()

        rgb = np.empty(
            [self.RECORDING_HEIGHT, self.RECORDING_WIDTH, 3], dtype=np.uint16
        )
        d = np.empty([self.RECORDING_HEIGHT, self.RECORDING_WIDTH], dtype=np.uint16)

        color_frame = frameset.get_color_frame()
        rgb = np.asanyarray(color_frame.get_data())

        depth_frame = frameset.get_depth_frame()
        d = (
            np.asanyarray(depth_frame.get_data()) * depth_frame.get_units() * 1000
        )  # in millimeters

        return rgb, d

    def _get_sensors(self):
        """
        Prompts the device to output a single frame of the sensor data.
        Output has the following format: `{'time': timestamp, 'rgb': rgb_vals, 'd': depth_vals}`

        Returns:
        -------
        - `sensor_data` (dict): Sensor data in the format `{'time': float, 'rgb': NDArray[uint16], 'd': NDArray[uint16]}`.
        """
        # get all data from all topics
        rgb, d = self.get_rgbd()
        timestamp = time.time()
        return {"time": timestamp, "rgb": rgb, "d": d}

    def close(self):
        success = super().close()
        self.pipe.stop()
        return success

    @staticmethod
    def get_devices(
        amount=-1, height: int = 480, width: int = 640, **kwargs
    ) -> list["RealSense"]:
        """
        Finds and returns specific amount of instances of this class.

        Parameters:
        ----------
        - `amount` (int): Maximum amount of instances to be found. Leaving out `amount` or `amount = -1` returns all instances.
        - `height` (int): Pixel-height of captured frames. Default: `480`
        - `width` (int): Pixel-width of captured frames. Default: `640`
        - `**kwargs`: Arbitrary keyword arguments.

        Returns:
        --------
        - `devices` (list[RealSense]): List of found devices. If no devices are found, `[]` is returned.
        """
        super(RealSense, RealSense).get_devices(
            amount, height=height, width=width, device_type="RealSense", **kwargs
        )
        cam_list = rs.context().query_devices()
        cams = []
        counter = 0
        for device in cam_list:
            if amount != -1 and counter >= amount:
                break
            cam = RealSense(
                device.get_info(rs.camera_info.serial_number),
                height=height,
                width=width,
            )
            cams.append(cam)
            counter += 1
        return cams


if __name__ == "__main__":
    cam = RealSense()
    cam.connect()
    print(cam.get_intrinsics_dict())

    for i in range(100):
        rgb, d = cam.get_rgbd()
        print(i, rgb.shape)
