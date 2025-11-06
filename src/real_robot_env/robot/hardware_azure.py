import logging
import time
from typing import Optional

import cv2
import pykinect_azure as pykinect

from real_robot_env.robot.hardware_cameras import DiscreteCamera

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Azure(DiscreteCamera):
    """
    This class can be considered a wrapper class for Azure cameras specifically for frame collection.

    This class inherits its functions from `real_robot_env.robot.hardware_cameras.DiscreteCamera`.
    """

    def __init__(
        self, device_id, name=None, height=512, width=512, start_frame_latency=0
    ):
        super().__init__(
            device_id,
            name if name else f"Azure_{device_id}",
            height,
            width,
            start_frame_latency,
        )

        self.__set_device_configuration()  # sets self.device_config
        self.device = None

    def _setup_connect(self):
        super()._setup_connect()

        pykinect.initialize_libraries()
        self.device = pykinect.start_device(
            device_index=self.device_id, config=self.device_config
        )

    def _get_sensors(self):
        """
        Prompts the device to output a single frame of the sensor data.
        Output has the following format: `{'time': timestamp, 'rgb': rgb_vals}`

        Returns:
        -------
        - `sensor_data` (dict): Sensor data in the format `{'time': float, 'rgb': Any}`.
        """
        if not self.device:
            raise Exception(f"Not connected to {self.name}")

        success = False
        while not success:
            capture = self.device.update()
            success, image = capture.get_color_image()
            timestamp = time.time()

        return {"time": timestamp, "rgb": image}

    def close(self):
        success = super().close()
        if self.device:
            self.device.close()
        self.device = None
        return success

    def __set_device_configuration(self):
        self.device_config = pykinect.default_configuration
        self.device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        self.device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1536P
        # self.device_config.depth_mode = pykinect.K4A_DEPTH_MODE_OFF

    @staticmethod
    def get_devices(
        amount, height: int = 512, width: int = 512, **kwargs
    ) -> list["Azure"]:
        """
        Returns specific amount of instances of this class.

        Parameters:
        ----------
        - `amount` (int): Amount of instances to be created.
        - `height` (int): Pixel-height of captured frames. Default: `512`
        - `width` (int): Pixel-width of captured frames. Default: `512`
        - `**kwargs`: Arbitrary keyword arguments.

        Returns:
        --------
        - `devices` (list[Azure]): List of created instances.
        """
        super(Azure, Azure).get_devices(
            amount, height=height, width=width, device_type="Azure", **kwargs
        )
        cams = []
        for i in range(amount):
            cam = Azure(device_id=i, height=height, width=width)
            cams.append(cam)
        return cams

    def get_point_cloud(self):
        if not self.device:
            raise Exception(f"Not connected to {self.name}")
        success = False
        while not success:
            capture = self.device.update()
            success, pc = capture.get_transformed_pointcloud()

        return pc

    # see https://github.com/ibaiGorordo/pyKinectAzure/blob/fdfd70ee0fc4287e0750d6b99a658210cf53cf7c/pykinect_azure/k4a/calibration.py#L6
    def get_intrinsics(self, sensor_type: str = "color"):
        if not self.device:
            raise Exception(f"Not connected to {self.name}")
        if sensor_type == "color":
            type = pykinect.K4A_CALIBRATION_TYPE_COLOR
        else:
            type = pykinect.K4A_CALIBRATION_TYPE_DEPTH

        return self.device.get_calibration().get_matrix(type)


if __name__ == "__main__":
    rs = Azure(device_id=1)
    rs.connect()

    for i in range(50):
        img = rs.get_sensors()
        if img["rgb"] is not None:
            print("Received image{} of size:".format(i), img["rgb"].shape, flush=True)
            cv2.imshow("rgb", img["rgb"])
            cv2.waitKey(1)

        if img["rgb"] is None:
            print(img)

        time.sleep(0.1)

    rs.close()
