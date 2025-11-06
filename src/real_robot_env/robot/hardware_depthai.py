import logging
import time
from enum import Enum

import cv2

from real_robot_env.robot.hardware_cameras import DiscreteCamera

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DAICameraType(Enum):
    OAK_D = 0
    OAK_D_LITE = 1
    OAK_D_SR = 2


# adapted by TimWindecker
class DepthAI(DiscreteCamera):
    """
    This class can be considered a wrapper class for DepthAI cameras specifically for frame collection.

    This class inherits its functions from `real_robot_env.robot.hardware_devices.DiscreteDevice`.
    """

    def __init__(
        self,
        device_id,
        name=None,
        height=512,
        width=512,
        start_frame_latency=0,
        camera_type: DAICameraType = DAICameraType.OAK_D_LITE,
        **kwargs,
    ):
        super().__init__(
            device_id,
            name if name else f"DepthAI_{device_id}",
            height,
            width,
            start_frame_latency,
        )

        self.camera_type = camera_type

    def _setup_connect(self):
        super()._setup_connect()

        # DepthAI is bugged for multiprocessing, this fixes it (See https://github.com/luxonis/depthai/issues/697)
        import depthai as dai

        if self.camera_type in [DAICameraType.OAK_D, DAICameraType.OAK_D_LITE]:
            cam, board_socket, resolution = (
                dai.node.ColorCamera,
                dai.CameraBoardSocket.CAM_A,
                dai.ColorCameraProperties.SensorResolution.THE_1080_P,
            )
        elif self.camera_type == DAICameraType.OAK_D_SR:
            cam, board_socket, resolution = (
                dai.node.ColorCamera,
                dai.CameraBoardSocket.CAM_C,
                dai.ColorCameraProperties.SensorResolution.THE_800_P,
            )

        pipeline = dai.Pipeline()
        camRgb = pipeline.create(cam)
        camRgb.setBoardSocket(board_socket)
        camRgb.setResolution(resolution)
        xoutRgb = pipeline.create(dai.node.XLinkOut)
        xoutRgb.setStreamName("rgb")
        camRgb.video.link(xoutRgb.input)

        self.device_info = dai.DeviceInfo(self.device_id)
        self.pipeline = pipeline

        self.device = dai.Device(self.pipeline, self.device_info)
        self.qRgb = self.device.getOutputQueue(name="rgb", maxSize=1, blocking=False)

    def _get_sensors(self):
        """
        Prompts the device to output a single frame of the sensor data.
        Output has the following format: `{'time': timestamp, 'rgb': rgb_vals}`

        Depth is currently not supported ('d': NDArray[float64]).

        Returns:
        -------
        - `sensor_data` (dict): Sensor data in the format `{'time': float, 'rgb': Any}`.
        """

        inRgb = self.qRgb.get()
        bgr_img = inRgb.getCvFrame()
        frame = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        timestamp = time.time()

        return {"time": timestamp, "rgb": frame}

    def close(self):
        success = super().close()
        if hasattr(self, "device"):
            self.device.close()
        return success

    @staticmethod
    def get_devices(
        amount=-1, height: int = 512, width: int = 512, **kwargs
    ) -> list["DepthAI"]:
        """
        Finds and returns specific amount of instances of this class.

        Parameters:
        ----------
        - `amount` (int): Maximum amount of instances to be found. Leaving out `amount` or `amount = -1` returns all instances.
        - `height` (int): Pixel-height of captured frames. Default: `512`
        - `width` (int): Pixel-width of captured frames. Default: `512`
        - `**kwargs`: Arbitrary keyword arguments.

        Returns:
        --------
        - `devices` (list[DepthAI]): List of found devices. If no devices are found, `[]` is returned.
        """
        super(DepthAI, DepthAI).get_devices(
            amount, height=height, width=width, device_type="DepthAI", **kwargs
        )

        # DepthAI is bugged for multiprocessing, this fixes it (See https://github.com/luxonis/depthai/issues/697)
        import depthai as dai

        cam_list = dai.Device.getAllAvailableDevices()
        cams = []
        counter = 0
        for device in cam_list:
            if amount != -1 and counter >= amount:
                break
            cam = DepthAI(device.getMxId(), height=height, width=width)
            cams.append(cam)
            counter += 1
        return cams


if __name__ == "__main__":

    # DepthAI is bugged for multiprocessing, this fixes it (See https://github.com/luxonis/depthai/issues/697)
    # DepthAI is bugged for multiprocessing, this fixes it (See https://github.com/luxonis/depthai/issues/697)
    import depthai as dai

    print(dai.DeviceInfo())
    MXID = "1844301021D9BF1200"
    # MXID = '1844301071E7AB1200' # second
    rs = DepthAI(name="test cam", device_id=MXID)
    rs.connect()

    for i in range(1000000):
        img = rs.get_sensors()
        if img["rgb"] is not None:
            print("Received image{} of size:".format(i), img["rgb"].shape, flush=True)
            cv2.imshow("rgb", cv2.cvtColor(img["rgb"], cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)

        if img["rgb"] is None:
            print(img)

        time.sleep(0.1)

    rs.close()
