import json
import logging
import os
import time
from typing import Any, List, Optional, Union

import cv2
import numpy as np
from pyorbbecsdk import *
from pyorbbecsdk import FormatConvertFilter, OBConvertFormat, OBFormat, VideoFrame
from sympy import false

from real_robot_env.robot.hardware_cameras import DiscreteCamera

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm

config_file_path = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    "../../../pyorbbecsdk/config/multi_device_sync_config.json",
)


def read_config():
    multi_device_sync_config = {}
    with open(config_file_path, "r") as f:
        config = json.load(f)
    for device in config["devices"]:
        multi_device_sync_config[device["serial_number"]] = device
        print(f"Device {device['serial_number']}: {device['config']['mode']}")
    return multi_device_sync_config


def sync_mode_from_str(sync_mode_str: str) -> OBMultiDeviceSyncMode:
    # to lower case
    sync_mode_str = sync_mode_str.upper()
    if sync_mode_str == "FREE_RUN":
        return OBMultiDeviceSyncMode.FREE_RUN
    elif sync_mode_str == "STANDALONE":
        return OBMultiDeviceSyncMode.STANDALONE
    elif sync_mode_str == "PRIMARY":
        return OBMultiDeviceSyncMode.PRIMARY
    elif sync_mode_str == "SECONDARY":
        return OBMultiDeviceSyncMode.SECONDARY
    elif sync_mode_str == "SECONDARY_SYNCED":
        return OBMultiDeviceSyncMode.SECONDARY_SYNCED
    elif sync_mode_str == "SOFTWARE_TRIGGERING":
        return OBMultiDeviceSyncMode.SOFTWARE_TRIGGERING
    elif sync_mode_str == "HARDWARE_TRIGGERING":
        return OBMultiDeviceSyncMode.HARDWARE_TRIGGERING
    else:
        raise ValueError(f"Invalid sync mode: {sync_mode_str}")


def frame_to_bgr_image(frame: VideoFrame) -> Union[Optional[np.array], Any]:
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif color_format == OBFormat.I420:
        image = i420_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV12:
        image = nv12_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV21:
        image = nv21_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    else:
        print("Unsupported color format: {}".format(color_format))
        return None
    return image


class TemporalFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(
                frame, self.alpha, self.previous_frame, 1 - self.alpha, 0
            )
        self.previous_frame = result
        return result


class ORBBEC(DiscreteCamera):
    def __init__(
        self,
        orb_device,
        device_index: int,
        height=512,
        width=512,
        start_frame_latency=0,
    ):

        for _ in range(3):
            try:
                self.orb_device = orb_device
                self.orb_pipeline = Pipeline(orb_device)
                break
            except Exception as e:
                continue
        # self.temporal_filter = TemporalFilter(alpha=0.5)
        super().__init__(
            f"ORB_{device_index}",
            f"ORB_{device_index}",
            height,
            width,
            start_frame_latency,
        )

    def sync_setup(self):

        serial_number = self.orb_device.get_device_info().get_serial_number()
        # serial_number_list[i] = serial_number
        multi_device_sync_config = read_config()
        sync_config_json = multi_device_sync_config[serial_number]
        sync_config = self.orb_device.get_multi_device_sync_config()
        sync_config.mode = sync_mode_from_str(sync_config_json["config"]["mode"])
        sync_config.color_delay_us = sync_config_json["config"]["color_delay_us"]
        sync_config.depth_delay_us = sync_config_json["config"]["depth_delay_us"]
        sync_config.trigger_out_enable = sync_config_json["config"][
            "trigger_out_enable"
        ]
        sync_config.trigger_out_delay_us = sync_config_json["config"][
            "trigger_out_delay_us"
        ]
        sync_config.frames_per_trigger = sync_config_json["config"][
            "frames_per_trigger"
        ]
        print(f"Device {serial_number} sync config: {sync_config}")
        self.orb_device.set_multi_device_sync_config(sync_config)

    def _setup_connect(self):
        self.sync_setup()
        try:
            config = Config()
            # Get the list of color stream profiles
            color_profile_list = self.orb_pipeline.get_stream_profile_list(
                OBSensorType.COLOR_SENSOR
            )
            color_profile = color_profile_list.get_video_stream_profile(
                1280, 720, OBFormat.RGB, 30
            )
            # color_profile = color_profile_list.get_video_stream_profile(1920, 1080, OBFormat.RGB, 30)
            print("Color profile: ", color_profile)
            config.enable_stream(color_profile)

            depth_profile_list = self.orb_pipeline.get_stream_profile_list(
                OBSensorType.DEPTH_SENSOR
            )
            depth_profile = depth_profile_list.get_video_stream_profile(
                640, 576, OBFormat.Y16, 30
            )
            print("Depth profile: ", depth_profile)
            config.enable_stream(depth_profile)

            # Set the alignment mode to hardware alignment
            print("finish config")
            self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
            self.orb_pipeline.enable_frame_sync()
            self.orb_pipeline.start(config)
        except Exception as e:
            print(e)
            import traceback

            traceback.print_exc()
            assert e

        # self.config = Config()
        # serial_number = self.orb_device.get_device_info().get_serial_number()
        # # serial_number_list[i] = serial_number
        # multi_device_sync_config = read_config()
        # sync_config_json = multi_device_sync_config[serial_number]
        # sync_config = self.orb_device.get_multi_device_sync_config()
        # sync_config.mode = sync_mode_from_str(sync_config_json["config"]["mode"])
        # sync_config.color_delay_us = sync_config_json["config"]["color_delay_us"]
        # sync_config.depth_delay_us = sync_config_json["config"]["depth_delay_us"]
        # sync_config.trigger_out_enable = sync_config_json["config"]["trigger_out_enable"]
        # sync_config.trigger_out_delay_us = sync_config_json["config"]["trigger_out_delay_us"]
        # sync_config.frames_per_trigger = sync_config_json["config"]["frames_per_trigger"]
        # print(f"Device {serial_number} sync config: {sync_config}")
        # self.orb_device.set_multi_device_sync_config(sync_config)
        #
        # self.profile_list = self.orb_pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        # # self.color_profile = self.profile_list.get_default_video_stream_profile()
        # self.color_profile = self.profile_list[0]
        # print("length: ", len(self.profile_list))
        # print("Color profile list: {}".format(self.color_profile))
        # self.config.enable_stream(self.color_profile)
        #
        # depth_profile_list = self.orb_pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        # print("depth profile list: {}".format(len(depth_profile_list)))
        # assert depth_profile_list is not None
        # depth_profile = depth_profile_list.get_default_video_stream_profile()
        # assert depth_profile is not None
        # print("depth profile: ", depth_profile)
        # self.config.enable_stream(depth_profile)
        #
        # hw_d2c_profile_list = self.orb_pipeline.get_d2c_depth_profile_list(self.color_profile, OBAlignMode.HW_MODE)
        # print("hw d2c depth profile: ", hw_d2c_profile_list)
        # hw_d2c_profile = hw_d2c_profile_list[0]
        # print("hw_d2c_profile: ", hw_d2c_profile)
        # # Enable the depth and color streams
        # self.config.enable_stream(hw_d2c_profile)
        #
        # self.orb_pipeline.enable_frame_sync()
        # self.orb_pipeline.start(self.config)

    def get_rgb(self, frame):
        # frame = self.orb_pipeline.wait_for_frames(100)
        color_frame = frame.get_color_frame()
        if color_frame is None:
            return None
        # covert to RGB format
        image = frame_to_bgr_image(color_frame)
        image = cv2.resize(
            image, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )
        return image
        # return cv2.resize(image, (576, 640), interpolation=cv2.INTER_AREA)

    def get_depth(self, frame):
        # frames = self.orb_pipeline.wait_for_frames(100)
        depth_frame = frame.get_depth_frame()
        depth_format = depth_frame.get_format()
        assert depth_format == OBFormat.Y16, "depth format is not Y16"
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        scale = depth_frame.get_depth_scale()

        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_data = depth_data.reshape((height, width))

        depth_data = depth_data.astype(np.float32) * scale
        depth_data = np.where(
            (depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0
        )
        depth_data = depth_data.astype(np.uint16)
        depth_data = cv2.resize(
            depth_data, (self.width, self.height), interpolation=cv2.INTER_NEAREST
        )
        # Apply temporal filtering
        # depth_data = self.temporal_filter.process(depth_data)
        # print(depth_data.shape)
        return depth_data

        center_y = int(height / 2)
        center_x = int(width / 2)
        # center_distance = depth_data[center_y, center_x]

        # current_time = time.time()
        # if current_time - last_print_time >= PRINT_INTERVAL:
        #     print("center distance: ", center_distance)
        #     last_print_time = current_time

        depth_image = cv2.normalize(
            depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        return cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

    def get_sensors(self):
        """
        Prompts the device to output a single frame of the sensor data.
        Output has the following format: `{'time': timestamp, 'rgb': rgb_vals, 'd': depth_vals}`

        Returns:
        -------
        - `sensor_data` (dict): Sensor data in the format `{'time': float, 'rgb': NDArray[uint16], 'd': NDArray[uint16]}`.
        """
        # get all data from all topics
        frame = self.orb_pipeline.wait_for_frames(10)
        if frame is None:
            return None
        frame = self.align_filter.process(frame)
        if frame is None:
            return None
        frame = frame.as_frame_set()
        if frame is None:
            return None
        rgb = self.get_rgb(frame)
        if rgb is None:
            return None
        d = self.get_depth(frame)
        if d is None:
            return None
        # print(d.shape)
        timestamp = time.time()
        return {"time": timestamp, "rgb": rgb, "d": d}

    @staticmethod
    def get_devices(
        amount, height: int = 512, width: int = 512, **kwargs
    ) -> list["Azure"]:
        context = Context()
        context.enable_multi_device_sync(60000)
        device_list = context.query_devices()
        print(device_list.get_count())
        assert device_list.get_count() != 0, "There is no ORB Camera"
        assert amount <= device_list.get_count(), "device index out of device list"
        orb_device_list = []
        for device_index in range(amount):
            print("device_index: ", device_index)
            orb_device = device_list.get_device_by_index(device_index)
            orb_device_list.append(ORBBEC(orb_device, device_index, height, width))
        return orb_device_list

    def close(self):
        self.stop()

    def stop(self):
        self.orb_pipeline.stop()
        return True


if __name__ == "__main__":
    # ctx = Context()
    # device_list = ctx.query_devices()
    # amount = device_list.get_count()
    #
    # print(amount)
    #
    # pipelines: List[Pipeline] = []
    # configs: List[Config] = []
    #
    # for device_index in range(amount):
    #     orb_device = device_list.get_device_by_index(device_index)
    #
    #     sync_config = orb_device.get_multi_device_sync_config()
    #
    #     config = Config()
    #     pipeline = Pipeline(orb_device)
    #     profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    #
    #     color_profile: VideoStreamProfile = (
    #         profile_list.get_default_video_stream_profile()
    #     )
    #     config.enable_stream(color_profile)
    #
    #     profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    #     depth_profile = profile_list.get_default_video_stream_profile()
    #     config.enable_stream(depth_profile)
    #
    #     pipelines.append(pipeline)
    #     configs.append(config)

    device_list = ORBBEC.get_devices(2, 180, 320)
    for device in device_list:
        device._setup_connect()

    time.sleep(3)

    while True:
        try:
            for device in device_list:
                data = device.get_sensors()
                # if device.name == "ORB_1" and data is not None:

                if data is not None:
                    # print(device.name, data['rgb'].shape, data['d'].shape)
                    cv2.imshow(f"{device.name}Color Viewer", data["rgb"])
                    key = cv2.waitKey(1)
                    cv2.imshow(f"{device.name}Depth Viewer", data["d"])
                    key = cv2.waitKey(1)
        except KeyboardInterrupt:
            break
