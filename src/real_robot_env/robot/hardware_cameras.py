"This file contains the abstract classes useful for integrating various camera types."

import datetime
import time
from abc import abstractmethod
from multiprocessing import Event, Pipe, Process
from pathlib import Path
from typing import Any, Generic, Optional, Type, TypeVar

import cv2
from moviepy import VideoFileClip

from real_robot_env.robot.hardware_devices import (
    AsynchronousDevice,
    ContinuousDevice,
    DiscreteDevice,
)


class DiscreteCamera(DiscreteDevice):
    """
    This class acts as a generalization for cameras, whose recording is captured frame by frame.

    Additionally, this class inherits from `DiscreteDevice`, so its functionality is also included.
    """

    def __init__(
        self,
        device_id: str,
        name: Optional[str] = None,
        height: int = 512,
        width: int = 512,
        start_frame_latency: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            device_id=device_id,
            name=name if name else f"discrete_cam_{device_id}",
            start_frame_latency=start_frame_latency,
            **kwargs,
        )
        self.formats = [".png"]
        self.height, self.width = height, width

        # Variables for storing frames
        self.reader, self.writer = Pipe(False)
        self.stop_frame_storage_event = Event()
        self.write_process = Process(
            target=self.__store_frames,
            args=[self.reader, self.stop_frame_storage_event],
        )

    @abstractmethod  # Should be overwritten but also called by subclass.
    def _setup_connect(self):
        self.write_process.start()

    def _failed_connect(self):
        """
        When a connection fails, the connection is closed and the process is terminated.
        """
        self.close()

    @abstractmethod
    def _get_sensors(self) -> dict[str, Any]:
        """
        Prompts the camera to output a single frame and returns the *unprocessed* sensor data.
        Is overwritten by subclass.

        Output should have the following format:
        `{'time': POSIX timestamp (like time.time()), 'rgb': rgb_vals, 'd' [opt]: depth_vals}`

        Returns:
            sensor_data (dict): Sensor data in the format as described above.
        """

    def get_sensors(self) -> dict[str, Any]:
        """
        Prompts the camera to output a single frame and returns **processed** sensor data.

        Output should have the following format:
        `{'time': timestamp, 'rgb': rgb_vals, 'd' [opt]: depth_vals}`

        Returns:
            sensor_data (dict): Sensor data in the format as described above.

        """
        sensor_data = self._get_sensors()
        img = sensor_data["rgb"]
        resized_img = cv2.resize(img, (self.width, self.height))
        sensor_data["rgb"] = resized_img
        return sensor_data

    def store_last_frame(self, directory: Path, filename: str = None):
        """
        Stores the last frame received by camera (only the RGB data) as a `self.formats[0]`
        (default: ".png").

        NOTE:
        - If you require a different format, you can overwrite `self.formats` in the subclass.
        - If you want to store additional data, you can overwrite this method in the subclass.

        Parameters:
            directory (Path): Directory, where last frame should be stored.
            filename (str): Title of the frame. If none is given, the frame is stored with
                            the current timestamp as title.
        """
        sensor_data = self.get_sensors()
        img = sensor_data["rgb"]
        self.timestamps.append(sensor_data["time"])
        if filename is None:
            timestamp = datetime.datetime.fromtimestamp(sensor_data["time"])
            filename = str(directory / timestamp.isoformat()) + self.formats[0]
        else:
            filename = str(directory / f"{filename}") + self.formats[0]
        self.writer.send((img, filename))

    @staticmethod
    def __store_frames(reader, stop_frame_storage_event):
        try:
            while not stop_frame_storage_event.is_set():
                if not reader.poll(0.1):
                    continue
                (img, img_path) = reader.recv()
                cvt_color_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    img_path,
                    cvt_color_img,
                )
            print("Stopping frame storage process.")
        finally:
            reader.close()

    @abstractmethod  # Should be overwritten but also called by subclass.
    def close(self) -> bool:
        """
        Closes the connection to this instance and stops the frame storage process.

        Returns:
            success (bool): Indicates a successful disconnection.
        """
        self.stop_frame_storage_event.set()
        self.write_process.join()
        self.write_process.close()
        self.reader.close()
        self.writer.close()
        return True

    @staticmethod
    @abstractmethod  # Should be overwritten but also called by subclass.
    def get_devices(
        amount: int,
        device_type="discrete",
        height: int = 512,
        width: int = 512,
        **kwargs,
    ) -> list["DiscreteCamera"]:
        """
        Finds and returns specific amount of instances of this class. Is overwritten by subclass.

        Parameters:
            amount (int): Maximum amount of instances to be found.
                          Leaving out `amount` may return all instances (not always).
            height (int): Pixel-height of captured frames. Default: `512`
            width (int): Pixel-width of captured frames. Default: `512`
            **kwargs: Arbitrary keyword arguments.

        Returns:
            devices (list): List of found devices. If no devices are found, `[]` is returned.
        """
        print(
            f"Looking for {'up to ' + str(amount) if amount!=-1 else 'all'} {device_type} cameras."
        )


class ContinuousCamera(ContinuousDevice):
    """
    This class acts as a generalization for cameras, whose recording is captured in its entirety.

    It is not recommended to implement a `ContinuousCamera`, but instead to implement a
    DiscreteCamera and use it in an `AsynchronousCamera`, which will then act as a
    `ContinuousCamera`.
    """

    def __init__(
        self,
        device_id: str,
        name: Optional[str] = None,
        height: int = 512,
        width: int = 512,
        default_fps: float = 20,
        cut_ending=True,
        **kwargs,
    ) -> None:
        super().__init__(
            device_id=device_id,
            name=name if name else f"continuous_cam_{device_id}" ** kwargs,
        )
        self.formats = [".mp4"]
        self.height, self.width = height, width
        self.latency = 0.0  # in s
        self.default_fps = default_fps
        self.cut_ending = cut_ending
        self.recording_start = 0.0  # timestamp, where recording actually started
        self.recording_stop = 0.0  # timestamp, where recording should end

        self.frame_extraction_processes = []

    @abstractmethod  # Should be overwritten but also called by subclass.
    def start_recording(self) -> bool:
        self.recording_start = time.time() + self.latency
        return True

    @abstractmethod  # Should be overwritten but also called by subclass.
    def stop_recording(self) -> bool:
        self.recording_stop = time.time()
        return True

    def store_recording(self, directory, filename=None, timestamps=None):
        filename = filename if filename else f"{self.name}_recording"
        video_file = str(directory / filename) + self.formats[0]
        self._store_video(video_file)

        if timestamps:
            dur = timestamps[-1] - timestamps[0]
            print(
                f"duration: {dur}, len(timestamps): {len(timestamps)} => fps: {len(timestamps)/dur}"
            )
            self.__extract_frames_at_timestamps(
                video_file, directory, timestamps, self.recording_start
            )
        else:
            self.__extract_frames(
                video_file,
                directory,
                self.default_fps,
                self.recording_start,
                self.recording_stop,
                self.cut_ending,
            )
        return True

    @abstractmethod
    def _store_video(self, video_file: str):
        pass

    def __extract_frames_at_timestamps(
        self, video_file, directory, timestamps, recording_start
    ):
        def convert_to_images():
            vidcap = cv2.VideoCapture(video_file)
            idx = 0
            for timestamp in timestamps:
                ms_time = max(
                    (timestamp - recording_start) * 1000, 0
                )  # relative video position in ms
                vidcap.set(cv2.CAP_PROP_POS_MSEC, ms_time)
                success, image = vidcap.read()
                if success:
                    resized_img = cv2.resize(image, (self.width, self.height))
                    cv2.imwrite(
                        str(directory / f"{idx}.png"), resized_img
                    )  # save frame as png file
                    idx += 1

        process = Process(target=convert_to_images)
        process.start()
        self.frame_extraction_processes.append(process)

    def __extract_frames(
        self,
        video_file,
        directory,
        fps,
        recording_start,
        recording_stop,
        cut_ending=True,
    ):
        def convert_to_images():
            if cut_ending:
                duration = recording_stop - recording_start  # in seconds
                clip = VideoFileClip(video_file).subclip(0, duration)
            else:
                clip = VideoFileClip(video_file)
            clip.write_images_sequence(
                str(directory / "%d.png"), fps=fps, logger=None
            )  # logger='bar'

        process = Process(target=convert_to_images)
        process.start()
        self.frame_extraction_processes.append(process)

    def close(self) -> bool:
        for process in self.frame_extraction_processes:
            process.join()
        return True

    @staticmethod
    @abstractmethod  # Should be overwritten but also called by subclass.
    def get_devices(
        amount: int,
        device_type="continuous",
        height: int = 512,
        width: int = 512,
        **kwargs,
    ) -> list["DiscreteCamera"]:
        """
        Finds and returns specific amount of instances of this class. Is overwritten by subclass.

        Parameters:
            amount (int): Maximum amount of instances to be found
                          Leaving out `amount` may return all instances (not always).
            height (int): Pixel-height of captured frames. Default: `512`
            width (int): Pixel-width of captured frames. Default: `512`
            **kwargs: Arbitrary keyword arguments.

        Returns:
            devices (list): List of found devices. If no devices are found, `[]` is returned.
        """
        assert amount == -1 or amount > 0, "Amount must be -1 or greater than 0."
        print(
            f"Looking for {'up to ' + str(amount) if amount!=-1 else 'all'} {device_type} cameras."
        )


# author of class: TimWindecker
T = TypeVar("T", bound=DiscreteCamera)


class AsynchronousCamera(AsynchronousDevice, ContinuousCamera, Generic[T]):
    """
    This class is a wrapper for a DiscreteCamera to act as a ContinuousCamera by running it in a
    separate process.
    """

    def __init__(self, camera_class: Type[T], capture_interval=0, **kwargs):
        super().__init__(
            device_class=camera_class, capture_interval=capture_interval, **kwargs
        )

    def _store_video(self, video_file: str):
        raise NotImplementedError
