"This file contains the abstract classes necessary for all types of devices for recording."

import shutil
from abc import ABC, abstractmethod
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Event, Process
from multiprocessing.managers import BaseManager
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep
from typing import Any, Generic, List, Optional, Type, TypeVar


class RecordingDevice(ABC):
    """
    This class acts as an interface for all external devices (cameras and microphones)
    used for recording. All instances of `RecordingDevice` can be connected to (`device.connect()`)
    and disconnected from (`device.close()`). Additionally, each concrete subclass of
    RecordingDevice contains the function `get_devices`, which allows to easily find all/a certain
    amount of connected devices of that subclass and returns a list of instances.
    """

    def __init__(self, device_id: str, name: Optional[str] = None, **kwargs) -> None:
        """
        Instantiates a device.

        Parameters:
            device_id (str): Id used for connecting with the device.
            name (Optional[str]): Name of the device. Default: "device_`device_id`"
        """

        self.device_id = device_id
        self.formats: list[str] = []
        self.name = name if name else f"device_{device_id}"

    def connect(self) -> bool:
        """
        Connects to this instance.

        Returns:
            success (bool): Indicates a successful connection.
        """
        print(f"Connecting to {self.name}: ", end="")
        try:
            self._setup_connect()
        except Exception as e:
            print("Failed with exception: ", e)
            self._failed_connect()
            return False
        print("Success")
        return True

    @abstractmethod
    def _setup_connect(self):
        "This method contains the connection logic and has to be overwritten by subclass."

    def _failed_connect(self):
        """
        This method performs the logic, in case of a connection failure.
        It does not have to be overwritten.
        """

    @abstractmethod
    def close(self) -> bool:
        """
        Closes the connection to this instance. Is overwritten by subclass.

        Returns:
            success (bool): Indicates a successful disconnection.
        """

    @staticmethod
    @abstractmethod
    def get_devices(
        amount: int, device_type: str = "recording", **kwargs
    ) -> list["RecordingDevice"]:
        """
        Finds and returns specific amount of instances of this class. Is overwritten by subclass.

        Parameters:
            amount (int): Maximum amount of instances to be found.
                          Leaving out `amount` may return all instances (not always).
            type (str): Type of recording device. Default: `"recording"`
            **kwargs: Arbitrary keyword arguments for initializing devices.

        Returns:
        --------
        - `devices` (list): List of found devices. If no devices are found, `[]` is returned.
        """
        print(
            f"Looking for {'up to ' + str(amount) if amount!=-1 else 'all'} {device_type} devices."
        )

    def get_formats(self) -> list[str]:
        """
        Returns the list of formats that this device stores.

        Returns:
            formats (list[str]): List of saved formats.
        """
        return self.formats


# Frame Recording


class DiscreteDevice(RecordingDevice):
    """
    This class is an interface for *discrete* recording devices, which are prompted to output their
    sensor data in single frames (`device.get_sensors()`).

    Additionally, this class inherits from `RecordingDevice`, so its functionality is also included.
    """

    def __init__(
        self,
        device_id: str,
        name: Optional[str] = None,
        start_frame_latency: int = 0,
        **kwargs,
    ) -> None:
        """
        Instantiates a discrete device.

        Parameters:
            device_id (str): Id used for connecting with the device.
            name (Optional[str]): Name of the device. Default: "discrete_device_`device_id`"
            start_frame_latency (int): Number of frames to skip at the start of the recording.
        """
        super().__init__(
            device_id=device_id,
            name=name if name else f"discrete_device_{device_id}",
            **kwargs,
        )
        self.timestamps = []
        self.start_frame_latency = start_frame_latency

    @abstractmethod
    def store_last_frame(self, directory: Path, filename: str = None):
        """
        Stores the last frame received by the device at a given directory under the given title.

        Parameters:
            directory (str): Directory, where last frame should be stored.
            filename (str): Title of the frame. If none is given, the frame is stored with
                            the current timestamp as title.
        """

    @abstractmethod
    def get_sensors(self) -> dict[str, Any]:
        """
        Prompts the device to output a single frame of the sensor data. Is overwritten by subclass.

        Format of the returned sensor data: `{'time': timestamp, sensor: sensor_vals, ...}`

        Returns:
            sensor_data (dict): Sensor data in the format as described above.
        """

    @staticmethod
    @abstractmethod
    def get_devices(
        amount: int, device_type: str = "discrete", **kwargs
    ) -> list["DiscreteDevice"]:
        super(DiscreteDevice, DiscreteDevice).get_devices(
            amount, device_type=device_type, **kwargs
        )


# Coninuous Recording


class ContinuousDevice(RecordingDevice):
    """
    This class is an interface for *continuous* recording devices, which record their sensory data
    continuously and return the entire recording at the end (instead of frame by frame).
    They can be prompted to start (`device.start_recording()`), stop (`device.stop_recording()`),
    store (`device.store_recording(file_name)`) and delete (`device.delete_recording()`) the
    recording.

    Additionally, this class inherits from `RecordingDevice`, so its functionality is also included.
    """

    def __init__(self, device_id: str, name: Optional[str] = None, **kwargs) -> None:
        """
        Instantiates a continuous device.

        Parameters:
            device_id (str): Id used for connecting with the device.
            name (Optional[str]): Name of the device. Default: "continuous_device_`device_id`"
        """
        super().__init__(
            device_id=device_id,
            name=name if name else f"continuous_device_{device_id}",
            **kwargs,
        )

    @abstractmethod
    def start_recording(self) -> bool:
        """
        Starts the recording on the device (perhaps starts recording process or sends HTTP request).
        Is overwritten by subclass.
        """

    @abstractmethod
    def stop_recording(self) -> bool:
        "Stops the recording on the device. Is overwritten by subclass."

    @abstractmethod
    def store_recording(
        self,
        directory: Path,
        filename: Optional[str] = None,
        timestamps: Optional[list] = None,
    ) -> bool:
        """
        Stores the recording at given directory with (optional) given filename.
        Is overwritten by subclass.
        """

    @abstractmethod
    def delete_recording(self) -> bool:
        "Discards the recording. Is overwritten by subclass."

    @staticmethod
    @abstractmethod
    def get_devices(
        amount: int, device_type: str = "continuous", **kwargs
    ) -> list["ContinuousDevice"]:
        super(ContinuousDevice, ContinuousDevice).get_devices(
            amount, device_type=device_type, **kwargs
        )


# author of class: TimWindecker

T = TypeVar("T", bound=DiscreteDevice)


class AsynchronousDevice(ContinuousDevice, Generic[T]):
    """
    This class is a wrapper for a DiscreteDevice to act as a ContinuousDevice by running it in a
    separate process.

    NOTE: To ensure that all of the output files for each frame are stored correctly, one must set
    `self.formats` to the strings that represent the distinctive file endings, e.g.
    `self.formats = ["_img.png", "_depth.png"]`.
    The function `store_last_frame` of the wrapped device must therefore store the frames with
    those exact suffixes itself.
    """

    def __init__(self, device_class: Type[T], capture_interval=0, **kwargs) -> None:
        super().__init__(**kwargs)

        # Create event to signal that the process should stop
        self._stop = Event()

        class DeviceManager(BaseManager):
            "Custom manager to share the device object between processes."

        V = TypeVar("V")

        @dataclass
        class Container(Generic[V]):
            "Simple container to hold a value that can be shared between processes."
            value: V

            def get_value(self) -> V:
                "Returns the value stored in the container."
                return self.value

        DeviceManager.register(
            "Device",
            device_class,
            exposed=(
                "_setup_connect",
                "_failed_connect",
                "close",
                "store_last_frame",
                "get_formats",
            ),
        )
        DeviceManager.register("CaptureInterval", Container[int])
        DeviceManager.register("TempDirectoryPath", Container[str])

        # Start manager and create shared objects
        self._manager = DeviceManager()
        self._manager.start()
        self._proxy_device = self._manager.Device(**kwargs)
        self._proxy_capture_interval = self._manager.CaptureInterval(capture_interval)
        self._temp_dir = TemporaryDirectory(prefix="camera_tmp_")
        self._proxy_temp_dir_path = self._manager.TempDirectoryPath(self._temp_dir.name)
        assert Path(self._proxy_temp_dir_path.get_value()).exists()

        self._process = None

    def _setup_connect(self):
        self._proxy_device._setup_connect()

    def _failed_connect(self):
        self._proxy_device._failed_connect()

    def close(self):
        super().close()
        self._proxy_device.close()

    @staticmethod
    def get_devices(amount: int, **kwargs) -> list["T"]:
        """
        Finds and returns specific amount of instances of this class.
        Note: this calls the `get_devices` method of the wrapped `DiscreteDevice` class.
        """
        return T.get_devices(amount, **kwargs)

    def start_recording(self) -> bool:
        super().start_recording()

        # Clear temp folder
        self._clear_temp()

        # Start process
        self._process = Process(
            target=self._continuous_capture,
            args=(
                self._stop,
                self._proxy_device,
                self._proxy_capture_interval,
                self._proxy_temp_dir_path,
            ),
            daemon=True,
        )
        self._stop.clear()
        self._process.start()

        return True

    def stop_recording(self) -> bool:
        super().stop_recording()
        self._stop.set()
        self._process.join()
        return True

    def store_recording(
        self, directory: Path, filename: str = None, timestamps: List[float] = None
    ):
        """
        Stores the previously recorded frames that where received by the device. When passing
        `timestamps`, the closest previous frame for each timestamp is saved. This can lead to
        duplicates or frames being ignored depending on the frequency of the timestamps.

        Parameters:
            directory (Path): Directory, where last frame should be stored.
            filename (str): Unused parameter, as the frames are stored with their step number.
            timestamps (List[float]): Timesamps of the required images as list of seconds since the
                                      Unix epoch. If the exact timestamp is not available the
                                      closest image before is selected.
        """

        if timestamps is None:
            # If no timestamps are given, copy all images
            temp_dir = Path(self._proxy_temp_dir_path.get_value())
            for frame_format in self._proxy_device.get_formats():
                for frame_path in temp_dir.glob(f"*{frame_format}"):
                    destination_path = directory / f"{frame_path.name}"
                    shutil.copy(frame_path, destination_path)
        else:
            # Extract map from timestamps to image paths
            temp_dir = Path(self._proxy_temp_dir_path.get_value())
            for frame_format in self._proxy_device.get_formats():
                timestamp_path_map = []
                for frame_path in temp_dir.glob(f"*{frame_format}"):
                    try:
                        timestamp_str = frame_path.name.removesuffix(frame_format)
                        timestamp = datetime.fromisoformat(timestamp_str)
                        timestamp_unix = timestamp.timestamp()
                        timestamp_path_map.append((timestamp_unix, frame_path))
                    except ValueError:
                        print(
                            f"Skipping file with invalid timestamp: {frame_path.name}"
                        )

                # Sort map by timestamps
                timestamp_path_map.sort(key=lambda t: t[0])

                # Map desired timestamps to paths
                sorted_timetamps = [t for t, _ in timestamp_path_map]
                print(
                    f"{self.name}'s frame rate: {len(sorted_timetamps) / (sorted_timetamps[-1] - sorted_timetamps[0]):.2f} Hz"
                )
                desired_indices = [
                    max(bisect_right(sorted_timetamps, t) - 1, 0) for t in timestamps
                ]  # Find index of element <= t
                desired_paths = [timestamp_path_map[i][1] for i in desired_indices]

                # Copy images
                for i, path in enumerate(desired_paths):
                    destination_path = directory / f"{i}{frame_format}"
                    shutil.copy(path, destination_path)

        # Clear temp folder
        self._clear_temp()

    def delete_recording(self):
        self._clear_temp()

    def _clear_temp(self):
        for file in Path(self._proxy_temp_dir_path.get_value()).rglob("*"):
            file.unlink()

    def _continuous_capture(
        self, stop_event, device, capture_interval, dir_path
    ) -> None:

        directory = Path(dir_path.get_value())

        while not stop_event.is_set():

            # Capture and store
            device.store_last_frame(directory)  # store frame in temp dir with timestamp

            # Sleep
            sleep(capture_interval.get_value())
