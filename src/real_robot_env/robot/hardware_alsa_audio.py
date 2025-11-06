"Because pyaudio struggles with collecting a single audio recording with no skipping and artifacts, this will be an alternative implementation using pyalsaaudio."

import array
import tempfile
import time
import wave
from multiprocessing import Event, Process, Queue

import alsaaudio
import numpy as np

from real_robot_env.robot.hardware_devices import ContinuousDevice


class AudioInput:
    def __init__(
        self,
        frames_per_buffer: int = 1024,
        channels: int = 2,
        rate: int = 44100,
        audio_format: int = alsaaudio.PCM_FORMAT_S16_LE,
        sample_width: int = 2,
    ):
        self.frames_per_buffer = frames_per_buffer
        self.channels = channels
        self.rate = rate
        self.audio_format = audio_format
        self.sample_width = sample_width

    def __str__(self) -> str:
        return f"[AudioInput with frames_per_buffer={self.frames_per_buffer}, channels={self.channels}, rate={self.rate}, audio_format={self.audio_format}]"


class AudioInterface(ContinuousDevice):
    """
    Audio Interface, that runs on a separate process (how it always should be as opposed to being on a separate thread).

    This is implemented because the audio recordings glitch out due to the buffer overloading, when the audio recording happens on the same process as the main data collection loop.
    """

    def __init__(self, device_id: str, name=None, audio_input=AudioInput()) -> None:
        super().__init__(device_id, name if name else f"AudioInterface_{device_id}")

        self.device = None
        self.timestamp = None
        self._stop = Event()
        self._is_stopped = Event()
        self.storage_queue = None
        self.format = ".wav"
        self.audio_input = audio_input

    def _setup_connect(self):
        self.device = alsaaudio.PCM(
            alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NONBLOCK, device=self.device_id
        )
        self.device.setchannels(self.audio_input.channels)
        self.device.setrate(self.audio_input.rate)
        self.device.setformat(self.audio_input.audio_format)
        self.device.setperiodsize(self.audio_input.frames_per_buffer)

    def close(self):
        self._process.join()
        self.device.close()
        self.device = None
        return True

    def start_recording(self) -> bool:
        self.storage_queue = Queue()
        self._process = Process(
            target=self._continuous_capture,
            args=(self._stop, self._is_stopped, self.device, self.storage_queue),
            daemon=True,
        )
        self._stop.clear()
        self._is_stopped.clear()
        self.timestamp = time.time()
        self._process.start()

        return True

    def stop_recording(self) -> bool:
        print("> stop the recording.")
        self._stop.set()
        self._is_stopped.wait()
        return True

    def store_recording(self, directory, filename=None, _=None):
        # convert queue to list
        frames = self.storage_queue.get()
        self.storage_queue.close()

        # store list
        # with open(str(directory / filename) + self.format, 'wb') as f:
        #     for frame in frames:
        #         f.write(frame)

        filename = filename if filename else f"{self.name}_recording"
        print(f"Attempting to store {len(frames)} frames for {self.name}")
        wf = wave.open(str(directory / filename) + self.format, "wb")
        wf.setnchannels(self.audio_input.channels)
        wf.setsampwidth(2)  # TODO for 16 bit its 2 bytes..
        wf.setframerate(self.audio_input.rate)
        wf.writeframes(b"".join(frames))
        wf.close()
        print(f"Frames stored for {self.name}")
        return True

    def delete_recording(self):
        self.storage_queue.close()
        return True

    @staticmethod
    def _continuous_capture(
        stop_event, is_stopped_event, device, storage_queue: Queue
    ) -> None:
        is_stopped_event.clear()
        frames = []
        while not stop_event.is_set():
            l, data = device.read()

            if l < 0:
                print("Capture buffer overrun! Continuing nonetheless ...")
            elif l:
                frames.append(data)
                time.sleep(0.001)
        storage_queue.put(frames, block=False)
        is_stopped_event.set()
        # print("Stream closed")

    @staticmethod
    def print_found_devices():
        "Prints all audio devices connected to the PC and info about them on the console."
        for device_name in alsaaudio.pcms(alsaaudio.PCM_CAPTURE):
            print(f"{device_name}")
            # try:
            #     d = alsaaudio.PCM(device=device_name)
            #     print(d.info())
            # except: pass

    @staticmethod
    def get_specific_devices(
        iface_name: str, amount: int = -1
    ) -> list["AudioInterface"]:
        """
        Finds and returns a list of audio interfaces with a specific name.

        Parameters:
        ----------
        - `iface_name` (str): Name of audio interfaces.
        - `amount` (int): Maximum amount of instances to be found. Leaving out `amount` or `amount = -1` returns all instances.

        Returns:
        --------
        - `audio_iface` (list[AsyncAudioInterface]): List of found audio interfaces. If no devices are found, `[]` is returned.
        """
        getter = AudioInterface.__construct_getter(
            lambda device_info: device_info["name"].startswith(iface_name)
        )
        return getter(amount)

    @staticmethod
    def get_devices(amount: int = -1, **kwargs) -> list["AudioInterface"]:
        """
        Finds and returns specific amount of instances of this class. Suitable devices are determined by `input_channels > 0`.

        Better practice: Use `AsyncAudioInterface.get_specific_devices(iface_name)` instead!

        Parameters:
        ----------
        - `amount` (int): Maximum amount of instances to be found. Leaving out `amount` or `amount = -1` returns all instances.
        - `**kwargs`: Arbitrary keyword arguments.

        Returns:
        --------
        - `devices` (list[AsyncAudioInterface]): List of found devices. If no devices are found, `[]` is returned.
        """
        super(AudioInterface, AudioInterface).get_devices(
            amount, type="AudioInterface", **kwargs
        )
        getter = AudioInterface.__construct_getter(
            lambda device_info: int(device_info["channels"]) > 0
            and device_info["name"] != "default"
        )
        return getter(amount)

    @staticmethod
    def __construct_getter(condition):
        def get_cond_devices(amount: int = -1):
            audio_ifaces = []
            counter = 0
            for device_name in alsaaudio.pcms(alsaaudio.PCM_CAPTURE):
                if amount != -1 and counter >= amount:
                    break
                try:
                    test_pcm = alsaaudio.PCM(device=device_name)
                    device_info = test_pcm.info()
                except:
                    continue
                if condition(device_info):
                    audio_input = AudioInput(
                        channels=int(device_info["channels"]),
                        rate=int(device_info["rate"]),
                        audio_format=device_info["format"],
                        sample_width=int(device_info["physical_bits"] / 8),
                    )
                    audio_ifaces.append(
                        AudioInterface(device_id=device_name, audio_input=audio_input)
                    )
                    counter += 1
            return audio_ifaces

        return get_cond_devices

    def __str__(self) -> str:
        return f"AudioInterface: {self.name}"
