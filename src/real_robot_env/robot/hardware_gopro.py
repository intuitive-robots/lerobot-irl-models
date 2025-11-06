"GoPro Hero 9 Black camera implementation for the Real Robot Environment."

import time
from multiprocessing import Process

import netifaces as ni
from goprocam import GoProCamera, constants

from real_robot_env.robot.hardware_cameras import ContinuousCamera


class GoPro(ContinuousCamera):
    """
    This implementation only works with **GoPro Hero 9 Black** and acts as a workaround for the
    following issue: https://github.com/KonradIT/gopro-py-api/issues/184

    TLDR: When sending a request to the GoPro to record a video/take a photo, the GoPro disconnects
    the USB connection.

    Usage:
    1. Connect GoPro to PC through USB.
    2. Find GoPro interface and IP and create `GoPro` instance (`GoPro.get_devices()`).
    3. Start recording (`gopro.start_recording()`).
       - After receiving the request, the GoPro disconnects until recording is stopped.
    4. After calling `gopro.stop_recording()`, manually stop recording on GoPro (you have 30s).
       - GoPro reconnects to computer, now with a different interface and IP.
       - `stop_recording` waits until it finds a suitable interface and IP and reconnects to it.
    5. Store (`gopro.store_recording(filename)`) or discard (`gopro.delete_recording()`) recording.
    6. At the end of use, close the connection (`gopro.close()`).

    Also only works, when only one GoPro is connected, due to bug above.
    """

    def __init__(
        self,
        device_id: str,
        name=None,
        height: int = 512,
        width: int = 512,
        default_fps: float = 20,
        cut_ending=True,
    ) -> None:
        super().__init__(
            device_id,
            name if name else f"GoPro_{device_id}",
            height=height,
            width=width,
            default_fps=default_fps,
            cut_ending=cut_ending,
        )
        self.formats = [".mp4"]
        self.gopro_device = None

        self.rec_process = None

    def _setup_connect(self):
        self.gopro_device = GoProCamera.GoPro(
            ip_address=GoProCamera.GoPro.getWebcamIP(self.device_id),
            camera=constants.gpcontrol,
            webcam_device=self.device_id,
        )
        # self.gopro_device.gpControlSet(constants.Video.FRAME_RATE, constants.Video.FrameRate.FR30)

        # determine latency
        self.latency = 2.351970832  # slightly too low: 2.272919736

    def _failed_connect(self):
        self.gopro_device = None

    def start_recording(self):
        super().start_recording()
        if self.device_id not in ni.interfaces() or self.gopro_device is None:
            return False

        # shoot_video will fail, due to connection to GoPro being lost.
        # Camera will reconnect to USB port after finishing recoring.
        def shoot_video():
            try:
                self.gopro_device.shoot_video(0)
            except Exception as e:
                print(f"Error while shooting video: {e}")
                self.gopro_device = None

        self.rec_process = Process(target=shoot_video)
        self.rec_process.start()
        return True

    def stop_recording(self):
        """
        This method does NOT stop the recording, but waits for manual stopping of the recording
        (see class docstring).
        """
        super().stop_recording()
        if self.gopro_device is None:
            return False
        self.rec_process.join()

        # You have to manually stop the recording! (GoPro Hero 9 Black Bug)
        count = 0
        gopro_iface = self.get_single_interface()
        print()
        print("Please manually stop the recording of the GoPro")
        while not gopro_iface:
            if count >= 30 * 10:  # you have 30 seconds to stop the recording..
                raise RuntimeError(
                    "Camera did not reconnect after recording. "
                    + "Did you forget to manually stop the recording?"
                )
            time.sleep(0.1)
            count += 1
            gopro_iface = self.get_single_interface()
        self.device_id = gopro_iface
        self.connect()
        return True

    def _store_video(self, video_file: str):
        if self.device_id not in ni.interfaces() or self.gopro_device is None:
            return False
        self.gopro_device.downloadLastMedia(custom_filename=video_file)
        return True

    def delete_recording(self):
        if self.device_id not in ni.interfaces() or self.gopro_device is None:
            return False
        self.gopro_device.delete("last")
        return True

    def close(self) -> bool:
        success = super().close()
        self.gopro_device = None
        return success

    @staticmethod
    def get_interfaces(amount=-1):
        """
        Finds and returns GoPro interfaces connected to the computer.
        """
        ifaces = ni.interfaces()
        gopro_ifaces = []
        counter = 0
        for iface in ifaces:
            if amount != -1 and counter >= amount:
                break
            try:
                ip = ni.ifaddresses(iface)[ni.AF_INET][0]["addr"]
                if ip.startswith("172.") and iface.startswith("enx"):
                    gopro_ifaces.append(iface)
                    counter += 1
            except Exception as e:
                continue  # Ignore interfaces that do not have an IP address or are not GoPro interfaces
        return gopro_ifaces

    @staticmethod
    def get_single_interface():
        """
        Finds and returns a single GoPro interface connected to the computer.
        """
        gopro_ifaces = GoPro.get_interfaces(1)
        if gopro_ifaces:
            return gopro_ifaces[0]
        else:
            return None

    @staticmethod
    def get_devices(amount: int = 1, **kwargs) -> list["GoPro"]:
        """
        Finds and returns specific amount of instances of this class.
        Currently, only 1 GoPro is allowed!

        Parameters:
            amount (int): Maximum amount of instances to be found.
                          Currently, only `amount = 1` is allowed.
            fps (int): FPS with which the video is sampled to create the frames.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            devices (list[GoPro]): List containing single found device.
                                   If it isn't found, `[]` is returned.
        """
        assert amount == 1, "Only one GoPro camera supported currently."
        super(GoPro, GoPro).get_devices(amount, device_type="GoPro", **kwargs)
        gopro_iface = GoPro.get_single_interface()
        if gopro_iface:
            return [GoPro(device_id=gopro_iface)]
        return []
