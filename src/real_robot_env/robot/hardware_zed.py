import time
import cv2
import numpy as np
import logging

from real_robot_env.robot.hardware_cameras import DiscreteCamera

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZED(DiscreteCamera):
    """
    Wrapper that implements boilerplate code for ZED cameras from Stereolabs.
    Includes robust connection handling for Linux/USB bandwidth issues.
    """

    def __init__(
        self,
        device_id,
        name=None,
        height=720, # Tipp: Für 3 Kameras nutze height=376
        width=1280, # Tipp: Für 3 Kameras nutze width=672
        fps=30,     # Tipp: Für 3 Kameras nutze fps=15
        depth_mode="PERFORMANCE", # Tipp: Für 3 Kameras nutze "NONE"
        start_frame_latency=0,
    ):
        super().__init__(
            device_id,
            name if name else f"ZED_{device_id}",
            height,
            width,
            start_frame_latency,
        )
        self.fps = fps
        self.depth_mode = depth_mode
        self.zed = None
        self.runtime_parameters = None

    def _setup_connect(self):
        super()._setup_connect()

        try:
            import pyzed.sl as sl
        except ImportError:
            raise ImportError(
                "pyzed package not found. Please install the ZED SDK and pyzed package."
            )

        # Create a Camera object
        self.zed = sl.Camera()

        # Create InitParameters object
        init_params = sl.InitParameters()

        # --- ÄNDERUNG 1: Robuste Verbindung über InputType (wie im Test-Skript) ---
        input_type = sl.InputType()
        # Wir stellen sicher, dass die ID ein int ist
        input_type.set_from_serial_number(int(self.device_id))
        init_params.input = input_type

        # --- ÄNDERUNG 2: Timeout erhöhen & Verbose Mode (gegen "Camera Opening Timeout") ---
        init_params.open_timeout_sec = 60.0  # Gibt dem USB-Bus Zeit (WICHTIG!)
        init_params.sdk_verbose = 1          # Zeigt Logs im Terminal, falls es hängt
        init_params.async_grab_camera_recovery = True # Versucht Frame-Recovery bei USB-Hickups

        # Set resolution
        if self.width == 2208 and self.height == 1242:
            init_params.camera_resolution = sl.RESOLUTION.HD2K
        elif self.width == 1920 and self.height == 1080:
            init_params.camera_resolution = sl.RESOLUTION.HD1080
        elif self.width == 1280 and self.height == 720:
            init_params.camera_resolution = sl.RESOLUTION.HD720
        elif self.width == 672 and self.height == 376:
            init_params.camera_resolution = sl.RESOLUTION.VGA
        else:
            # Fallback
            init_params.camera_resolution = sl.RESOLUTION.HD720
            logger.warning(
                f"Resolution {self.width}x{self.height} not directly supported. "
                "Using HD720 and will resize."
            )

        # Set FPS
        init_params.camera_fps = self.fps

        # Set depth mode mapping
        depth_modes = {
            "NONE": sl.DEPTH_MODE.NONE,
            "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
            "QUALITY": sl.DEPTH_MODE.QUALITY,
            "ULTRA": sl.DEPTH_MODE.ULTRA,
            "NEURAL": sl.DEPTH_MODE.NEURAL,
        }
        # Fallback zu PERFORMANCE, wenn key nicht existiert
        self.selected_depth_mode = depth_modes.get(
            self.depth_mode, sl.DEPTH_MODE.PERFORMANCE
        )
        init_params.depth_mode = self.selected_depth_mode

        # Set units to millimeters
        init_params.coordinate_units = sl.UNIT.MILLIMETER

        # Open the camera
        logger.info(f"Attempting to open ZED {self.device_id} (Timeout 60s)...")
        err = self.zed.open(init_params)
        
        if err != sl.ERROR_CODE.SUCCESS:
            # Versuche einen Reboot im Fehlerfall (wie im Debug-Skript)
            logger.warning(f"Initial open failed for {self.device_id}: {err}. Trying reboot logic...")
            sl.Camera.reboot(int(self.device_id))
            time.sleep(5)
            raise RuntimeError(f"Failed to open ZED camera {self.device_id}: {err}")

        # Create runtime parameters
        self.runtime_parameters = sl.RuntimeParameters()
        
        # Optimization: If no depth needed, disable it in runtime too
        if self.depth_mode == "NONE":
            self.runtime_parameters.enable_depth = False

        # Create Mat objects to store images
        self.image = sl.Mat()
        self.depth = sl.Mat()

        logger.info(f"ZED camera {self.device_id} connected successfully")

    def _get_sensors(self):
        """
        Prompts the device to output a single frame of the sensor data.
        """
        if self.zed is None:
            raise RuntimeError(f"Not connected to {self.name}")

        import pyzed.sl as sl

        # Grab a new frame
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            timestamp = time.time()

            # Retrieve left image (RGB)
            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            rgb = self.image.get_data()[:, :, :3]  # Remove alpha channel

            # --- ÄNDERUNG 3: Depth nur abrufen, wenn wir nicht im "NONE" Mode sind ---
            if self.depth_mode != "NONE":
                self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
                depth = self.depth.get_data()
            else:
                # Return empty depth array or zeros if mode is NONE
                # This prevents crashes when saving bandwidth
                depth = np.zeros((self.height, self.width), dtype=np.float32)

            return {"time": timestamp, "rgb": rgb, "d": depth}
        else:
            # Log warning instead of hard crash allows temporary frame drops
            # raise RuntimeError(f"Failed to grab frame from ZED camera {self.device_id}")
            logger.warning(f"Frame drop on ZED {self.device_id}")
            return None # Or handle gracefully in main loop

    def get_camera_information(self):
        """
        Get camera calibration and information.
        """
        if self.zed is None:
            raise RuntimeError("Camera not connected")

        import pyzed.sl as sl

        camera_info = self.zed.get_camera_information()
        calib_params = camera_info.camera_configuration.calibration_parameters

        return {
            "serial_number": camera_info.serial_number,
            "camera_model": str(camera_info.camera_model),
            "fx": calib_params.left_cam.fx,
            "fy": calib_params.left_cam.fy,
            "cx": calib_params.left_cam.cx,
            "cy": calib_params.left_cam.cy,
            "baseline": calib_params.get_camera_baseline(),
        }

    def close(self):
        """Close the camera connection."""
        success = super().close()
        if self.zed is not None:
            self.zed.close()
            self.zed = None
        return success

    @staticmethod
    def get_devices(
        amount=-1, height: int = 720, width: int = 1280, **kwargs
    ) -> list["ZED"]:
        super(ZED, ZED).get_devices(
            amount, height=height, width=width, device_type="ZED", **kwargs
        )

        try:
            import pyzed.sl as sl
        except ImportError:
            logger.error(
                "pyzed package not found. Please install the ZED SDK and pyzed package."
            )
            return []

        # Get list of connected ZED cameras
        cam_list = sl.Camera.get_device_list()
        cams = []
        counter = 0

        for device in cam_list:
            if amount != -1 and counter >= amount:
                break
            # Pass arguments correctly to __init__
            cam = ZED(device.serial_number, height=height, width=width, **kwargs)
            cams.append(cam)
            counter += 1

        return cams
