import abc
import logging
import cv2

from utils import compat

logger = logging.getLogger(__name__)

class CameraAdapter(abc.ABC):
    """Abstract base class for camera adapters."""

    @abc.abstractmethod
    def connect(self):
        # type: () -> bool
        """Connects to the camera. Returns True if successful, False otherwise."""
        pass

    @abc.abstractmethod
    def capture_frame(self):
        # type: () -> any
        """Captures a single frame from the camera. Returns None if capture fails."""
        pass

    @abc.abstractmethod
    def release(self):
        # type: () -> None
        """Releases the camera resources."""
        pass

    @abc.abstractmethod
    def get_fps(self):
        # type: () -> float
        """Returns the camera's frames per second (FPS)."""
        pass

    @abc.abstractmethod
    def get_resolution(self):
        # type: () -> tuple
        """Returns the camera's resolution as (width, height)."""
        pass


class USBCameraAdapter(CameraAdapter):
    """Concrete implementation of CameraAdapter for USB cameras using OpenCV."""

    def __init__(self, camera_index=0, backend=compat.CAP_ANY):
        # type: (int, int) -> None
        """
        Initializes the USB camera adapter.
        
        Args:
            camera_index: The index of the camera (default is 0).
            backend: The OpenCV capture backend (e.g., cv2.CAP_V4L2, cv2.CAP_DSHOW).
        """
        self.camera_index = camera_index
        self.backend = backend
        self.cap = None

    def connect(self):
        # type: () -> bool
        """Connects to the USB camera."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index, self.backend)
            if not self.cap.isOpened():
                logger.error("Failed to open camera with index {0}".format(self.camera_index))
                return False
            logger.info("Successfully connected to camera with index {0}".format(self.camera_index))
            return True
        except Exception as e:
            logger.error("Error connecting to camera: {0}".format(e))
            return False

    def capture_frame(self):
        # type: () -> any
        """Captures a frame from the USB camera."""
        if self.cap is None or not self.cap.isOpened():
            logger.warning("Camera is not connected. Call connect() first.")
            return None

        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to read frame from camera.")
            return None

        return frame

    def release(self):
        # type: () -> None
        """Releases the USB camera."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            logger.info("Camera released.")

    def get_fps(self):
        # type: () -> float
        """Gets the FPS of the USB camera."""
        if self.cap is not None and self.cap.isOpened():
            return self.cap.get(compat.CAP_PROP_FPS)
        return 0.0

    def get_resolution(self):
        # type: () -> tuple
        """Gets the resolution of the USB camera."""
        if self.cap is not None and self.cap.isOpened():
            width = int(self.cap.get(compat.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(compat.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (0, 0)
