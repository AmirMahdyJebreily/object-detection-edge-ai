import abc
import logging
from typing import Tuple, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class CameraAdapter(abc.ABC):
    """Abstract base class for camera adapters."""

    @abc.abstractmethod
    def connect(self) -> bool:
        """Connects to the camera. Returns True if successful, False otherwise."""
        pass

    @abc.abstractmethod
    def capture_frame(self) -> Optional[np.ndarray]:
        """Captures a single frame from the camera. Returns None if capture fails."""
        pass

    @abc.abstractmethod
    def release(self) -> None:
        """Releases the camera resources."""
        pass

    @abc.abstractmethod
    def get_fps(self) -> float:
        """Returns the camera's frames per second (FPS)."""
        pass

    @abc.abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """Returns the camera's resolution as (width, height)."""
        pass


class USBCameraAdapter(CameraAdapter):
    """Concrete implementation of CameraAdapter for USB cameras using OpenCV."""

    def __init__(self, camera_index: int = 0, backend: int = cv2.CAP_ANY):
        """
        Initializes the USB camera adapter.
        
        Args:
            camera_index: The index of the camera (default is 0).
            backend: The OpenCV capture backend (e.g., cv2.CAP_V4L2, cv2.CAP_DSHOW).
        """
        self.camera_index = camera_index
        self.backend = backend
        self.cap: Optional[cv2.VideoCapture] = None

    def connect(self) -> bool:
        """Connects to the USB camera."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index, self.backend)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera with index {self.camera_index}")
                return False
            logger.info(f"Successfully connected to camera with index {self.camera_index}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to camera: {e}")
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """Captures a frame from the USB camera."""
        if self.cap is None or not self.cap.isOpened():
            logger.warning("Camera is not connected. Call connect() first.")
            return None

        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to read frame from camera.")
            return None

        return frame

    def release(self) -> None:
        """Releases the USB camera."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            logger.info("Camera released.")

    def get_fps(self) -> float:
        """Gets the FPS of the USB camera."""
        if self.cap is not None and self.cap.isOpened():
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 0.0

    def get_resolution(self) -> Tuple[int, int]:
        """Gets the resolution of the USB camera."""
        if self.cap is not None and self.cap.isOpened():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (0, 0)
