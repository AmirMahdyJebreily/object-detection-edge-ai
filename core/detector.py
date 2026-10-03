import logging
import threading
import time
import numpy as np

from adapters.camera_adapter import CameraAdapter
from adapters.model_adapter import ModelAdapter, Detection

logger = logging.getLogger(__name__)

class ObjectDetector:
    """
    Main component that orchestrates the camera and model for continuous detection.
    Runs a detection loop in a separate thread.
    """

    def __init__(self, camera, model):
        # type: (CameraAdapter, ModelAdapter) -> None
        """
        Initializes the ObjectDetector.
        
        Args:
            camera: An instance of a CameraAdapter.
            model: An instance of a ModelAdapter.
        """
        self.camera = camera
        self.model = model
        
        self.is_running = False
        self._thread = None
        self._lock = threading.Lock()
        
        self.latest_frame = None
        self.latest_detections = []
        self.current_fps = 0.0

    def start(self):
        # type: () -> None
        """Starts the continuous detection loop in a background thread."""
        if self.is_running:
            logger.warning("Detector is already running.")
            return

        if not self.camera.connect():
            logger.error("Cannot start detector: Camera connection failed.")
            return

        self.is_running = True
        self._thread = threading.Thread(target=self._detection_loop)
        self._thread.daemon = True
        self._thread.start()
        logger.info("Detection loop started.")

    def stop(self):
        # type: () -> None
        """Stops the detection loop and releases resources."""
        if not self.is_running:
            return

        self.is_running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            
        self.camera.release()
        logger.info("Detection loop stopped.")

    def process_single_frame(self):
        # type: () -> tuple
        """Processes a single frame synchronously (useful for notebooks/testing)."""
        frame = self.camera.capture_frame()
        if frame is None:
            return None, []
            
        detections = self.model.predict(frame)
        return frame, detections

    def _detection_loop(self):
        # type: () -> None
        """The internal loop running in a separate thread."""
        prev_time = time.time()
        fps_filter = 0.9 # Smoothing factor for FPS calculation
        
        # Target FPS to limit CPU usage (15 FPS is usually enough for conveyor belts)
        target_fps = 15.0
        target_interval = 1.0 / target_fps
        
        while self.is_running:
            loop_start = time.time()
            
            frame = self.camera.capture_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            try:
                # Run inference (This is the heavy part)
                detections = self.model.predict(frame)
                
                # Update shared state securely
                with self._lock:
                    self.latest_frame = frame.copy()
                    self.latest_detections = detections
                    
                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - prev_time)
                prev_time = current_time
                self.current_fps = (fps_filter * self.current_fps) + ((1 - fps_filter) * fps)
                
            except Exception as e:
                logger.error("Error in detection loop: {0}".format(e))
                time.sleep(0.1)
                
            # Sleep to cap the frame rate and save CPU cycles
            elapsed = time.time() - loop_start
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)

    def get_latest_results(self):
        # type: () -> tuple
        """
        Retrieves the latest frame, detections, and calculated FPS.
        Thread-safe.
        """
        with self._lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
            detections = list(self.latest_detections)
            fps = self.current_fps
            
        return frame, detections, fps
