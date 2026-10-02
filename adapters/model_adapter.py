import abc
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """Data class representing a single object detection."""
    class_id: int
    class_name: str
    confidence: float
    # Bounding box format: (x_min, y_min, width, height) in pixels
    bbox: Tuple[int, int, int, int] 


class ModelAdapter(abc.ABC):
    """Abstract base class for object detection models."""

    @abc.abstractmethod
    def load_model(self, model_path: str) -> None:
        """Loads the model from the specified path."""
        pass

    @abc.abstractmethod
    def preprocess(self, image: np.ndarray) -> Any:
        """Preprocesses the image for the model."""
        pass

    @abc.abstractmethod
    def infer(self, preprocessed_input: Any) -> Any:
        """Runs inference on the preprocessed input."""
        pass

    @abc.abstractmethod
    def postprocess(self, raw_output: Any, original_image_shape: Tuple[int, int]) -> List[Detection]:
        """Converts raw model output into a list of Detection objects."""
        pass
    
    def predict(self, image: np.ndarray) -> List[Detection]:
        """End-to-end prediction pipeline: preprocess -> infer -> postprocess."""
        preprocessed = self.preprocess(image)
        raw_output = self.infer(preprocessed)
        return self.postprocess(raw_output, image.shape[:2])

    @abc.abstractmethod
    def get_input_shape(self) -> Tuple[int, int]:
        """Returns the expected input shape (width, height) for the model."""
        pass

    @abc.abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Returns metadata about the loaded model."""
        pass


class MediaPipeModelAdapter(ModelAdapter):
    """Concrete implementation of ModelAdapter using MediaPipe Object Detection."""

    def __init__(self, score_threshold: float = 0.5, max_results: int = 5):
        """
        Initializes the MediaPipe model adapter.
        
        Args:
            score_threshold: Confidence threshold for detections.
            max_results: Maximum number of objects to detect.
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is not installed. Please install it using 'pip install mediapipe'.")
            
        self.score_threshold = score_threshold
        self.max_results = max_results
        self.detector: Optional[vision.ObjectDetector] = None
        self.model_path: str = ""

    def load_model(self, model_path: str) -> None:
        """Loads the MediaPipe TFLite model."""
        try:
            self.model_path = model_path
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.ObjectDetectorOptions(
                base_options=base_options,
                score_threshold=self.score_threshold,
                max_results=self.max_results
            )
            self.detector = vision.ObjectDetector.create_from_options(options)
            logger.info(f"Successfully loaded MediaPipe model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading MediaPipe model: {e}")
            raise

    def preprocess(self, image: np.ndarray) -> mp.Image:
        """Converts a numpy array (BGR or RGB) to a MediaPipe Image."""
        # Assume input is BGR (OpenCV default), MediaPipe expects RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Need to create contiguous array for MediaPipe
            rgb_image = np.ascontiguousarray(image[:, :, ::-1])
        else:
            rgb_image = np.ascontiguousarray(image)
            
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    def infer(self, preprocessed_input: mp.Image) -> vision.ObjectDetectorResult:
        """Runs inference using the MediaPipe detector."""
        if self.detector is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        
        return self.detector.detect(preprocessed_input)

    def postprocess(self, raw_output: vision.ObjectDetectorResult, original_image_shape: Tuple[int, int]) -> List[Detection]:
        """Converts MediaPipe detection results to standard Detection objects."""
        detections = []
        
        # original_image_shape is (height, width)
        
        for detection in raw_output.detections:
            # Get bounding box
            bbox = detection.bounding_box
            x_min = int(bbox.origin_x)
            y_min = int(bbox.origin_y)
            width = int(bbox.width)
            height = int(bbox.height)
            
            # Get category (take the highest scoring one)
            category = detection.categories[0]
            
            det = Detection(
                class_id=category.index if hasattr(category, 'index') else -1,
                class_name=category.category_name,
                confidence=category.score,
                bbox=(x_min, y_min, width, height)
            )
            detections.append(det)
            
        return detections

    def get_input_shape(self) -> Tuple[int, int]:
        """MediaPipe handles resizing internally, returning dynamic/unknown."""
        return (-1, -1)

    def get_model_info(self) -> Dict[str, Any]:
        """Returns info about the MediaPipe model."""
        return {
            "type": "MediaPipe Object Detector",
            "path": self.model_path,
            "score_threshold": self.score_threshold,
            "max_results": self.max_results
        }
