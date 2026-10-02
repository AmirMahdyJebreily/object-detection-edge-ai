import abc
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import cv2

try:
    # Try importing tflite_runtime (for edge devices like NanoPi/Raspberry Pi)
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        # Fallback to full tensorflow (for Desktop testing)
        import tensorflow as tf
        tflite = tf.lite
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False


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
        pass

    @abc.abstractmethod
    def preprocess(self, image: np.ndarray) -> Any:
        pass

    @abc.abstractmethod
    def infer(self, preprocessed_input: Any) -> Any:
        pass

    @abc.abstractmethod
    def postprocess(self, raw_output: Any, original_image_shape: Tuple[int, int]) -> List[Detection]:
        pass
    
    def predict(self, image: np.ndarray) -> List[Detection]:
        preprocessed = self.preprocess(image)
        raw_output = self.infer(preprocessed)
        return self.postprocess(raw_output, image.shape[:2])

    @abc.abstractmethod
    def get_input_shape(self) -> Tuple[int, int]:
        pass

    @abc.abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        pass


class TFLiteModelAdapter(ModelAdapter):
    """Concrete implementation using tflite_runtime (Zero MediaPipe dependency)."""

    def __init__(self, score_threshold: float = 0.5, max_results: int = 5):
        if not TFLITE_AVAILABLE:
            raise ImportError("tflite_runtime or tensorflow is not installed.")
            
        self.score_threshold = score_threshold
        self.max_results = max_results
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.model_path = ""
        self.input_shape = (320, 320) # Will be updated on load
        self.is_quantized = False
        
        # Sample COCO labels (IDs usually shift by 1 depending on model)
        self.labels = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 
                       4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 
                       39: 'bottle', 43: 'knife', 47: 'apple'}

    def load_model(self, model_path: str) -> None:
        self.model_path = model_path
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        shape = self.input_details[0]['shape']
        self.input_shape = (shape[2], shape[1]) # (width, height)
        self.is_quantized = self.input_details[0]['dtype'] in [np.uint8, np.int8]
        logger.info(f"Loaded TFLite model from {model_path}. Quantized: {self.is_quantized}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_image, self.input_shape)
        input_data = np.expand_dims(resized, axis=0)
        
        if not self.is_quantized:
            input_data = (np.float32(input_data) - 127.5) / 127.5
            
        # Ensure it matches model expectation (uint8 vs int8)
        if self.input_details[0]['dtype'] == np.int8:
            input_data = input_data.astype(np.int8)
        elif self.input_details[0]['dtype'] == np.uint8:
            input_data = input_data.astype(np.uint8)
            
        return input_data

    def infer(self, preprocessed_input: np.ndarray) -> List[np.ndarray]:
        if self.interpreter is None:
            raise RuntimeError("Model not loaded.")
            
        self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed_input)
        self.interpreter.invoke()
        
        outputs = []
        for out in self.output_details:
            outputs.append(self.interpreter.get_tensor(out['index']))
        return outputs

    def postprocess(self, raw_output: List[np.ndarray], original_image_shape: Tuple[int, int]) -> List[Detection]:
        # Try to identify boxes, classes, scores based on tensor shapes
        boxes = None
        classes = None
        scores = None
        
        for out in raw_output:
            if len(out.shape) == 3 and out.shape[2] == 4:
                boxes = out[0]
            elif len(out.shape) == 2 or (len(out.shape) == 3 and out.shape[2] == 1):
                out_sq = np.squeeze(out)
                if np.max(out_sq) <= 1.0 and np.any(out_sq > 0):
                    scores = out_sq
                else:
                    classes = out_sq

        detections = []
        if boxes is None or scores is None or classes is None:
            return detections
            
        img_h, img_w = original_image_shape
        
        for i in range(len(scores)):
            if i >= self.max_results:
                break
                
            score = float(scores[i])
            if score < self.score_threshold:
                continue
                
            class_id = int(classes[i])
            # Default tflite output: [ymin, xmin, ymax, xmax]
            ymin, xmin, ymax, xmax = boxes[i]
            
            x_min = int(xmin * img_w)
            y_min = int(ymin * img_h)
            x_max = int(xmax * img_w)
            y_max = int(ymax * img_h)
            
            width = x_max - x_min
            height = y_max - y_min
            
            class_name = self.labels.get(class_id, f"Object_{class_id}")
            
            det = Detection(class_id, class_name, score, (x_min, y_min, width, height))
            detections.append(det)
            
        return detections

    def get_input_shape(self) -> Tuple[int, int]:
        return self.input_shape

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "type": "TFLite Runtime Object Detector",
            "path": self.model_path,
            "quantized": self.is_quantized,
            "score_threshold": self.score_threshold
        }

# Alias so we don't have to rename imports in main.py, server.py, etc.
MediaPipeModelAdapter = TFLiteModelAdapter
