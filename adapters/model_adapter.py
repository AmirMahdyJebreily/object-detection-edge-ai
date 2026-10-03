import abc
import logging
import collections
import numpy as np
import cv2

from utils import compat

try:
    # Try importing tflite_runtime (for edge devices like NanoPi/Raspberry Pi)
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False


logger = logging.getLogger(__name__)

class Detection:
    """Class representing a single object detection."""
    def __init__(self, class_id, class_name, confidence, bbox):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        # Bounding box format: (x_min, y_min, width, height) in pixels
        self.bbox = bbox


class ModelAdapter(abc.ABC):
    """Abstract base class for object detection models."""

    @abc.abstractmethod
    def load_model(self, model_path):
        # type: (str) -> None
        pass

    @abc.abstractmethod
    def preprocess(self, image):
        # type: (np.ndarray) -> any
        pass

    @abc.abstractmethod
    def infer(self, preprocessed_input):
        # type: (any) -> any
        pass

    @abc.abstractmethod
    def postprocess(self, raw_output, original_image_shape):
        # type: (any, tuple) -> list
        pass
    
    def predict(self, image):
        # type: (np.ndarray) -> list
        preprocessed = self.preprocess(image)
        raw_output = self.infer(preprocessed)
        return self.postprocess(raw_output, image.shape[:2])

    @abc.abstractmethod
    def get_input_shape(self):
        # type: () -> tuple
        pass

    @abc.abstractmethod
    def get_model_info(self):
        # type: () -> dict
        pass


class TFLiteModelAdapter(ModelAdapter):
    """Concrete implementation using tflite_runtime (Zero MediaPipe dependency)."""

    def __init__(self, score_threshold=0.5, max_results=5):
        # type: (float, int) -> None
        if not TFLITE_AVAILABLE:
            raise ImportError("tflite_runtime is not installed.")
            
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

    def load_model(self, model_path):
        # type: (str) -> None
        self.model_path = model_path
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        shape = self.input_details[0]['shape']
        self.input_shape = (shape[2], shape[1]) # (width, height)
        self.is_quantized = self.input_details[0]['dtype'] in [np.uint8, np.int8]
        logger.info("Loaded TFLite model from {0}. Quantized: {1}".format(model_path, self.is_quantized))

    def preprocess(self, image):
        # type: (np.ndarray) -> np.ndarray
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

    def infer(self, preprocessed_input):
        # type: (np.ndarray) -> list
        if self.interpreter is None:
            raise RuntimeError("Model not loaded.")
            
        self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed_input)
        self.interpreter.invoke()
        
        outputs = []
        for out in self.output_details:
            outputs.append(self.interpreter.get_tensor(out['index']))
        return outputs

    def postprocess(self, raw_output, original_image_shape):
        # type: (list, tuple) -> list
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
            
            class_name = self.labels.get(class_id, "Object_{0}".format(class_id))
            
            det = Detection(class_id, class_name, score, (x_min, y_min, width, height))
            detections.append(det)
            
        return detections

    def get_input_shape(self):
        # type: () -> tuple
        return self.input_shape

    def get_model_info(self):
        # type: () -> dict
        return {
            "type": "TFLite Runtime Object Detector",
            "path": self.model_path,
            "quantized": self.is_quantized,
            "score_threshold": self.score_threshold
        }

class ClassicMotionAdapter(ModelAdapter):
    """Classic computer vision background subtraction adapter."""
    def __init__(self, score_threshold=0.5, max_results=5):
        # type: (float, int) -> None
        self.score_threshold = score_threshold
        self.max_results = max_results
        self.bg_subtractor = compat.create_background_subtractor()
        self.input_shape = (640, 480)
        self.min_area = 500
        
    def load_model(self, model_path):
        # type: (str) -> None
        logger.info("ClassicMotionAdapter does not require a model file.")
        
    def preprocess(self, image):
        # type: (np.ndarray) -> np.ndarray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        return blurred
        
    def infer(self, preprocessed_input):
        # type: (np.ndarray) -> list
        fg_mask = self.bg_subtractor.apply(preprocessed_input)
        _, thresh = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=2)
        contours, hierarchy = compat.find_contours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
        
    def postprocess(self, raw_output, original_image_shape):
        # type: (list, tuple) -> list
        detections = []
        for c in raw_output:
            if cv2.contourArea(c) < self.min_area:
                continue
                
            (x, y, w, h) = cv2.boundingRect(c)
            # Fake confidence
            det = Detection(0, "Motion", 1.0, (x, y, w, h))
            detections.append(det)
            if len(detections) >= self.max_results:
                break
        return detections
        
    def get_input_shape(self):
        # type: () -> tuple
        return self.input_shape
        
    def get_model_info(self):
        # type: () -> dict
        return {
            "type": "Classic Motion Detector (Background Subtraction)",
            "path": "none",
            "quantized": False,
            "score_threshold": self.score_threshold
        }

if TFLITE_AVAILABLE:
    MediaPipeModelAdapter = TFLiteModelAdapter
else:
    MediaPipeModelAdapter = ClassicMotionAdapter
