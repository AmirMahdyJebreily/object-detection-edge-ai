import os

class CameraConfig:
    def __init__(self):
        self.index = 0
        self.backend = 0 # 0 for default (cv2.CAP_ANY)
        self.width = 640
        self.height = 480
        self.fps_target = 30

class ModelConfig:
    def __init__(self):
        # Default model path assumes running from project root
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "coco_ssd.tflite")
        self.score_threshold = 0.5
        self.max_results = 10
        self.quantized = True # Typically True for Edge TPU / microcontrollers

class TrackerConfig:
    def __init__(self):
        self.max_distance = 80.0
        self.max_disappeared = 5
        self.max_history = 30

class AppConfig:
    def __init__(self):
        self.camera = CameraConfig()
        self.model = ModelConfig()
        self.tracker = TrackerConfig()
    
# Global default configuration instance
default_config = AppConfig()
