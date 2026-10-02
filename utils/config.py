import os
from dataclasses import dataclass, field

@dataclass
class CameraConfig:
    index: int = 0
    backend: int = 0 # 0 for default (cv2.CAP_ANY)
    width: int = 640
    height: int = 480
    fps_target: int = 30

@dataclass
class ModelConfig:
    # Default model path assumes running from project root
    model_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "efficientdet_lite0.tflite")
    score_threshold: float = 0.5
    max_results: int = 10
    quantized: bool = True # Typically True for Edge TPU / microcontrollers

@dataclass
class TrackerConfig:
    max_distance: float = 80.0
    max_disappeared: int = 5
    max_history: int = 30

@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    
# Global default configuration instance
default_config = AppConfig()
