import argparse
import logging
import time
import sys
import os
import cv2

from adapters.camera_adapter import USBCameraAdapter
from adapters.model_adapter import MediaPipeModelAdapter
from core.detector import ObjectDetector
from core.tracker import SimpleCentroidTracker
from utils.visualizer import Visualizer
from utils.config import default_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Conveyor Belt Object Detection System")
    parser.add_argument("--camera", type=int, default=default_config.camera.index,
                        help="Camera index (default: 0)")
    parser.add_argument("--model", type=str, default=default_config.model.model_path,
                        help="Path to TFLite model")
    parser.add_argument("--threshold", type=float, default=default_config.model.score_threshold,
                        help="Detection confidence threshold")
    parser.add_argument("--headless", action="store_true",
                        help="Run without displaying video window")
    return parser.parse_args()

def main():
    args = parse_args()

    # Ensure model exists or provide helpful error
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        logger.info("Please download a MediaPipe compatible model.")
        logger.info("Example: wget -q -O models/efficientdet_lite0.tflite -q https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite")
        sys.exit(1)

    # 1. Initialize Adapters
    logger.info("Initializing components...")
    camera = USBCameraAdapter(camera_index=args.camera)
    model = MediaPipeModelAdapter(score_threshold=args.threshold)
    
    try:
        model.load_model(args.model)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # 2. Initialize Core Components
    detector = ObjectDetector(camera=camera, model=model)
    tracker = SimpleCentroidTracker(
        max_distance=default_config.tracker.max_distance,
        max_disappeared=default_config.tracker.max_disappeared,
        max_history=default_config.tracker.max_history
    )
    visualizer = Visualizer()

    # 3. Start System
    logger.info("Starting detection system. Press Ctrl+C to stop.")
    detector.start()
    
    try:
        # Wait a moment for the camera to warm up
        time.sleep(2)
        
        while detector.is_running:
            # Get latest data from detector thread
            frame, detections, fps = detector.get_latest_results()
            
            if frame is None:
                time.sleep(0.01)
                continue
                
            # Update tracker with new detections
            tracks = tracker.update(detections)
            
            # Visualization
            if not args.headless:
                # We could draw raw detections, but tracks are more informative
                # display_frame = visualizer.draw_detections(frame, detections)
                display_frame = frame.copy()
                
                # Check for anomalies and alert
                for obj_id in tracks:
                    if tracker.is_moving_anomaly(obj_id):
                        display_frame = visualizer.draw_anomaly_alert(display_frame, f"Anomaly ID {obj_id}")
                
                # Draw tracks
                display_frame = visualizer.draw_tracked_objects(display_frame, tracks)
                
                # Draw FPS
                display_frame = visualizer.draw_fps(display_frame, fps)
                
                cv2.imshow("Conveyor Belt Monitor", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q') or key == 27: # 'q' or ESC
                    logger.info("Quit signal received.")
                    break
            else:
                # In headless mode, just log occasionally
                if int(time.time()) % 5 == 0:
                    logger.info(f"Tracking {len(tracks)} objects. FPS: {fps:.1f}")
                time.sleep(0.03)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Cleanup
        detector.stop()
        if not args.headless:
            cv2.destroyAllWindows()
        logger.info("System shutdown complete.")

if __name__ == "__main__":
    main()
