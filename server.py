import argparse
import logging
import time
import sys
import os
import cv2
from flask import Flask, Response, render_template_string

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

app = Flask(__name__)

# Global variables to hold our components
detector = None
tracker = None
visualizer = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Conveyor Belt Monitor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f0f2f5;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        h1 {
            color: #333;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        img {
            max-width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 4px;
        }
        .footer {
            margin-top: 20px;
            color: #777;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Conveyor Belt Live Feed</h1>
        <img src="{{ url_for('video_feed') }}" alt="Live stream from Conveyor Belt">
        <div class="footer">Streaming via MJPEG. Press Ctrl+C in terminal to stop.</div>
    </div>
</body>
</html>
"""

def generate_frames():
    """Generator function to continuously yield JPEG frames for MJPEG streaming."""
    while True:
        if detector is None or not detector.is_running:
            time.sleep(0.1)
            continue
            
        frame, detections, fps = detector.get_latest_results()
        
        if frame is None:
            time.sleep(0.01)
            continue
            
        # Update tracker with new detections
        tracks = tracker.update(detections)
        
        # Visualization
        display_frame = frame.copy()
        
        # Check for anomalies and alert
        for obj_id in tracks:
            if tracker.is_moving_anomaly(obj_id):
                display_frame = visualizer.draw_anomaly_alert(display_frame, f"Anomaly ID {obj_id}")
        
        # Draw tracks
        display_frame = visualizer.draw_tracked_objects(display_frame, tracks)
        
        # Draw FPS
        display_frame = visualizer.draw_fps(display_frame, fps)
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', display_frame)
        if not ret:
            logger.error("Failed to encode frame to JPEG")
            continue
            
        frame_bytes = buffer.tobytes()
        
        # Yield in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
        # Slight sleep to avoid 100% CPU on generator if processing is very fast
        time.sleep(0.01)

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template_string(HTML_TEMPLATE)

def init_system(camera_index: int, model_path: str, threshold: float):
    """Initialize the object detection components."""
    global detector, tracker, visualizer
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)

    logger.info("Initializing components for web server...")
    camera = USBCameraAdapter(camera_index=camera_index)
    model = MediaPipeModelAdapter(score_threshold=threshold)
    
    try:
        model.load_model(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    detector = ObjectDetector(camera=camera, model=model)
    tracker = SimpleCentroidTracker(
        max_distance=default_config.tracker.max_distance,
        max_disappeared=default_config.tracker.max_disappeared,
        max_history=default_config.tracker.max_history
    )
    visualizer = Visualizer()
    
    # Start detection in background thread
    detector.start()
    time.sleep(1) # wait for warmup

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Conveyor Belt Web Server")
    parser.add_argument("--camera", type=int, default=default_config.camera.index, help="Camera index")
    parser.add_argument("--model", type=str, default=default_config.model.model_path, help="Path to TFLite model")
    parser.add_argument("--threshold", type=float, default=default_config.model.score_threshold, help="Confidence threshold")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Web server host")
    
    args = parser.parse_args()
    
    try:
        # Initialize camera and models
        init_system(args.camera, args.model, args.threshold)
        
        # Start Flask web server (blocks until stopped)
        logger.info(f"Starting web server at http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        if detector:
            detector.stop()
        logger.info("Server shutdown complete.")
