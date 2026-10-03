import argparse
import logging
import time
import sys
import os
import json
import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

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

# Global variables to hold our components
detector = None
tracker = None
visualizer = None
enable_monitor = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

HTML_PAGE = b"""
<!DOCTYPE html>
<html>
<head>
    <title>Conveyor Belt Monitor</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center; }
        h1 { color: #ffffff; font-weight: 300; letter-spacing: 1px;}
        .container { background-color: #1e1e1e; padding: 25px; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.5); display: flex; flex-direction: column; align-items: center; max-width: 800px; width: 100%; }
        img { max-width: 100%; height: auto; border: 2px solid #333; border-radius: 8px; margin-bottom: 20px;}
        .stats { display: flex; justify-content: space-around; width: 100%; background: #2c2c2c; padding: 15px; border-radius: 8px; margin-bottom: 15px;}
        .stat-box { text-align: center; }
        .stat-value { font-size: 24px; font-weight: bold; color: #4caf50; }
        .stat-label { font-size: 12px; color: #aaa; text-transform: uppercase; letter-spacing: 1px;}
        .footer { margin-top: 20px; color: #666; font-size: 13px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Conveyor Belt Live Feed</h1>
        
        <div id="monitor-panel" class="stats" style="display: none;">
            <div class="stat-box">
                <div class="stat-value" id="cpu-val">--%</div>
                <div class="stat-label">CPU Usage</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="ram-val">--%</div>
                <div class="stat-label">RAM Usage</div>
            </div>
        </div>

        <img src="/video_feed" alt="Live stream from Conveyor Belt">
        <div class="footer">Streaming via MJPEG (Zero Dependencies). Press Ctrl+C in terminal to stop.</div>
    </div>
    <script>
        const enableMonitor = ENABLE_MONITOR_PLACEHOLDER;
        if (enableMonitor) {
            document.getElementById('monitor-panel').style.display = 'flex';
            setInterval(() => {
                fetch('/stats')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('cpu-val').innerText = data.cpu.toFixed(1) + '%';
                        document.getElementById('ram-val').innerText = data.ram.toFixed(1) + '%';
                        
                        document.getElementById('cpu-val').style.color = data.cpu > 80 ? '#f44336' : (data.cpu > 50 ? '#ff9800' : '#4caf50');
                        document.getElementById('ram-val').style.color = data.ram > 85 ? '#f44336' : '#4caf50';
                    })
                    .catch(err => console.error(err));
            }, 2000); // Lightweight background polling
        }
    </script>
</body>
</html>
"""

class VideoStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            
            # Inject the monitoring flag state into HTML
            html = HTML_PAGE.decode('utf-8').replace('ENABLE_MONITOR_PLACEHOLDER', 'true' if enable_monitor else 'false')
            self.wfile.write(html.encode('utf-8'))
            
        elif self.path == '/stats' and enable_monitor:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            if PSUTIL_AVAILABLE:
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                data = {"cpu": cpu, "ram": ram}
            else:
                data = {"cpu": 0, "ram": 0}
                
            self.wfile.write(json.dumps(data).encode('utf-8'))
            
        elif self.path == '/video_feed':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            
            try:
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
                            display_frame = visualizer.draw_anomaly_alert(display_frame, "Anomaly ID {0}".format(obj_id))
                    
                    # Draw tracks & FPS
                    display_frame = visualizer.draw_tracked_objects(display_frame, tracks)
                    display_frame = visualizer.draw_fps(display_frame, fps)
                    
                    # Encode frame to JPEG
                    ret, jpeg = cv2.imencode('.jpg', display_frame)
                    if not ret:
                        continue
                        
                    frame_bytes = jpeg.tobytes()
                    
                    # Write multipart frame
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(frame_bytes)))
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b'\r\n')
                    
                    time.sleep(0.01)
            except Exception as e:
                logger.info("Stream disconnected: {0}".format(e))
        else:
            self.send_error(404)

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    pass

def init_system(camera_index, model_path, threshold):
    """Initialize the object detection components."""
    global detector, tracker, visualizer
    
    if not os.path.exists(model_path):
        logger.error("Model file not found: {0}".format(model_path))
        sys.exit(1)

    logger.info("Initializing components for native web server...")
    camera = USBCameraAdapter(camera_index=camera_index)
    model = MediaPipeModelAdapter(score_threshold=threshold)
    
    try:
        model.load_model(model_path)
    except Exception as e:
        logger.error("Failed to load model: {0}".format(e))
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
    parser = argparse.ArgumentParser(description="Conveyor Belt Native Web Server")
    parser.add_argument("--camera", type=int, default=default_config.camera.index, help="Camera index")
    parser.add_argument("--model", type=str, default=default_config.model.model_path, help="Path to TFLite model file")
    parser.add_argument("--threshold", type=float, default=default_config.model.score_threshold, help="Confidence threshold")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Web server host")
    parser.add_argument("--monitor", action="store_true", help="Enable CPU and RAM monitoring on the HTML dashboard")
    
    args = parser.parse_args()
    
    # Set global flags
    enable_monitor = args.monitor
    if enable_monitor:
        if not PSUTIL_AVAILABLE:
            logger.warning("Monitoring enabled but 'psutil' is not installed. CPU/RAM will show 0%. Run 'pip install psutil'.")
        else:
            # Initialize psutil counters
            psutil.cpu_percent(interval=None)

    try:
        # Initialize camera and models
        init_system(args.camera, args.model, args.threshold)
        
        # Start Native web server
        server = ThreadedHTTPServer((args.host, args.port), VideoStreamHandler)
        logger.info("Starting native web server at http://{0}:{1}".format(args.host, args.port))
        server.serve_forever()
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        if detector:
            detector.stop()
        logger.info("Server shutdown complete.")
