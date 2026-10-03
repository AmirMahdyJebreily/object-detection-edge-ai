import cv2
import numpy as np

from adapters.model_adapter import Detection
from core.tracker import TrackedObject

class Visualizer:
    """Utility class for drawing overlays on images."""

    def __init__(self):
        # Color palette for different object IDs
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), 
            (255, 255, 0), (0, 255, 255), (255, 0, 255),
            (192, 192, 192), (128, 128, 128), (128, 0, 0)
        ]

    def _get_color(self, object_id):
        # type: (int) -> tuple
        """Assigns a consistent color based on object ID."""
        return self.colors[object_id % len(self.colors)]

    def draw_detections(self, image, detections):
        # type: (np.ndarray, list) -> np.ndarray
        """Draws bounding boxes and labels for untracked detections."""
        output = image.copy()
        for det in detections:
            x, y, w, h = det.bbox
            color = (0, 255, 0) # Default green for raw detections
            
            # Draw box
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = "{0}: {1:.2f}".format(det.class_name, det.confidence)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output, (x, y - text_h - 5), (x + text_w, y), color, -1)
            cv2.putText(output, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        return output

    def draw_tracked_objects(self, image, tracks):
        # type: (np.ndarray, dict) -> np.ndarray
        """Draws tracked objects and their trajectories."""
        output = image.copy()
        
        for obj_id, track in tracks.items():
            color = self._get_color(obj_id)
            
            # Draw bounding box if available
            if hasattr(track, 'last_bbox'):
                x, y, w, h = track.last_bbox
                cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            
            # Draw the trajectory points
            
            history = list(track.history)
            if len(history) > 1:
                # Draw trajectory line
                for i in range(1, len(history)):
                    pt1 = history[i-1]
                    pt2 = history[i]
                    # Fade out older points
                    thickness = int(np.sqrt(64 / float(len(history) - i + 1)) * 2)
                    cv2.line(output, pt1, pt2, color, max(1, thickness))
                    
            if history:
                latest_pt = history[-1]
                # Draw centroid circle
                cv2.circle(output, latest_pt, 4, color, -1)
                
                # Draw ID label
                label = "ID: {0} {1}".format(obj_id, track.class_name)
                cv2.putText(output, label, (latest_pt[0] - 10, latest_pt[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        return output

    def draw_anomaly_alert(self, image, message):
        # type: (np.ndarray, str) -> np.ndarray
        """Draws an alert message on the screen."""
        output = image.copy()
        cv2.putText(output, "ALERT: {0}".format(message), (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        return output
        
    def draw_fps(self, image, fps):
        # type: (np.ndarray, float) -> np.ndarray
        """Draws FPS on top left corner."""
        output = image.copy()
        cv2.putText(output, "FPS: {0:.1f}".format(fps), (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return output
