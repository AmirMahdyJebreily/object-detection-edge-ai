import math
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque

from adapters.model_adapter import Detection

logger = logging.getLogger(__name__)

class TrackedObject:
    """Represents an object being tracked across frames."""
    
    def __init__(self, object_id: int, initial_detection: Detection, max_history: int = 30):
        self.object_id = object_id
        self.class_name = initial_detection.class_name
        self.history = deque(maxlen=max_history)
        self.missed_frames = 0
        
        self.update(initial_detection)

    def update(self, detection: Detection) -> None:
        """Updates the track with a new detection."""
        self.missed_frames = 0
        centroid = self._get_centroid(detection.bbox)
        self.history.append(centroid)

    def get_latest_centroid(self) -> Tuple[int, int]:
        """Returns the most recent centroid."""
        return self.history[-1] if self.history else (0, 0)

    def _get_centroid(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Calculates the center (x, y) of a bounding box."""
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)


class SimpleCentroidTracker:
    """
    A basic object tracker using Euclidean distance between centroids.
    Suitable for objects moving in a predictable manner like on a conveyor belt.
    """

    def __init__(self, max_distance: float = 50.0, max_disappeared: int = 5, max_history: int = 30):
        """
        Initializes the tracker.
        
        Args:
            max_distance: Maximum distance to associate a detection with an existing track.
            max_disappeared: Maximum consecutive frames an object can be missed before deregistration.
            max_history: Number of historical points to keep for trajectory.
        """
        self.next_object_id = 0
        self.tracks: Dict[int, TrackedObject] = {}
        
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.max_history = max_history

    def update(self, detections: List[Detection]) -> Dict[int, TrackedObject]:
        """
        Updates the tracker with new detections.
        
        Args:
            detections: List of Detection objects from the current frame.
            
        Returns:
            Dictionary mapping object IDs to their TrackedObject instances.
        """
        if not detections:
            # Increment missed frames for all existing tracks
            tracks_to_delete = []
            for obj_id, track in self.tracks.items():
                track.missed_frames += 1
                if track.missed_frames > self.max_disappeared:
                    tracks_to_delete.append(obj_id)
            
            for obj_id in tracks_to_delete:
                del self.tracks[obj_id]
                
            return self.tracks

        # Get centroids for current detections
        input_centroids = [
            (det.bbox[0] + det.bbox[2] // 2, det.bbox[1] + det.bbox[3] // 2)
            for det in detections
        ]

        if not self.tracks:
            # Register all new detections
            for det in detections:
                self._register(det)
        else:
            # Match existing tracks to new detections
            track_ids = list(self.tracks.keys())
            track_centroids = [t.get_latest_centroid() for t in self.tracks.values()]
            
            # Compute distance matrix
            dist_matrix = np.zeros((len(track_ids), len(input_centroids)))
            for i, tc in enumerate(track_centroids):
                for j, ic in enumerate(input_centroids):
                    dist_matrix[i, j] = math.hypot(tc[0] - ic[0], tc[1] - ic[1])
                    
            # Greedy matching
            used_rows = set()
            used_cols = set()
            
            # Sort distances
            sorted_indices = np.unravel_index(np.argsort(dist_matrix, axis=None), dist_matrix.shape)
            
            for row, col in zip(sorted_indices[0], sorted_indices[1]):
                if row in used_rows or col in used_cols:
                    continue
                    
                distance = dist_matrix[row, col]
                if distance > self.max_distance:
                    continue # Discard assignments too far apart
                    
                object_id = track_ids[row]
                self.tracks[object_id].update(detections[col])
                
                used_rows.add(row)
                used_cols.add(col)
                
            # Handle unmatched tracks (disappeared)
            unused_rows = set(range(dist_matrix.shape[0])) - used_rows
            for row in unused_rows:
                object_id = track_ids[row]
                self.tracks[object_id].missed_frames += 1
                if self.tracks[object_id].missed_frames > self.max_disappeared:
                    del self.tracks[object_id]
                    
            # Handle unmatched detections (new objects)
            unused_cols = set(range(dist_matrix.shape[1])) - used_cols
            for col in unused_cols:
                self._register(detections[col])
                
        return self.tracks

    def _register(self, detection: Detection) -> None:
        """Registers a new object."""
        self.tracks[self.next_object_id] = TrackedObject(
            self.next_object_id, detection, self.max_history
        )
        self.next_object_id += 1

    def get_trajectory(self, object_id: int) -> List[Tuple[int, int]]:
        """Gets the trajectory points for a specific object."""
        if object_id in self.tracks:
            return list(self.tracks[object_id].history)
        return []

    def is_moving_anomaly(self, object_id: int) -> bool:
        """
        Detects if an object has anomalous movement (e.g., sudden jump).
        Placeholder logic for a conveyor belt use case.
        """
        if object_id not in self.tracks:
            return False
            
        track = self.tracks[object_id]
        if len(track.history) < 3:
            return False
            
        # Example: check if movement vector changes direction sharply
        hist = list(track.history)
        dx1 = hist[-2][0] - hist[-3][0]
        dy1 = hist[-2][1] - hist[-3][1]
        
        dx2 = hist[-1][0] - hist[-2][0]
        dy2 = hist[-1][1] - hist[-2][1]
        
        # Calculate angle between vectors using dot product
        dot_product = dx1 * dx2 + dy1 * dy2
        mag1 = math.hypot(dx1, dy1)
        mag2 = math.hypot(dx2, dy2)
        
        if mag1 > 5 and mag2 > 5: # Only check if it's actually moving
            cos_angle = dot_product / (mag1 * mag2)
            # Clamp for safety
            cos_angle = max(min(cos_angle, 1.0), -1.0)
            angle = math.degrees(math.acos(cos_angle))
            
            if angle > 90: # Sudden reversal or sharp turn
                return True
                
        return False
