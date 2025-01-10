import torch
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from pathlib import Path
import sys
import os
YOLOV7_ROOT = os.path.abspath('./yolov7')  # Adjust this path to your YOLOv7 directory
sys.path.append(YOLOV7_ROOT)
from typing import Dict, List, Tuple, Optional
from yolov7.models.experimental import attempt_load
from yolov7.utils.torch_utils import select_device
from yolov7.utils.general import check_img_size, non_max_suppression

class YOLOv7EquipmentDetector:
    """Detector for gym equipment using YOLOv7."""
    
    def __init__(self, model_path: str, equipment_list: List[str], 
                 conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        """
        Initialize the YOLOv7 model for equipment detection.
        
        Args:
            model_path: Path to YOLOv7 weights file
            equipment_list: List of equipment classes to detect
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
        """
        self.equipment_list = equipment_list
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load model
        self.model, self.device, self.stride, self.img_size = self._load_model(model_path)
        print(f"Equipment detector initialized on {self.device}")
        
    def _load_model(self, weights_path: str, img_size: int = 640) -> Tuple:
        """Load and configure YOLOv7 model."""
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
            
        # Select device
        device = select_device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        model = attempt_load(weights_path, map_location=device)
        stride = int(model.stride.max())
        img_size = check_img_size(img_size, s=stride)
        
        # Convert to half precision if on GPU
        if device.type != 'cpu':
            model.half()
            
        return model, device, stride, img_size
        
    def detect_equipment(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect equipment in a frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            List of dictionaries containing detection info:
            [{'label': str, 'confidence': float, 'bbox': tuple, 'center': tuple}]
        """
        # Preprocess frame
        resized_frame = cv2.resize(frame, (self.img_size, self.img_size))
        frame_tensor = torch.from_numpy(resized_frame).to(self.device).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1)
        
        if self.device.type != 'cpu':
            frame_tensor = frame_tensor.half()
            
        # Add batch dimension
        if frame_tensor.ndimension() == 3:
            frame_tensor = frame_tensor.unsqueeze(0)
            
        # Run inference
        with torch.no_grad():
            predictions = self.model(frame_tensor)[0]
            
        # Apply NMS
        predictions = non_max_suppression(
            predictions, 
            conf_thres=self.conf_threshold,
            iou_thres=self.iou_threshold
        )
        
        detections = []
        if len(predictions):
            # Scale coordinates to original image size
            scale_factor = frame.shape[1] / self.img_size  # Assuming square input
            
            for *xyxy, conf, cls in predictions[0]:
                label = self.model.names[int(cls)]
                
                if label in self.equipment_list:
                    # Convert bbox coordinates
                    x1, y1, x2, y2 = [coord.item() * scale_factor for coord in xyxy]
                    
                    # Calculate center point
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    detections.append({
                        'label': label,
                        'confidence': conf.item(),
                        'bbox': (x1, y1, x2, y2),
                        'center': (center_x, center_y)
                    })
        
        return detections
        
    def visualize_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detected equipment on frame."""
        frame_copy = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            center_x, center_y = map(int, det['center'])
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{det['label']} {det['confidence']:.2f}"
            cv2.putText(frame_copy, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(frame_copy, (center_x, center_y), 4, (255, 0, 0), -1)
            
        return frame_copy

class PoseDetector:

    """Enhanced pose detector with smoothing and spike detection."""
    
    def __init__(self, smoothing_window_size: int = 5, 
                 spike_threshold: float = 0.2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize pose detector with smoothing capabilities.
        
        Args:
            smoothing_window_size: Window size for moving average smoothing
            spike_threshold: Threshold for spike detection (in standard deviations)
            min_detection_confidence: MediaPipe detection confidence threshold
            min_tracking_confidence: MediaPipe tracking confidence threshold
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Smoothing parameters
        self.window_size = smoothing_window_size
        self.spike_threshold = spike_threshold
        
        # Initialize landmark history
        self._init_landmark_history()
        
    def _init_landmark_history(self):
        """Initialize deques for landmark position history."""
        self.landmark_history = {}
        for landmark in self.mp_pose.PoseLandmark:
            self.landmark_history[landmark.name] = deque(maxlen=self.window_size)
            
    def get_joint_positions_from_frame(self, frame: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """
        Get joint positions from a single frame.
        This is the main method that exercise_analyzer.py expects.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            Dictionary mapping landmark names to (x, y) coordinates
        """
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        landmarks = []
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                name = self.mp_pose.PoseLandmark(idx).name.lower()
                current_pos = np.array([landmark.x, landmark.y])
                
                # Get smoothed position
                # smoothed_pos = self._smooth_position(name.upper(), current_pos)
                landmarks[name] = tuple(smoothed_pos)        
        return landmarks
        
    def apply_butterworth_filter(self, landmarks: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
        pass
    def visualize_pose(self, frame: np.ndarray, 
                      landmarks: Dict[str, Tuple[float, float]]) -> np.ndarray:
        """Draw pose landmarks on frame."""
        frame_copy = frame.copy()
        
        if not landmarks:
            return frame_copy
            
        # Draw landmarks and connections
        h, w, _ = frame.shape
        for name, (x, y) in landmarks.items():
            x_px, y_px = int(x * w), int(y * h)
            cv2.circle(frame_copy, (x_px, y_px), 5, (0, 255, 0), -1)
            cv2.putText(frame_copy, name, (x_px, y_px - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        return frame_copy