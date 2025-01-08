from helper import *
import numpy as np
from typing import Dict, List, Tuple

class TorsoRules:
    def __init__(self):
        """Initialize torso rule extraction with previous state tracking."""
        self.previous_landmarks = {
            'torso': None,
            'left_shoulder': None,
            'right_shoulder': None,
            'left_hip': None,
            'right_hip': None
        }
        
        # Define thresholds
        self.VERTICAL_THRESHOLD = 0.15
        self.HORIZONTAL_THRESHOLD = 0.1
        self.LEAN_THRESHOLD = 0.2
        self.SHOULDER_HIP_RATIO = 0.2
        
        # Define angle thresholds
        self.UPRIGHT_RANGE = (75, 105)
        self.LEAN_FORWARD_THRESHOLD = 75
        self.LEAN_BACK_THRESHOLD = 105
        self.RETRACTION_THRESHOLD = 0.1
        self.HIP_HINGE_THRESHOLD = 30
        
        # Motion detection thresholds
        self.MOTION_THRESHOLD = 0.05
        self.ROTATION_THRESHOLD = 0.08

    def extract_torso_rules(self, landmarks: Dict[str, Tuple[float, float]]) -> Dict[str, Dict[str, List[str]]]:
        """Extract comprehensive torso position and motion rules."""
        torso_positions = get_empty_torso_position()
        
        # Get key landmarks
        left_shoulder = landmarks.get('left_shoulder')
        right_shoulder = landmarks.get('right_shoulder')
        left_hip = landmarks.get('left_hip')
        right_hip = landmarks.get('right_hip')
        
        if all(x is not None for x in [left_shoulder, right_shoulder, left_hip, right_hip]):
            # Convert to numpy arrays for calculations
            left_shoulder = np.array(left_shoulder)
            right_shoulder = np.array(right_shoulder)
            left_hip = np.array(left_hip)
            right_hip = np.array(right_hip)
            
            # Calculate centers
            shoulder_center = np.mean([left_shoulder, right_shoulder], axis=0)
            hip_center = np.mean([left_hip, right_hip], axis=0)
            torso_center = np.mean([shoulder_center, hip_center], axis=0)
            
            # Get positions using all necessary landmarks
            torso_positions['torso']['position'] = self._get_torso_position(
                shoulder_center, hip_center, left_hip, right_hip)
            
            torso_positions['left_hip']['position'] = self._get_hip_position(
                left_hip, right_hip, left_shoulder, 'left')
            torso_positions['right_hip']['position'] = self._get_hip_position(
                right_hip, left_hip, right_shoulder, 'right')
            
            torso_positions['left_shoulder']['position'] = self._get_shoulder_position(
                left_shoulder, right_shoulder, left_hip, 'left')
            torso_positions['right_shoulder']['position'] = self._get_shoulder_position(
                right_shoulder, left_shoulder, right_hip, 'right')
            
            # Get motions if we have previous landmarks
            if all(self.previous_landmarks[k] is not None for k in self.previous_landmarks):
                # Torso motion
                torso_positions['torso']['motion'] = self._detect_torso_motion(
                    self.previous_landmarks['torso'], torso_center)
                
                # Hip motions
                torso_positions['left_hip']['motion'] = self._detect_hip_motion(
                    self.previous_landmarks['left_hip'],
                    left_hip,
                    self.previous_landmarks['left_shoulder'],
                    left_shoulder,
                    'left'
                )
                torso_positions['right_hip']['motion'] = self._detect_hip_motion(
                    self.previous_landmarks['right_hip'],
                    right_hip,
                    self.previous_landmarks['right_shoulder'],
                    right_shoulder,
                    'right'
                )
                
                # Shoulder motions
                torso_positions['left_shoulder']['motion'] = self._detect_shoulder_motion(
                    self.previous_landmarks['left_shoulder'],
                    left_shoulder,
                    self.previous_landmarks['right_shoulder'],
                    right_shoulder,
                    'left'
                )
                torso_positions['right_shoulder']['motion'] = self._detect_shoulder_motion(
                    self.previous_landmarks['right_shoulder'],
                    right_shoulder,
                    self.previous_landmarks['left_shoulder'],
                    left_shoulder,
                    'right'
                )
            
            # Update previous landmarks
            self.previous_landmarks['torso'] = torso_center
            self.previous_landmarks['left_shoulder'] = left_shoulder
            self.previous_landmarks['right_shoulder'] = right_shoulder
            self.previous_landmarks['left_hip'] = left_hip
            self.previous_landmarks['right_hip'] = right_hip
        
        return torso_positions

    def _get_torso_position(self, shoulder_center: np.ndarray, hip_center: np.ndarray,
                          left_hip: np.ndarray, right_hip: np.ndarray) -> List[str]:
        """Determine torso position state."""
        positions = []
        
        # Calculate spine angle (vertical = 90 degrees)
        spine_vector = shoulder_center - hip_center
        spine_angle = np.degrees(np.arctan2(spine_vector[1], spine_vector[0])) + 90
        
        # Check basic positions
        if self.UPRIGHT_RANGE[0] <= spine_angle <= self.UPRIGHT_RANGE[1]:
            positions.append('upright')
            
        # Check neutral spine
        shoulder_hip_tilt = abs(spine_vector[0] / (spine_vector[1] + 1e-6))
        hip_tilt = abs(left_hip[1] - right_hip[1]) / (abs(left_hip[0] - right_hip[0]) + 1e-6)
        
        if shoulder_hip_tilt < self.HORIZONTAL_THRESHOLD and hip_tilt < self.HORIZONTAL_THRESHOLD:
            positions.append('neutral spine')
        
        # Check leaning positions
        if spine_angle < self.LEAN_FORWARD_THRESHOLD:
            positions.append('leaning forward')
        elif spine_angle > self.LEAN_BACK_THRESHOLD:
            positions.append('leaning backward')
            
        # Check if on bench
        if hip_center[1] > shoulder_center[1] + self.VERTICAL_THRESHOLD:
            positions.append('on bench')
            
        return positions

    def _get_hip_position(self, hip: np.ndarray, other_hip: np.ndarray,
                         shoulder: np.ndarray, side: str) -> List[str]:
        """Determine hip position state."""
        positions = []
        
        # Calculate relative positions
        hip_shoulder_vector = shoulder - hip
        hip_angle = np.degrees(np.arctan2(hip_shoulder_vector[1], hip_shoulder_vector[0])) + 90
        
        # Check for seated position
        if hip[1] > shoulder[1] + self.VERTICAL_THRESHOLD:
            positions.append('seated')
        
        # Check for hip hinge
        hip_tilt = abs(hip[1] - other_hip[1]) / (abs(hip[0] - other_hip[0]) + 1e-6)
        if hip_tilt > self.SHOULDER_HIP_RATIO:
            positions.append('hip hinge')
            
        # Check if upright
        if self.UPRIGHT_RANGE[0] <= hip_angle <= self.UPRIGHT_RANGE[1]:
            positions.append('upright')
        
        return positions

    def _get_shoulder_position(self, shoulder: np.ndarray, other_shoulder: np.ndarray,
                             hip: np.ndarray, side: str) -> List[str]:
        """Determine shoulder position state."""
        positions = []
        
        # Check if on bench
        if shoulder[1] < hip[1] - self.VERTICAL_THRESHOLD:
            positions.append('on bench')
            
        # Calculate shoulder retraction/protraction
        shoulder_diff = shoulder - other_shoulder
        shoulder_distance = np.linalg.norm(shoulder_diff)
        
        # Determine retraction/protraction state
        if shoulder_distance > self.RETRACTION_THRESHOLD:
            if ((side == 'left' and shoulder[0] < other_shoulder[0]) or
                (side == 'right' and shoulder[0] > other_shoulder[0])):
                positions.append('retracted')
            else:
                positions.append('protracted')
        else:
            positions.append('neutral')
            
        return positions

    def _detect_hip_motion(self, prev_hip: np.ndarray, curr_hip: np.ndarray,
                         prev_shoulder: np.ndarray, curr_shoulder: np.ndarray,
                         side: str) -> List[str]:
        """Detect hip motion patterns."""
        motions = []
        
        # Calculate movement vectors
        hip_movement = curr_hip - prev_hip
        shoulder_movement = curr_shoulder - prev_shoulder
        
        # Vertical movements (extension/flexion)
        if abs(hip_movement[1]) > self.MOTION_THRESHOLD:
            if hip_movement[1] < 0:  # Moving upward
                motions.append('extension')
            else:  # Moving downward
                motions.append('flexion')
                
        # Horizontal movements
        if abs(hip_movement[0]) > self.MOTION_THRESHOLD:
            # Calculate relative to shoulder movement to determine if it's real hip movement
            relative_movement = hip_movement[0] - shoulder_movement[0]
            if abs(relative_movement) > self.MOTION_THRESHOLD:
                if relative_movement > 0:
                    motions.append('abduction')
                else:
                    motions.append('adduction')
        
        # Rotation detection
        hip_rotation = np.arctan2(hip_movement[1], hip_movement[0])
        if abs(hip_rotation) > self.ROTATION_THRESHOLD:
            if side == 'left':
                if hip_rotation > 0:
                    motions.append('medial rotation')
                else:
                    motions.append('lateral rotation')
            else:  # right side
                if hip_rotation > 0:
                    motions.append('lateral rotation')
                else:
                    motions.append('medial rotation')
        
        # If no significant movement detected
        if not motions:
            motions.append('stationary')
            
        return motions

    def _detect_shoulder_motion(self, prev_shoulder: np.ndarray, curr_shoulder: np.ndarray,
                              prev_other_shoulder: np.ndarray, curr_other_shoulder: np.ndarray,
                              side: str) -> List[str]:
        """Detect shoulder motion patterns."""
        motions = []
        
        # Calculate movement vectors
        shoulder_movement = curr_shoulder - prev_shoulder
        other_shoulder_movement = curr_other_shoulder - prev_other_shoulder
        
        # Vertical movements
        if abs(shoulder_movement[1]) > self.MOTION_THRESHOLD:
            if shoulder_movement[1] < 0:  # Moving upward
                motions.append('elevation')
            else:  # Moving downward
                motions.append('depression')
        
        # Horizontal movements
        if abs(shoulder_movement[0]) > self.MOTION_THRESHOLD:
            # Calculate relative movement to detect true protraction/retraction
            relative_movement = shoulder_movement[0] - other_shoulder_movement[0]
            if abs(relative_movement) > self.MOTION_THRESHOLD:
                if ((side == 'left' and relative_movement < 0) or 
                    (side == 'right' and relative_movement > 0)):
                    motions.append('scapular retraction')
                else:
                    motions.append('scapular protraction')
        
        # Rotation detection
        shoulder_rotation = np.arctan2(shoulder_movement[1], shoulder_movement[0])
        if abs(shoulder_rotation) > self.ROTATION_THRESHOLD:
            if side == 'left':
                if shoulder_rotation > 0:
                    motions.append('medial rotation')
                else:
                    motions.append('lateral rotation')
            else:  # right side
                if shoulder_rotation > 0:
                    motions.append('lateral rotation')
                else:
                    motions.append('medial rotation')
        
        # If no significant movement detected
        if not motions:
            motions.append('stationary')
            
        return motions

    def _detect_torso_motion(self, prev_center: np.ndarray, 
                           curr_center: np.ndarray) -> List[str]:
        """Detect torso motion patterns."""
        motions = []
        
        # Calculate movement vector
        movement = curr_center - prev_center
        
        # Vertical movements
        if abs(movement[1]) > self.MOTION_THRESHOLD:
            if movement[1] < 0:  # Moving upward
                motions.append('extension')
            else:  # Moving downward
                motions.append('flexion')
        
        # Lateral movements
        if abs(movement[0]) > self.MOTION_THRESHOLD:
            if movement[0] > 0:
                motions.append('lateral flexion right')
            else:
                motions.append('lateral flexion left')
        
        # If no significant movement
        if not motions:
            motions.append('stationary')
            
        return motions