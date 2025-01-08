from helper import *
import numpy as np
from typing import Dict, List, Tuple

class LegRules:
    def __init__(self):
        """Initialize leg rule extraction with previous state tracking."""
        self.previous_landmarks = {
            'left_knee': None,
            'right_knee': None,
            'left_foot': None,
            'right_foot': None,
            'left_hip': None,
            'right_hip': None,
            'left_ankle': None,
            'right_ankle': None
        }
        
        # Define position thresholds
        self.FOOT_GROUND_THRESHOLD = 0.9  # Y-coordinate threshold for "on ground"
        self.KNEE_FLEX_THRESHOLD = 45  # Degrees
        self.KNEE_EXTENSION_THRESHOLD = 160  # Degrees
        self.SHOULDER_WIDTH_THRESHOLD = 0.3  # For foot positioning
        self.VERTICAL_MOVEMENT_THRESHOLD = 0.05
        
        # Define angle thresholds
        self.SLIGHT_BEND = 160  # Degrees
        self.DEEP_BEND = 90    # Degrees
        self.NINETY_DEG_RANGE = (85, 95)  # Range for "bent at 90 degrees"

    def extract_leg_rules(self, landmarks: Dict[str, Tuple[float, float]]) -> Dict[str, Dict[str, List[str]]]:
        """
        Extract comprehensive leg position and motion rules.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            
        Returns:
            Dictionary containing leg position and motion rules
        """
        leg_positions = get_empty_leg_position()
        
        # Process each leg component
        for side in ['left', 'right']:
            # Get current landmarks
            knee = landmarks.get(f'{side}_knee')
            foot = landmarks.get(f'{side}_foot')
            hip = landmarks.get(f'{side}_hip')
            ankle = landmarks.get(f'{side}_ankle')
            
            if all(x is not None for x in [knee, foot, hip, ankle]):
                # Get positions
                leg_positions[f'{side}_knee']['position'] = self._get_knee_position(
                    hip, knee, ankle, side)
                leg_positions[f'{side}_foot']['position'] = self._get_foot_position(
                    foot, ankle, landmarks.get(f'other_{side}_foot'))
                
                # Get motions if we have previous landmarks
                prev_knee = self.previous_landmarks[f'{side}_knee']
                prev_foot = self.previous_landmarks[f'{side}_foot']
                
                if prev_knee is not None:
                    leg_positions[f'{side}_knee']['motion'] = self._detect_knee_motion(
                        prev_knee, knee, hip)
                if prev_foot is not None:
                    leg_positions[f'{side}_foot']['motion'] = self._detect_foot_motion(
                        prev_foot, foot, ankle)
                
                # Update previous landmarks
                self.previous_landmarks[f'{side}_knee'] = knee
                self.previous_landmarks[f'{side}_foot'] = foot
                self.previous_landmarks[f'{side}_hip'] = hip
                self.previous_landmarks[f'{side}_ankle'] = ankle
        
        return leg_positions

    def _get_knee_position(self, hip: Tuple[float, float], knee: Tuple[float, float], 
                         ankle: Tuple[float, float], side: str) -> List[str]:
        """Determine knee position state."""
        positions = []
        
        # Calculate knee angle
        angle = calculate_angle(hip, knee, ankle)
        
        # Determine knee bend state
        if angle < self.KNEE_FLEX_THRESHOLD:
            positions.append('flexed')
        elif angle < self.DEEP_BEND:
            positions.append('bent')
        elif self.NINETY_DEG_RANGE[0] <= angle <= self.NINETY_DEG_RANGE[1]:
            positions.append('bent at 90 degrees')
        elif angle < self.SLIGHT_BEND:
            positions.append('slightly bent')
        else:
            positions.append('extended')
            
        return positions

    def _get_foot_position(self, foot: Tuple[float, float], ankle: Tuple[float, float],
                         other_foot: Tuple[float, float] = None) -> List[str]:
        """Determine foot position state."""
        positions = []
        
        # Check if foot is on ground
        if foot[1] > self.FOOT_GROUND_THRESHOLD:
            positions.append('on ground')
            
        # Check if foot is flat
        foot_ankle_angle = np.abs(foot[1] - ankle[1])
        if foot_ankle_angle < 0.1:  # Small threshold for "flat" position
            positions.append('flat')
            
        # Check shoulder-width stance if other foot is available
        if other_foot is not None:
            foot_distance = np.abs(foot[0] - other_foot[0])
            if self.SHOULDER_WIDTH_THRESHOLD <= foot_distance <= self.SHOULDER_WIDTH_THRESHOLD * 1.5:
                positions.append('shoulder-width apart')
        
        return positions

    def _detect_knee_motion(self, prev_pos: Tuple[float, float], curr_pos: Tuple[float, float], 
                         hip: Tuple[float, float]) -> List[str]:
        """Detect knee motion patterns."""
        motions = []
        
        # Calculate movement vectors
        movement = np.array(curr_pos) - np.array(prev_pos)
        
        # Check for extension/flexion
        if abs(movement[1]) > self.VERTICAL_MOVEMENT_THRESHOLD:
            if movement[1] > 0:
                motions.append('flexion')
            else:
                motions.append('extension')
        else:
            motions.append('stationary')
            
        return motions

    def _detect_foot_motion(self, prev_pos: Tuple[float, float], curr_pos: Tuple[float, float], 
                         ankle: Tuple[float, float]) -> List[str]:
        """Detect foot motion patterns."""
        motions = []
        
        # Calculate movement vectors
        movement = np.array(curr_pos) - np.array(prev_pos)
        
        # Check for plantar flexion/dorsiflexion
        if abs(movement[1]) > self.VERTICAL_MOVEMENT_THRESHOLD:
            if movement[1] > 0:
                motions.append('plantar flexion')
            else:
                motions.append('dorsiflexion')
        else:
            motions.append('stationary')
            
        return motions