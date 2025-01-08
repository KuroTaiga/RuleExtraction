from helper import *
import numpy as np
from typing import Dict, List, Tuple

class ArmRules:
    def __init__(self):
        """Initialize arm rule extraction with previous state tracking."""
        self.previous_landmarks = {
            'left_elbow': None,
            'right_elbow': None,
            'left_hand': None,
            'right_hand': None,
            'left_shoulder': None,
            'right_shoulder': None
        }
        
        # Define position thresholds
        self.VERTICAL_THRESHOLD = 0.2
        self.HORIZONTAL_THRESHOLD = 0.2
        self.TORSO_PROXIMITY = 0.1
        self.EQUIPMENT_HOLD_THRESHOLD = 0.05
        
        # Define angle thresholds
        self.EXTENSION_ANGLE = 150  # Nearly straight arm
        self.FLEXION_ANGLE = 30    # Highly bent arm
        self.SLIGHT_FLEX_ANGLE = 80  # Slightly bent arm
        self.NINETY_DEG_RANGE = (80, 100)  # Range for "bent at 90 degrees"

    def extract_arm_rules(self, landmarks: Dict[str, Tuple[float, float]], equipment_centers: List[Tuple[float, float]] = None) -> Dict[str, Dict[str, List[str]]]:
        """
        Extract comprehensive arm position and motion rules.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            equipment_centers: Optional list of equipment center points
            
        Returns:
            Dictionary containing arm position and motion rules
        """
        arm_positions = get_empty_arm_position()
        
        # Process each arm component
        for side in ['left', 'right']:
            # Get current landmarks
            elbow = landmarks.get(f'{side}_elbow')
            hand = landmarks.get(f'{side}_hand')
            shoulder = landmarks.get(f'{side}_shoulder')
            
            if all(x is not None for x in [elbow, hand, shoulder]):
                # Get positions
                arm_positions[f'{side}_elbow']['position'] = self._get_elbow_position(
                    shoulder, elbow, hand, side)
                arm_positions[f'{side}_hand']['position'] = self._get_hand_position(
                    shoulder, elbow, hand, equipment_centers)
                
                # Get motions if we have previous landmarks
                prev_elbow = self.previous_landmarks[f'{side}_elbow']
                prev_hand = self.previous_landmarks[f'{side}_hand']
                
                if prev_elbow is not None:
                    arm_positions[f'{side}_elbow']['motion'] = self._detect_motion(
                        prev_elbow, elbow, shoulder)
                if prev_hand is not None:
                    arm_positions[f'{side}_hand']['motion'] = self._detect_motion(
                        prev_hand, hand, shoulder)
                
                # Update previous landmarks
                self.previous_landmarks[f'{side}_elbow'] = elbow
                self.previous_landmarks[f'{side}_hand'] = hand
                self.previous_landmarks[f'{side}_shoulder'] = shoulder
        
        return arm_positions

    def _get_elbow_position(self, shoulder: Tuple[float, float], elbow: Tuple[float, float], 
                          hand: Tuple[float, float], side: str) -> List[str]:
        """Determine elbow position state."""
        positions = []
        
        # Calculate angle
        angle = calculate_angle(shoulder, elbow, hand)
        
        # Check proximity to torso
        shoulder_elbow_diff = np.array(elbow) - np.array(shoulder)
        if abs(shoulder_elbow_diff[0]) < self.TORSO_PROXIMITY:
            positions.append('close to torso')
            
        # Determine flexion state
        if angle < self.FLEXION_ANGLE:
            positions.append('flexed')
        elif angle < self.SLIGHT_FLEX_ANGLE:
            positions.append('slightly flexed')
        elif self.NINETY_DEG_RANGE[0] <= angle <= self.NINETY_DEG_RANGE[1]:
            positions.append('bent at 90 degrees')
        elif angle < self.EXTENSION_ANGLE:
            positions.append('slightly extended')
        else:
            positions.append('extended')
            
        return positions

    def _get_hand_position(self, shoulder: Tuple[float, float], elbow: Tuple[float, float],
                          hand: Tuple[float, float], equipment_centers: List[Tuple[float, float]] = None) -> List[str]:
        """Determine hand position state."""
        positions = []
        
        # Calculate relative positions
        hand_elbow_diff = np.array(hand) - np.array(elbow)
        
        # Determine vertical/horizontal orientation
        if abs(hand_elbow_diff[1]) > abs(hand_elbow_diff[0]):
            # More vertical movement
            if hand_elbow_diff[1] < 0:
                positions.append('vertical upward')
            else:
                positions.append('vertical downward')
        else:
            # More horizontal movement
            if hand_elbow_diff[0] > 0:
                positions.append('horizontal outward')
            else:
                positions.append('horizontal inward')
        
        # Check chest/head position
        shoulder_hand_diff = np.array(hand) - np.array(shoulder)
        if shoulder_hand_diff[1] < -self.VERTICAL_THRESHOLD:
            positions.append('over head')
        elif abs(shoulder_hand_diff[0]) < self.HORIZONTAL_THRESHOLD:
            positions.append('over chest')
            
        # Check equipment holding
        if equipment_centers:
            for center in equipment_centers:
                if calculate_distance(hand, center) < self.EQUIPMENT_HOLD_THRESHOLD:
                    positions.append('holding equipment')
                    break
                    
        return positions

    def _detect_motion(self, prev_pos: Tuple[float, float], curr_pos: Tuple[float, float], 
                      shoulder: Tuple[float, float]) -> List[str]:
        """Detect motion patterns between frames."""
        motions = []
        
        # Calculate movement vectors
        movement = np.array(curr_pos) - np.array(prev_pos)
        relative_to_shoulder = np.array(curr_pos) - np.array(shoulder)
        
        # Vertical movements
        if abs(movement[1]) > self.VERTICAL_THRESHOLD:
            if movement[1] < 0:
                motions.append('vertical upward')
            else:
                motions.append('vertical downward')
                
        # Horizontal movements
        if abs(movement[0]) > self.HORIZONTAL_THRESHOLD:
            if movement[0] > 0:
                motions.append('horizontal outward')
            else:
                motions.append('horizontal inward')
                
        # Check for rowing motion
        if (movement[1] > self.VERTICAL_THRESHOLD and 
            abs(movement[0]) > self.HORIZONTAL_THRESHOLD):
            motions.append('row')
            
        # Check for flexion/extension
        if abs(movement[1]) < self.VERTICAL_THRESHOLD:
            if movement[0] < 0:
                motions.append('flexion')
            else:
                motions.append('extension')
                
        # If no significant movement detected
        if not motions:
            motions.append('stationary')
            
        return motions