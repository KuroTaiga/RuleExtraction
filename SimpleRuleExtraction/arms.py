#Extract rules for arm
#Author: Jiankun Dong
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
        
            
        return positions

    def _get_hand_position(self, shoulder: Tuple[float, float], elbow: Tuple[float, float],
                          hand: Tuple[float, float], equipment_centers: List[Tuple[float, float]] = None) -> List[str]:
        """Determine hand position state."""
        positions = []
        
        # Calculate relative positions
        hand_elbow_diff = np.array(hand) - np.array(elbow)
        
        return positions

    def _detect_motion(self, prev_pos: Tuple[float, float], curr_pos: Tuple[float, float], 
                      shoulder: Tuple[float, float]) -> List[str]:
        """Detect motion patterns between frames."""
        motions = []
        
        # Calculate movement vectors
        movement = np.array(curr_pos) - np.array(prev_pos)
        
            
        return motions