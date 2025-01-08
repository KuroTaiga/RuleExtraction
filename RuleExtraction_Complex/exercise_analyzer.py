from arms import ArmRules
from legs import LegRules
from torso import TorsoRules
from helper import *
from collections import Counter, defaultdict
import json
import cv2
import numpy as np
from typing import Dict, List, Tuple

class ExerciseRuleExtractor:
    def __init__(self):
        """Initialize all rule extractors and tracking state."""
        self.arm_rules = ArmRules()
        self.leg_rules = LegRules()
        self.torso_rules = TorsoRules()
        self.current_activity = None
        
    def extract_all_rules(self, landmarks: Dict, equipment_centers: List[Tuple[float, float]] = None) -> Dict:
        """
        Extract all rules for the current frame and format them according to results.json structure.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            equipment_centers: Optional list of equipment center points
            
        Returns:
            Dictionary formatted like results.json entries
        """
        # Extract rules from each component
        arm_positions = self.arm_rules.extract_arm_rules(landmarks, equipment_centers)
        leg_positions = self.leg_rules.extract_leg_rules(landmarks)
        torso_positions = self.torso_rules.extract_torso_rules(landmarks)
        
        # Combine all rules into results.json format
        body_landmarks = {}
        
        # Add arm rules
        for key in ['left_elbow', 'right_elbow', 'left_hand', 'right_hand']:
            if key in arm_positions:
                body_landmarks[key] = arm_positions[key]
        
        # Add leg rules
        for key in ['left_knee', 'right_knee', 'left_foot', 'right_foot']:
            if key in leg_positions:
                body_landmarks[key] = leg_positions[key]
        
        # Add torso rules
        for key in ['torso', 'left_hip', 'right_hip', 'left_shoulder', 'right_shoulder']:
            if key in torso_positions:
                body_landmarks[key] = torso_positions[key]
        
        return body_landmarks

def process_video_with_rules(video_path: str, equipment_detector, pose_detector) -> Dict:
    """
    Process video and extract exercise rules with equipment detection.
    
    Args:
        video_path: Path to the video file
        equipment_detector: Initialized equipment detector
        pose_detector: Initialized pose detector
        
    Returns:
        Dictionary containing complete exercise analysis
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    
    rule_extractor = ExerciseRuleExtractor()
    equipment_counter = Counter()
    all_frames_landmarks = []
    aggregated_landmarks = defaultdict(lambda: defaultdict(list))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect equipment and pose
        detected_equipment = equipment_detector.detect_equipment(frame)
        joint_positions = pose_detector.get_joint_positions_from_frame(frame)
        
        if joint_positions:
            # Get equipment centers (if implemented in your equipment detector)
            equipment_centers = []  # Implement if available from your detector
            
            # Extract rules for current frame
            frame_rules = rule_extractor.extract_all_rules(joint_positions, equipment_centers)
            
            # Aggregate rules across frames
            for landmark, data in frame_rules.items():
                for rule_type in ['position', 'motion']:
                    if rule_type in data:
                        aggregated_landmarks[landmark][rule_type].extend(data[rule_type])
            
            # Update equipment counter
            equipment_counter.update(detected_equipment)
    
    cap.release()
    
    # Process aggregated rules
    final_rules = {
        'body_landmarks': {},
        'equipment': {
            'type': [eq for eq, count in equipment_counter.most_common(2)]  # Get top 2 most common equipment
        },
        'other': {
            'mirrored': 'true'  # Could be determined by analysis
        }
    }
    
    # Convert aggregated landmarks to most common rules
    for landmark, data in aggregated_landmarks.items():
        final_rules['body_landmarks'][landmark] = {
            'position': [pos for pos, _ in Counter(data['position']).most_common(3)],
            'motion': [mot for mot, _ in Counter(data['motion']).most_common(3)]
        }
    
    return final_rules

def compare_with_reference_exercises(extracted_rules: Dict, reference_json: str, top_n: int = 3) -> List[Tuple[str, float]]:
    """
    Compare extracted rules with reference exercises and find best matches.
    
    Args:
        extracted_rules: Dictionary of extracted rules
        reference_json: Path to reference results.json
        top_n: Number of top matches to return
        
    Returns:
        List of (exercise_name, similarity_score) tuples
    """
    with open(reference_json, 'r') as f:
        reference_data = json.load(f)
    
    matches = []
    detailed_matches = {}
    
    for exercise in reference_data:
        score, details = calculate_similarity_with_details(extracted_rules, exercise)
        matches.append((exercise['activity'], score))
        detailed_matches[exercise['activity']] = details
    return sorted(matches, key=lambda x: x[1], reverse=True)[:top_n]