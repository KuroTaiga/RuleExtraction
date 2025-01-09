# math_helper.py
import os
import mediapipe as mp
import math
import cv2
from typing import Tuple
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import json
import numpy as np

def calculate_rule_similarity(extracted_rules: dict, reference_rules: dict, 
                            weights: dict = {
                                'equipment': 4.0,  # 40%
                                'position': 4.0,   # 40%
                                'motion': 2.0      # 20%
                            }) -> float:
    """
    Calculate similarity between extracted and reference rules with weighted components.
    
    Args:
        extracted_rules: Rules extracted from video analysis
        reference_rules: Reference rules from exercise definitions
        weights: Component weights (should sum to 10.0)
    
    Returns:
        float: Similarity score between 0 and 1
    """
    total_weight = sum(weights.values())
    if abs(total_weight - 10.0) > 0.001:
        raise ValueError(f"Weights must sum to 10.0, got {total_weight}")

    # Calculate position similarity
    position_scores = []
    for joint in reference_rules:
        if joint not in extracted_rules:
            continue
            
        ref_positions = set(reference_rules[joint].get('position', []))
        ext_positions = set(extracted_rules[joint].get('position', []))
        
        if ref_positions and ext_positions:
            intersection = len(ref_positions.intersection(ext_positions))
            union = len(ref_positions.union(ext_positions))
            position_scores.append(intersection / union)
    
    # Calculate motion similarity
    motion_scores = []
    for joint in reference_rules:
        if joint not in extracted_rules:
            continue
            
        ref_motions = set(reference_rules[joint].get('motion', []))
        ext_motions = set(extracted_rules[joint].get('motion', []))
        
        if ref_motions and ext_motions:
            intersection = len(ref_motions.intersection(ext_motions))
            union = len(ref_motions.union(ext_motions))
            motion_scores.append(intersection / union)
    
    # Calculate average scores
    position_score = sum(position_scores) / len(position_scores) if position_scores else 0
    motion_score = sum(motion_scores) / len(motion_scores) if motion_scores else 0
    
    # Calculate final weighted score
    final_score = (
        (position_score * weights['position'] / 10.0) +
        (motion_score * weights['motion'] / 10.0)
    )
    
    return final_score

def calculate_equipment_similarity(extracted_equipment: dict, reference_equipment: dict) -> float:
    """
    Calculate similarity between the equipment used in extracted and reference rules.
    
    Args:
        extracted_rules: Rules extracted from video analysis, containing equipment type information.
        reference_rules: Reference rules for an exercise, containing equipment type information.
        
    Returns:
        float: Equipment similarity score between 0 and 1.
    """
    # Extract equipment types from the rules
    ref_equipment = set(reference_equipment.get("type", []))
    ext_equipment = set(extracted_equipment.get("type", []))
    
    # Calculate intersection over union for similarity
    if ref_equipment and ext_equipment:
        intersection = len(ref_equipment.intersection(ext_equipment))
        union = len(ref_equipment.union(ext_equipment))
        equipment_score = intersection / union
    else:
        equipment_score = 0.0  # No similarity if either set is empty
    
    return equipment_score*0.4

def calculate_similarity_with_details(extracted_rules: dict, reference_exercise: dict,
                                   weights: dict = {
                                       'equipment': 4.0,
                                       'position': 4.0,
                                       'motion': 2.0
                                   }) -> tuple:
    """
    Calculate similarity with detailed breakdown.
    
    Args:
        extracted_rules: Complete extracted rules including equipment
        reference_exercise: Complete reference exercise rules
        weights: Component weights
        
    Returns:
        tuple: (final_score, detailed_scores)
    """
    # Calculate equipment similarity
    equipment_score = 0.0
    if 'equipment' in extracted_rules and 'equipment' in reference_exercise:
        ext_equipment = set(extracted_rules['equipment']['type'])
        ref_equipment = set(reference_exercise['equipment']['type'])
        if ref_equipment and ext_equipment:
            intersection = len(ext_equipment.intersection(ref_equipment))
            union = len(ext_equipment.union(ref_equipment))
            equipment_score = intersection / union

    # Calculate position and motion similarities
    position_scores = {}
    motion_scores = {}
    
    for joint in reference_exercise['body_landmarks']:
        if joint not in extracted_rules['body_landmarks']:
            continue
            
        # Position similarity
        ref_positions = set(reference_exercise['body_landmarks'][joint].get('position', []))
        ext_positions = set(extracted_rules['body_landmarks'][joint].get('position', []))
        
        if ref_positions and ext_positions:
            intersection = len(ref_positions.intersection(ext_positions))
            union = len(ref_positions.union(ext_positions))
            position_scores[joint] = intersection / union
            
        # Motion similarity
        ref_motions = set(reference_exercise['body_landmarks'][joint].get('motion', []))
        ext_motions = set(extracted_rules['body_landmarks'][joint].get('motion', []))
        
        if ref_motions and ext_motions:
            intersection = len(ref_motions.intersection(ext_motions))
            union = len(ref_motions.union(ext_motions))
            motion_scores[joint] = intersection / union
    
    # Calculate average scores
    avg_position = sum(position_scores.values()) / len(position_scores) if position_scores else 0
    avg_motion = sum(motion_scores.values()) / len(motion_scores) if motion_scores else 0
    
    # Calculate weighted final score
    weighted_equipment = equipment_score * weights['equipment'] / 10.0
    weighted_position = avg_position * weights['position'] / 10.0
    weighted_motion = avg_motion * weights['motion'] / 10.0
    
    final_score = weighted_equipment + weighted_position + weighted_motion
    
    # Prepare detailed breakdown
    details = {
        'equipment': {
            'raw_score': equipment_score,
            'weighted_score': weighted_equipment,
            'weight': weights['equipment']
        },
        'position': {
            'raw_score': avg_position,
            'weighted_score': weighted_position,
            'weight': weights['position'],
            'joint_scores': position_scores
        },
        'motion': {
            'raw_score': avg_motion,
            'weighted_score': weighted_motion,
            'weight': weights['motion'],
            'joint_scores': motion_scores
        },
        'final_score': final_score
    }
    
    return final_score, details

def find_matching_activities(extracted_rules: Dict, results_json: Dict, top_n: int = 3) -> List[Tuple[str, float]]:
    """
    Find the top N matching activities from results.json.
    
    Args:
        extracted_rules: Rules extracted from video
        results_json: Complete results.json data
        top_n: Number of top matches to return
        
    Returns:
        List of tuples containing (activity_name, similarity_score)
    """
    scores = []
    
    for exercise in results_json['exercises']:
        activity_name = exercise['activity']
        reference_rules = exercise['body_landmarks']
        
        similarity = calculate_rule_similarity(extracted_rules, reference_rules)
        scores.append((activity_name, similarity))
    
    # Sort by similarity score and return top N
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]

def format_extracted_rules(arm_rules: Dict, leg_rules: Dict, torso_rules: Dict, 
                         equipment_detected: List[str]) -> Dict:
    """
    Format extracted rules to match results.json structure.
    
    Args:
        arm_rules: Rules extracted for arms
        leg_rules: Rules extracted for legs
        torso_rules: Rules extracted for torso
        equipment_detected: List of detected equipment
        
    Returns:
        Dictionary formatted like results.json entries
    """
    formatted_rules = {
        'body_landmarks': {},
        'equipment': {
            'type': equipment_detected
        },
        'other': {
            'mirrored': 'true'  # Default value, could be determined by analysis
        }
    }
    
    # Combine all rules
    formatted_rules['body_landmarks'].update(arm_rules)
    formatted_rules['body_landmarks'].update(leg_rules)
    formatted_rules['body_landmarks'].update(torso_rules)
    
    return formatted_rules

def calculate_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """
    Calculates the angle formed at point b by the line segments ba and bc.

    Args:
        a (Tuple[float, float]): Coordinates of point a (x, y).
        b (Tuple[float, float]): Coordinates of point b (x, y).
        c (Tuple[float, float]): Coordinates of point c (x, y).

    Returns:
        float: The angle in degrees between the lines ba and bc at point b.
    """
    # Calculate the angle using the arctangent of the determinant and dot product
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    dot_product = ab[0] * cb[0] + ab[1] * cb[1]
    determinant = ab[0] * cb[1] - ab[1] * cb[0]
    angle = math.degrees(math.atan2(determinant, dot_product))
    angle = angle + 360 if angle < 0 else angle
    return angle

def calculate_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """
    Calculates the Euclidean distance between two points.

    Args:
        a (Tuple[float, float]): Coordinates of point a (x, y).
        b (Tuple[float, float]): Coordinates of point b (x, y).

    Returns:
        float: The distance between points a and b.
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])

def get_joint_positions_from_video_old(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    joint_positions_over_time = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(image)

        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark
            joint_positions = {
                'left_shoulder': (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y),
                'right_shoulder': (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y),
                'left_elbow': (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y),
                'right_elbow': (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y),
                'left_wrist': (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y),
                'right_wrist': (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y),
                'left_hip': (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y),
                'right_hip': (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y),
                'left_knee': (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y),
                'right_knee': (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y),
                'left_ankle': (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y),
                'right_ankle': (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y),
                # 'nose': (landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                        #  landmarks[mp_pose.PoseLandmark.NOSE.value].y),
                'left_foot': (landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y),
                'right_foot': (landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y),
                'left_hand': (landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y),
                'right_hand': (landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y),
            }
            joint_positions_over_time.append(joint_positions)

    

    cap.release()

    return joint_positions_over_time

def calculate_match(actual,expected):
    return 0

def get_video_path(root_dir: str, target: list)->tuple:
    video_result = []
    exercise_result = []
    for dirpath,_,filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".mp4"):
                curr = os.path.splitext(filename)[0].lower()
                if curr in target:
                    exercise_result.append(curr)
                    video_result.append(os.path.join(dirpath,filename))
                else:
                    print(f"video name not found in exercise: {curr}")
    return video_result,exercise_result

def is_point_in_rectangle(point, rectangle_points):
    """
    Check if a point is inside a rectangle defined by four points.
    
    :param point: A list or tuple [x, y] representing the point to check.
    :param rectangle_points: A list of four points [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] representing the rectangle.
    :return: True if the point is inside the rectangle, False otherwise.
    """
    x, y = point
    
    # Extract x and y coordinates from the rectangle points
    x_coords = [p[0] for p in rectangle_points]
    y_coords = [p[1] for p in rectangle_points]
    
    # Get the boundaries of the rectangle
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    # Check if the point is within the bounds of the rectangle
    if min_x <= x <= max_x and min_y <= y <= max_y:
        return True
    return False

def load_exercise_rules(exercise_rules_file: str) -> List[Dict]:
    """
    Load exercise rules from a JSON file.
    
    Args:
        exercise_rules_file: Path to the exercise rules JSON file.
        
    Returns:
        List of exercise rules dictionaries.
    """
    with open(exercise_rules_file, 'r') as file:
        exercise_rules = json.load(file)
    return exercise_rules