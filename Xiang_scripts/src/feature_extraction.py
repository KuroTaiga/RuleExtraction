import logging
import cv2
import mediapipe as mp
import numpy as np
import sys

# 初始化 Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    """计算三点之间的角度"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def generate_joint_features(joint_positions, equipment, previous_joint_positions=None):
    """
    简化的特征提取函数，移除 movement 相关特征
    """
    features = {
        "equipment": equipment.lower(),
        "arm": {
            "position": "",  # up, down
            "state": "",    # straight, bent
        },
        "body": {
            "position": "",  # upright, bent
        },
        "leg": {
            "state": "",    # straight, bent
        }
    }

    def safe_get(dict_obj, key, default=None):
        return dict_obj.get(key, default)

    # 1. 分析手臂位置和状态
    left_shoulder = safe_get(joint_positions, 'left_shoulder')
    right_shoulder = safe_get(joint_positions, 'right_shoulder')
    left_elbow = safe_get(joint_positions, 'left_elbow')
    right_elbow = safe_get(joint_positions, 'right_elbow')
    left_wrist = safe_get(joint_positions, 'left_wrist')
    right_wrist = safe_get(joint_positions, 'right_wrist')

    if all([left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist]):
        # 判断手臂高度 - 简化为只判断高于或低于肩部
        avg_wrist_height = (left_wrist[1] + right_wrist[1]) / 2
        avg_shoulder_height = (left_shoulder[1] + right_shoulder[1]) / 2
        height_diff = avg_shoulder_height - avg_wrist_height

        if height_diff > 0.1:  # 手高于肩部
            features["arm"]["position"] = "up"
        else:  # 手低于肩部
            features["arm"]["position"] = "down"

        # 判断手臂伸直程度
        left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        avg_elbow_angle = (left_angle + right_angle) / 2

        if avg_elbow_angle > 150:
            features["arm"]["state"] = "straight"
        else:
            features["arm"]["state"] = "bent"

    # 2. 分析身体姿态
    hip_center = safe_get(joint_positions, 'hip_center')
    shoulder_center = safe_get(joint_positions, 'shoulder_center')

    if all([hip_center, shoulder_center]):
        # 计算躯干与垂直线的夹角
        torso_angle = calculate_angle(
            (shoulder_center[0], shoulder_center[1] - 0.5),
            shoulder_center,
            hip_center
        )

        # 简化为只判断直立或弯曲
        if torso_angle > 150:  # 更宽松的阈值
            features["body"]["position"] = "upright"
        else:
            features["body"]["position"] = "bent"

    # 3. 分析腿部状态
    left_hip = safe_get(joint_positions, 'left_hip')
    right_hip = safe_get(joint_positions, 'right_hip')
    left_knee = safe_get(joint_positions, 'left_knee')
    right_knee = safe_get(joint_positions, 'right_knee')
    left_ankle = safe_get(joint_positions, 'left_ankle')
    right_ankle = safe_get(joint_positions, 'right_ankle')

    if all([left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]):
        # 判断膝盖弯曲程度 - 简化为直或弯
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

        if avg_knee_angle > 150:  # 更宽松的阈值
            features["leg"]["state"] = "straight"
        else:
            features["leg"]["state"] = "bent"

    return features

def detect_limb_movement(current_points, previous_points, threshold=0.02):
    """
    检测肢体部分是否在运动
    """
    for curr, prev in zip(current_points, previous_points):
        if curr is not None and prev is not None:
            displacement = ((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)**0.5
            if displacement > threshold:
                return True
    return False

def aggregate_features(features_over_time):
    """
    简化版的特征聚合函数，只保留最后一帧的特征
    """
    if not features_over_time:
        return {}
    
    # 直接返回最后一帧的特征
    return features_over_time[-1]