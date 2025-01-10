import logging
import json
import cv2
import numpy as np
import os
import sys
from tqdm import tqdm
import mediapipe as mp
import signal
import subprocess
import pandas as pd
from collections import OrderedDict, defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import signal
import requests

# 更新为新电脑上的完整路径
BASE_PATH = '/home/bizon/xiang/new_computer'
SRC_PATH = os.path.join(BASE_PATH, 'src')
VIDEO_PATH = os.path.join(BASE_PATH, 'origin_test_video')
YOLO_PATH = os.path.join(BASE_PATH, 'yolov7')
YOLO_SCRIPT = os.path.join(YOLO_PATH, 'detect_revise.py')
YOLO_WEIGHTS = os.path.join(BASE_PATH, 'best.pt')
EXERCISE_JSON = os.path.join(BASE_PATH, 'exercise_generate_最新.json')
GENERATE_VIDEO_PATH = os.path.join(BASE_PATH, 'generate_video')

def verify_and_import():
    """验证所需路径和导入必要模块"""
    required_paths = {
        'Base Path': BASE_PATH,
        'Source Path': SRC_PATH,
        'Video Path': VIDEO_PATH,
        'YOLO Script': YOLO_SCRIPT,
        'YOLO Weights': YOLO_WEIGHTS,
        'Exercise JSON': EXERCISE_JSON,
        'Generate Video Path': GENERATE_VIDEO_PATH
    }

    for name, path in required_paths.items():
        if not os.path.exists(path):
            if name == 'Generate Video Path':
                print(f"创建输出视频目录: {path}")
                os.makedirs(path)
            else:
                raise Exception(f"Error: {name} not found at {path}")

    if SRC_PATH not in sys.path:
        sys.path.append(SRC_PATH)

    try:
        global generate_joint_features, aggregate_features, detect_equipment_via_script
        from feature_extraction import generate_joint_features, aggregate_features
        from equipment_detection import detect_equipment_via_script
        print("Successfully imported required modules from src")
    except ImportError as e:
        print(f"Error importing modules: {str(e)}")
        raise

def get_joint_positions_from_video(video_path):
    """使用与pose_processing.py相同的MediaPipe实现"""
    try:
        # 初始化MediaPipe，保持与pose_processing.py相同的参数
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=True
        )

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        print(f"处理视频: {os.path.basename(video_path)}")
        print(f"总帧数: {total_frames}, FPS: {fps}")

        joint_positions_over_time = []
        frame_count = 0

        # MediaPipe关键点映射
        keypoint_mapping = {
            'nose': mp_pose.PoseLandmark.NOSE,
            'left_shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'left_elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
            'right_elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
            'left_wrist': mp_pose.PoseLandmark.LEFT_WRIST,
            'right_wrist': mp_pose.PoseLandmark.RIGHT_WRIST,
            'left_hip': mp_pose.PoseLandmark.LEFT_HIP,
            'right_hip': mp_pose.PoseLandmark.RIGHT_HIP,
            'left_knee': mp_pose.PoseLandmark.LEFT_KNEE,
            'right_knee': mp_pose.PoseLandmark.RIGHT_KNEE,
            'left_ankle': mp_pose.PoseLandmark.LEFT_ANKLE,
            'right_ankle': mp_pose.PoseLandmark.RIGHT_ANKLE
        }

        with tqdm(total=total_frames) as pbar:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                # 转换BGR到RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    joint_positions = {}

                    # 提取基础关键点
                    for name, landmark_id in keypoint_mapping.items():
                        landmark = results.pose_landmarks.landmark[landmark_id]
                        joint_positions[name] = (landmark.x, landmark.y)

                    # 计算合成关键点
                    shoulder_center_x = (joint_positions['left_shoulder'][0] + joint_positions['right_shoulder'][0]) / 2
                    shoulder_center_y = (joint_positions['left_shoulder'][1] + joint_positions['right_shoulder'][1]) / 2
                    hip_center_x = (joint_positions['left_hip'][0] + joint_positions['right_hip'][0]) / 2
                    hip_center_y = (joint_positions['left_hip'][1] + joint_positions['right_hip'][1]) / 2

                    # 添加合成关键点
                    joint_positions.update({
                        'spine': (shoulder_center_x, shoulder_center_y),
                        'shoulder_center': (shoulder_center_x, shoulder_center_y),
                        'hip_center': (hip_center_x, hip_center_y),
                        'left_foot': joint_positions['left_ankle'],
                        'right_foot': joint_positions['right_ankle'],
                        'left_hand': joint_positions['left_wrist'],
                        'right_hand': joint_positions['right_wrist'],
                        'mid_hip': (hip_center_x, hip_center_y),
                        'mid_shoulder': (shoulder_center_x, shoulder_center_y),
                        'neck': (shoulder_center_x, shoulder_center_y)
                    })

                    joint_positions_over_time.append(joint_positions)

                frame_count += 1
                pbar.update(1)

        cap.release()
        pose.close()

        print(f"成功从 {len(joint_positions_over_time)} 帧中提取姿态数据")
        return joint_positions_over_time

    except Exception as e:
        print(f"处理视频时出错: {str(e)}")
        return []

def convert_features_to_text(features):
    """将特征转换为文本表示"""
    text = f"equipment:{features.get('equipment', 'none')} "

    for category in ['pose', 'movement']:
        if category in features:
            for key, value in features[category].items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        text += f"{category}_{key}_{sub_key}:{sub_value} "
                elif isinstance(value, list):
                    text += f"{category}_{key}:{' '.join(map(str, value))} "
                else:
                    text += f"{category}_{key}:{value} "

    if 'joint_actions' in features:
        for joint, actions in features['joint_actions'].items():
            if actions:
                text += f"{joint}_actions:{' '.join(map(str, actions))} "
            else:
                text += f"{joint}_actions:none "

    return text.strip()

def collect_video_files():
    """收集视频文件"""
    if not os.path.exists(VIDEO_PATH):
        raise Exception(f"Video directory not found: {VIDEO_PATH}")

    exercise_rules = build_exercise_rules()
    allowed_exercises = list(exercise_rules.keys())

    video_files = []
    exercise_names = []

    for file in os.listdir(VIDEO_PATH):
        if file.endswith(".mp4"):
            matched_exercise = next(
                (exercise for exercise in allowed_exercises
                 if exercise.lower() in file.lower()),
                None
            )

            if matched_exercise:
                video_path = os.path.join(VIDEO_PATH, file)
                video_files.append(video_path)
                exercise_names.append(matched_exercise)

    return video_files, exercise_names

def build_exercise_rules():
    """读取运动规则JSON文件"""
    try:
        with open(EXERCISE_JSON, 'r', encoding='utf-8') as f:
            return json.load(f, object_pairs_hook=OrderedDict)
    except Exception as e:
        print(f"Error loading exercise.json: {e}")
        return OrderedDict()

def calculate_custom_similarity(video_vector, exercise_vectors, video_features, exercise_rules):
    """计算自定义相似度"""
    base_similarities = cosine_similarity(video_vector, exercise_vectors)[0]
    adjusted_similarities = base_similarities.copy()

    for i, exercise in enumerate(exercise_rules):
        rule = exercise_rules[exercise]
        score = 1.0

        # 检查关键特征匹配
        for key_feature in ["movement.primary", "pose.torso", "pose.arms.shoulder_position"]:
            if key_feature in rule and key_feature in video_features:
                if rule[key_feature] == video_features[key_feature]:
                    score *= 1.2
                else:
                    score *= 0.8

        # 检查设备匹配
        if rule.get("equipment", "") == video_features.get("equipment", ""):
            score *= 1.5
        else:
            score *= 0.5

        adjusted_similarities[i] *= score

    return adjusted_similarities

def process_joint_positions(joint_positions_over_time, method='mediapipe'):
    """
    使用与pose_processing.py相同的滤波器实现

    Args:
        joint_positions_over_time: 原始关节位置数据
        method: 'mediapipe', 'bessel', 或 'butterworth'
    """
    if len(joint_positions_over_time) == 0:
        return []

    if method == 'mediapipe':
        return joint_positions_over_time

    # 转换数据格式
    joint_names = list(joint_positions_over_time[0].keys())
    x_series = {joint: [] for joint in joint_names}
    y_series = {joint: [] for joint in joint_names}

    for frame in joint_positions_over_time:
        for joint in joint_names:
            x, y = frame[joint]
            x_series[joint].append(x)
            y_series[joint].append(y)

    # 设置滤波器参数（与pose_processing.py保持一致）
    order = 4
    cutoff = 0.1

    processed_positions = []
    for i in range(len(joint_positions_over_time)):
        new_positions = {}

        for joint in joint_names:
            x_data = np.array(x_series[joint])
            y_data = np.array(y_series[joint])

            if method == 'bessel':
                # Bessel滤波器实现
                b, a = signal.bessel(order, cutoff, 'low')
                x_filtered = signal.filtfilt(b, a, x_data)
                y_filtered = signal.filtfilt(b, a, y_data)

            elif method == 'butterworth':
                # Butterworth滤波器实现
                b, a = signal.butter(order, cutoff, 'low')
                x_filtered = signal.filtfilt(b, a, x_data)
                y_filtered = signal.filtfilt(b, a, y_data)

            new_positions[joint] = (float(x_filtered[i]), float(y_filtered[i]))

        processed_positions.append(new_positions)

    return processed_positions

def main():
    try:
        # 验证环境
        verify_and_import()

        # 加载运动规则
        exercise_rules = build_exercise_rules()
        if not exercise_rules:
            print("Failed to load exercise rules. Exiting.")
            return

        exercise_names = list(exercise_rules.keys())
        print(f"Loaded {len(exercise_names)} exercise rules")

        # 收集视频文件
        video_files, actual_exercises = collect_video_files()
        if not video_files:
            print("No video files found")
            return

        print(f"Found {len(video_files)} video files")

        # 准备vectorizer
        exercise_texts = []
        for exercise, features in exercise_rules.items():
            text = convert_features_to_text(features)
            exercise_texts.append(text)

        vectorizer = TfidfVectorizer()
        vectorizer.fit(exercise_texts)
        exercise_vectors = vectorizer.transform(exercise_texts)

        # 设置输出路径
        output_path = os.path.join(BASE_PATH, 'exercise_recognition_results_(orginal).xlsx')
        existing_df = pd.DataFrame() if not os.path.exists(output_path) else pd.read_excel(output_path)
        processed_videos = set(existing_df['Video Path']) if not existing_df.empty else set()

        results = []

        # 处理每个视频
        for video_path, exercise_name in zip(video_files, actual_exercises):
            if video_path in processed_videos:
                print(f"Skipping already processed video: {video_path}")
                continue

            print(f"\nProcessing video: {video_path}")

            try:
                # 提取姿态
                joint_positions = get_joint_positions_from_video(video_path)
                if not joint_positions:
                    print("Failed to extract pose from video")
                    continue

                # 检测器材
                detected_equipment = detect_equipment_via_script(
                    YOLO_SCRIPT,
                    YOLO_WEIGHTS,
                    video_path
                )

                # 生成特征
                features = []
                prev_positions = None
                for positions in joint_positions:
                    feature = generate_joint_features(
                        positions,
                        detected_equipment,
                        prev_positions
                    )
                    features.append(feature)
                    prev_positions = positions

                # 聚合特征
                aggregated_features = aggregate_features(features)
                aggregated_features_json = json.dumps(aggregated_features, ensure_ascii=False)

                # 计算相似度
                video_text = convert_features_to_text(aggregated_features)
                video_vector = vectorizer.transform([video_text])
                similarities = calculate_custom_similarity(
                    video_vector,
                    exercise_vectors,
                    aggregated_features,
                    exercise_rules
                )

                # 获取Top3结果
                top_indices = similarities.argsort()[-3:][::-1]
                top_three = [(exercise_names[i], float(similarities[i] * 100)) for i in top_indices]

                # 创建结果
                is_correct = exercise_name in [ex for ex, _ in top_three]
                result = OrderedDict({
                    "Exercise Name": exercise_name,
                    "Video Path": video_path,
                    "Detected Equipment": detected_equipment,
                    "Actual in Top 3": "Yes" if is_correct else "No",
                })

                for ex in exercise_names:
                    score = next((score for pred_ex, score in top_three if pred_ex == ex), 0)
                    result[f"{ex} Score"] = f"{score:.2f}%"

                result.update({
                    "Detected Features": aggregated_features_json,
                    "Exercise.json Content": json.dumps(exercise_rules[exercise_name], ensure_ascii=False)
                })

                results.append(result)

                # 更新Excel
                updated_df = pd.concat([existing_df, pd.DataFrame([result])], ignore_index=True)
                updated_df.to_excel(output_path, index=False)
                existing_df = updated_df

                print(f"Results saved to {output_path}")
                print("Top 3 predictions:")
                for i, (ex, score) in enumerate(top_three, 1):
                    print(f"{i}. {ex}: {score:.2f}%")

            except Exception as e:
                print(f"Error processing video {video_path}: {str(e)}")
                continue

        # 计算最终统计
        total_videos = len(results)
        if total_videos > 0:
            correct_predictions = sum(1 for r in results if r["Actual in Top 3"] == "Yes")
            accuracy = (correct_predictions / total_videos) * 100

            print(f"\nFinal Statistics:")
            print(f"Total videos processed: {total_videos}")
            print(f"Correct predictions (in top 3): {correct_predictions}")
            print(f"Top-3 Accuracy: {accuracy:.2f}%")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        raise