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

BASE_PATH = '/home/bizon/dong/RuleExtraction/Xiang_scripts'
SRC_PATH = os.path.join(BASE_PATH, 'src')
VIDEO_PATH = os.path.join(BASE_PATH, 'videos/Jan13_Z200_Videos')
YOLO_PATH = os.path.join(BASE_PATH, 'yolov7')
YOLO_SCRIPT = os.path.join(YOLO_PATH, 'detect_revise.py')
YOLO_WEIGHTS = os.path.join('/home/bizon/xiang/new_computer/', 'best.pt')
EXERCISE_JSON = os.path.join(BASE_PATH, 'exercise_generate_最新.json')
GENERATE_VIDEO_PATH = os.path.join(BASE_PATH, 'generate_video/Jan13_Z200_Videos')

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
        # 修改全局变量声明
        global generate_joint_features, detect_equipment_via_script
        from feature_extraction import generate_joint_features
        from equipment_detection import detect_equipment_via_script  # 确保这行正确导入
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
                    for name, landmark_id in mp_pose.PoseLandmark.__members__.items():
                        landmark = results.pose_landmarks.landmark[landmark_id]
                        joint_positions[name.lower()] = (landmark.x, landmark.y)
                    
                    # 添加计算合成关键点的代码
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
    """将特征转换为文本表示，按照json文件的顺序，处理空值"""
    # 定义标准顺序
    standard_order = {
        "pose": ["hand", "foot", "torso", "hip", "knee", "shoulder", "elbow", "spine", "arms"],
        "arms": ["shoulder_position", "elbow_position", "wrist_position"],
        "movement": ["primary", "secondary"],
        "joint_actions": ["spine", "shoulder", "hip", "knee", "elbow", "wrist", "ankle"]
    }

    def clean_value(v):
        """清理值，去除数字和统计信息"""
        if isinstance(v, tuple):
            v = v[0]  # 只取描述文本
        return str(v) if not isinstance(v, (int, float)) and not str(v).replace('.', '').isdigit() else None

    text_parts = []
    
    # 设备检测（总是第一个）
    equipment = features.get('equipment', 'none')
    text_parts.append(f"equipment:{equipment}")
    
    # 按照标准顺序处理pose
    if 'pose' in features:
        for key in standard_order["pose"]:
            if key in features['pose']:
                value = features['pose'][key]
                if key == "arms" and isinstance(value, dict):
                    for arm_key in standard_order["arms"]:
                        arm_value = value.get(arm_key, '')
                        if arm_value:
                            text_parts.append(f"pose_{key}_{arm_key}:{clean_value(arm_value)}")
                else:
                    if isinstance(value, (list, tuple)):
                        # 清理列表值
                        value_list = [clean_value(v) for v in value if v]
                        value_list = [v for v in value_list if v]  # 移除None
                        if value_list:
                            text_parts.append(f"pose_{key}:{' '.join(value_list)}")
                    elif value:
                        cleaned = clean_value(value)
                        if cleaned:
                            text_parts.append(f"pose_{key}:{cleaned}")

    # 按照标准顺序处理movement
    if 'movement' in features:
        for key in standard_order["movement"]:
            if key in features['movement']:
                value = features['movement'][key]
                if isinstance(value, (list, tuple)):
                    # 过滤掉空值
                    value_list = [str(v) for v in value if v]
                    value_str = ' '.join(value_list) if value_list else 'none'
                else:
                    value_str = str(value) if value else 'none'
                if value_str != 'none':  # 只添加非空值
                    text_parts.append(f"movement_{key}:{value_str}")
    
    # 按照标准顺序处理joint_actions
    if 'joint_actions' in features:
        for joint in standard_order["joint_actions"]:
            if joint in features['joint_actions']:
                actions = features['joint_actions'][joint]
                if actions:
                    if isinstance(actions, (list, tuple)):
                        # 过滤掉空值
                        action_list = [str(a) for a in actions if a]
                        if action_list:  # 只添加非空列表
                            text_parts.append(f"{joint}_actions:{' '.join(action_list)}")
                    else:
                        text_parts.append(f"{joint}_actions:{actions}")

    return ' '.join(text_parts)

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

def apply_butterworth_filter(joint_positions_sequence, fps):
    try:
        from scipy import signal
        
        # 确保数据不为空
        if not joint_positions_sequence:
            return joint_positions_sequence
            
        # 创建一个新的序列来存储滤波后的数据
        filtered_sequence = []
        
        # 设置滤波器参数
        order = 4  # 滤波器阶数
        cutoff = 7.0  # 截止频率
        nyquist = fps / 2.0
        normalized_cutoff = cutoff / nyquist
        
        # 创建Butterworth滤波器
        b, a = signal.butter(order, normalized_cutoff, btype='low')
        
        # 对每个关键点分别进行滤波
        for frame in range(len(joint_positions_sequence)):
            filtered_positions = {}
            
            # 获取第一个有效帧的关键点列表
            if frame == 0:
                keypoints = joint_positions_sequence[frame].keys()
            
            for keypoint in keypoints:
                # 收集该关键点在所有帧中的x和y坐标
                x_coords = []
                y_coords = []
                
                for positions in joint_positions_sequence:
                    if keypoint in positions and positions[keypoint] is not None:
                        x, y = positions[keypoint]
                        x_coords.append(x)
                        y_coords.append(y)
                    else:
                        # 如果某帧缺失该关键点，使用前一帧的值
                        if x_coords:
                            x_coords.append(x_coords[-1])
                            y_coords.append(y_coords[-1])
                        else:
                            x_coords.append(0.0)
                            y_coords.append(0.0)
                
                # 应用滤波器
                if len(x_coords) > order:
                    filtered_x = signal.filtfilt(b, a, x_coords)
                    filtered_y = signal.filtfilt(b, a, y_coords)
                    filtered_positions[keypoint] = (filtered_x[frame], filtered_y[frame])
                else:
                    # 如果数据点太少，保持原始值
                    filtered_positions[keypoint] = (x_coords[frame], y_coords[frame])
            
            filtered_sequence.append(filtered_positions)
        
        return filtered_sequence
        
    except Exception as e:
        print(f"滤波处理出错，使用原始数据: {str(e)}")
        return joint_positions_sequence

def process_video_direct(video_name):
    try:
        print(f"\n开始处理视频: {video_name}")
        
        # 设置路径
        input_video_path = os.path.join(VIDEO_PATH, video_name)
        base_name = os.path.splitext(video_name)[0]
        filtered_path = os.path.join(GENERATE_VIDEO_PATH, f"{base_name}_filtered.mp4")
        features_path = os.path.join(GENERATE_VIDEO_PATH, f"{base_name}_features.mp4")

        # 获取视频信息
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件: {video_name}")
            return None

        # 检测器材 - 直接使用1.py中的方式
        try:
            detected_equipment = detect_equipment_via_script(
                YOLO_SCRIPT,
                YOLO_WEIGHTS,
                input_video_path
            )
            print(f"检测到的设备: {detected_equipment}")
        except Exception as e:
            print(f"设备检测失败: {str(e)}")
            detected_equipment = None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 第一步：收集所有帧的关节位置数据
        print("收集关节位置数据...")
        joint_positions_sequence = []
        valid_frames = 0
        total_frames_processed = 0
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        with tqdm(total=total_frames, desc="收集数据") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                total_frames_processed += 1
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                
                if results.pose_landmarks:
                    valid_frames += 1
                    positions = {}
                    for name, landmark_id in mp_pose.PoseLandmark.__members__.items():
                        landmark = results.pose_landmarks.landmark[landmark_id]
                        positions[name.lower()] = (landmark.x, landmark.y)
                    
                    # 添加合成关键点
                    shoulder_center_x = (positions['left_shoulder'][0] + positions['right_shoulder'][0]) / 2
                    shoulder_center_y = (positions['left_shoulder'][1] + positions['right_shoulder'][1]) / 2
                    hip_center_x = (positions['left_hip'][0] + positions['right_hip'][0]) / 2
                    hip_center_y = (positions['left_hip'][1] + positions['right_hip'][1]) / 2
                    
                    positions.update({
                        'spine': (shoulder_center_x, shoulder_center_y),
                        'shoulder_center': (shoulder_center_x, shoulder_center_y),
                        'hip_center': (hip_center_x, hip_center_y),
                        'left_foot': positions['left_ankle'],
                        'right_foot': positions['right_ankle'],
                        'left_hand': positions['left_wrist'],
                        'right_hand': positions['right_wrist'],
                        'mid_hip': (hip_center_x, hip_center_y),
                        'mid_shoulder': (shoulder_center_x, shoulder_center_y),
                        'neck': (shoulder_center_x, shoulder_center_y)
                    })
                    
                    joint_positions_sequence.append(positions)
                else:
                    joint_positions_sequence.append(None)  # 使用None标记无效帧
                
                pbar.update(1)

        # 验证数据质量
        valid_frame_ratio = valid_frames / total_frames_processed if total_frames_processed > 0 else 0
        print(f"\n有效帧率: {valid_frame_ratio:.2%} ({valid_frames}/{total_frames_processed})")
        
        if valid_frame_ratio < 0.5:  # 如果有效帧少于50%
            print(f"警告: 视频 {video_name} 的有效识别帧太少，无法生成可靠的分析结果")
            return None

        # 应用Butterworth滤波 (只对有效帧进行滤波)
        print("应用Butterworth滤波...")
        valid_positions = [pos for pos in joint_positions_sequence if pos is not None]
        if len(valid_positions) > 0:
            filtered_positions = apply_butterworth_filter(valid_positions, fps)
        else:
            print("错误: 没有足够的有效帧用于滤波")
            return None

        # 重置视频捕获器
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out_filtered = cv2.VideoWriter(filtered_path, fourcc, fps, (width, height))
        out_features = cv2.VideoWriter(features_path, fourcc, fps, (width, height))

        if not out_filtered.isOpened() or not out_features.isOpened():
            # 如果 MP4V 失败，尝试 MJPG
            print("MP4V编码器失败，尝试MJPG...")
            filtered_path = os.path.join(GENERATE_VIDEO_PATH, f"{base_name}_filtered.avi")
            features_path = os.path.join(GENERATE_VIDEO_PATH, f"{base_name}_features.avi")
            
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out_filtered = cv2.VideoWriter(filtered_path, fourcc, fps, (width, height))
            out_features = cv2.VideoWriter(features_path, fourcc, fps, (width, height))
            
            if not out_filtered.isOpened() or not out_features.isOpened():
                print("错误: 无法创建视频写入器")
                return None

        # 第二步：使用滤波后的数据生成视频
        print("生成视频...")
        frame_idx = 0
        filtered_idx = 0
        prev_positions = None  # 添加前一帧位置记录，移到循环外部
        
        with tqdm(total=total_frames, desc="生成视频") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 1. 生成滤波后的姿态视频
                filtered_frame = frame.copy()
                
                # 只在有效帧上绘制姿态
                if joint_positions_sequence[frame_idx] is not None and filtered_idx < len(filtered_positions):
                    positions = filtered_positions[filtered_idx]
                    # 绘制关键点和骨架
                    for joint_name, (x, y) in positions.items():
                        x_px = int(x * width)
                        y_px = int(y * height)
                        cv2.circle(filtered_frame, (x_px, y_px), 4, (0, 255, 0), -1)
                    
                    # 绘制骨架连接
                    connections = [
                        ('left_shoulder', 'right_shoulder'),
                        ('left_shoulder', 'left_elbow'),
                        ('right_shoulder', 'right_elbow'),
                        ('left_elbow', 'left_wrist'),
                        ('right_elbow', 'right_wrist'),
                        ('left_shoulder', 'left_hip'),
                        ('right_shoulder', 'right_hip'),
                        ('left_hip', 'right_hip'),
                        ('left_hip', 'left_knee'),
                        ('right_hip', 'right_knee'),
                        ('left_knee', 'left_ankle'),
                        ('right_knee', 'right_ankle')
                    ]
                    
                    for start_joint, end_joint in connections:
                        if (start_joint in positions and end_joint in positions):
                            start_pos = positions[start_joint]
                            end_pos = positions[end_joint]
                            start_px = (int(start_pos[0] * width), int(start_pos[1] * height))
                            end_px = (int(end_pos[0] * width), int(end_pos[1] * height))
                            cv2.line(filtered_frame, start_px, end_px, (0, 255, 0), 2)

                    filtered_idx += 1
                else:
                    # 在无效帧上显示警告
                    cv2.putText(filtered_frame, "No pose detected", 
                              (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, (0, 0, 255), 2)

                out_filtered.write(filtered_frame)

                # 2. 生成特征显示视频
                feature_frame = np.zeros((height, width, 3), dtype=np.uint8)
                if joint_positions_sequence[frame_idx] is not None and filtered_idx < len(filtered_positions):
                    features = generate_joint_features(
                        joint_positions_sequence[frame_idx],
                        detected_equipment,
                        prev_positions
                    )
                    prev_positions = joint_positions_sequence[frame_idx].copy()
                    filtered_features = filter_features(features)
                    
                    if filtered_features:
                        # 显示简化后的特征信息
                        y_offset = 30
                        line_height = 25
                        
                        # 显示帧号
                        cv2.putText(feature_frame, f"Frame: {frame_idx}", 
                                  (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.6, (255, 255, 255), 1)
                        y_offset += line_height

                        # 显示设备
                        if 'equipment' in filtered_features:
                            cv2.putText(feature_frame, f"Equipment: {filtered_features['equipment']}", 
                                      (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.6, (255, 255, 255), 1)
                            y_offset += line_height

                        # 显示手臂信息
                        if 'arm' in filtered_features:
                            cv2.putText(feature_frame, "ARM:", 
                                      (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.6, (255, 255, 255), 1)
                            y_offset += line_height
                            
                            for key, value in filtered_features['arm'].items():
                                cv2.putText(feature_frame, f"  {key}: {value}", 
                                          (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.6, (255, 255, 255), 1)
                                y_offset += line_height

                        # 显示身体信息
                        if 'body' in filtered_features:
                            cv2.putText(feature_frame, "BODY:", 
                                      (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.6, (255, 255, 255), 1)
                            y_offset += line_height
                            
                            cv2.putText(feature_frame, f"  position: {filtered_features['body']['position']}", 
                                      (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.6, (255, 255, 255), 1)
                            y_offset += line_height

                        # 显示腿部信息
                        if 'leg' in filtered_features:
                            cv2.putText(feature_frame, "LEG:", 
                                      (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.6, (255, 255, 255), 1)
                            y_offset += line_height
                            
                            for key, value in filtered_features['leg'].items():
                                cv2.putText(feature_frame, f"  {key}: {value}", 
                                          (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.6, (255, 255, 255), 1)
                                y_offset += line_height

                        # 每30帧打印一次调试信息
                        if frame_idx % 30 == 0:
                            print(f"\nFrame {frame_idx} 调试信息:")
                            print(f"设备检测结果: {detected_equipment}")
                            print(f"特征数据: {json.dumps(filtered_features, indent=2, ensure_ascii=False)}")
                    else:
                        cv2.putText(feature_frame, "No significant features", 
                                  (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 
                                  1, (0, 0, 255), 2)

                out_features.write(feature_frame)
                frame_idx += 1
                pbar.update(1)

        # 释放资源
        cap.release()
        out_filtered.release()
        out_features.release()
        pose.close()

        print("已生成视频:")
        print(f"滤波后的姿态视频: {filtered_path}")
        print(f"特征信息视频: {features_path}")

        return [filtered_path, features_path]

    except Exception as e:
        print(f"\n处理 {video_name} 时出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

# 在特征显示部分添加过滤函数
def filter_features(features):
    """过滤掉空的特征，适配新的简化特征结构"""
    if not features:
        return {}
        
    filtered = {}
    
    # 处理equipment
    if 'equipment' in features and features['equipment']:
        filtered['equipment'] = features['equipment']
    
    # 处理简化后的三个主要类别：arm, body, leg
    if 'arm' in features:
        arm_features = {}
        for key in ['position', 'state', 'moving']:
            if features['arm'].get(key):
                arm_features[key] = features['arm'][key]
        if arm_features:
            filtered['arm'] = arm_features
    
    # 处理body
    if 'body' in features:
        body_features = {}
        if features['body'].get('position'):
            body_features['position'] = features['body']['position']
        if body_features:
            filtered['body'] = body_features
    
    # 处理leg
    if 'leg' in features:
        leg_features = {}
        for key in ['state', 'moving']:
            if features['leg'].get(key):
                leg_features[key] = features['leg'][key]
        if leg_features:
            filtered['leg'] = leg_features
    
    return filtered

if __name__ == "__main__":
    try:
        print("Starting program...")
        verify_and_import()
        
        # 确保输出目录存在
        if not os.path.exists(GENERATE_VIDEO_PATH):
            os.makedirs(GENERATE_VIDEO_PATH)
            print(f"创建输出目录: {GENERATE_VIDEO_PATH}")
        
        # 获取所有视频文件
        video_files = [f for f in os.listdir(VIDEO_PATH) if f.endswith('.mp4')] #change this
        print(f"\n找到 {len(video_files)} 个视频文件")
        
        # 处理所有视频
        successful = 0
        failed = 0
        skipped = 0
        failed_videos = []
        skipped_videos = []
        for video in video_files:
            output_path = process_video_direct(video)
            if output_path:
                successful += 1
            elif output_path is None and os.path.exists(os.path.join(GENERATE_VIDEO_PATH, f"processed_{video}")):
                skipped += 1
                skipped_videos.append(video)
            else:
                failed += 1
                failed_videos.append(video)
        
        print("\n处理完成!")
        print(f"成功: {successful}")
        print(f"失败: {failed}")
        print(f"跳过: {skipped}")
        print(f"输出目录: {GENERATE_VIDEO_PATH}")
        print(f"Failed Videos: {failed_videos}")        
    except Exception as e:
        print(f"程序出错: {str(e)}")
        import traceback
        print(traceback.format_exc())