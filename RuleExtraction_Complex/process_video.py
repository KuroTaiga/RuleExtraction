import cv2
import json
import sys
sys.path.insert(0, './yolov7')
#sys.path.insert(0, './RuleExtraction/yolov7')
from collections import Counter
from GYMDetector import YOLOv7EquipmentDetector, PoseDetector
from arms import ArmRules
from legs import LegRules
from torso import TorsoRules
from helper import get_empty_landmarks

def process_video(eqpt_detector: YOLOv7EquipmentDetector, pose_detector: PoseDetector, video_path: str, log_to_file=False):
    """
    Process an MP4 video, detect equipment, poses, and other attributes. Visualizes the video with overlaid pose landmarks,
    detected equipment, and extracted rules.

    :param eqpt_detector: YOLOv7EquipmentDetector for equipment detection.
    :param pose_detector: PoseDetector for pose detection.
    :param video_path: Path to the MP4 video file.
    :param log_to_file: Boolean to indicate if logs should be saved to a file.
    :return: Tuple of (pose_dict, equipment_dict, other_dict)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Error: Could not open video file {video_path}")

    frame_count = 0
    equipment_counter = Counter()
    pose_dict = get_empty_landmarks()
    all_frames_landmarks = []
    arm_rules = ArmRules()
    leg_rules = LegRules()
    torso_rules = TorsoRules()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detect equipment in the current frame
        detected_equipment = eqpt_detector.detect_equipment(frame)

        # Detect pose landmarks in the current frame
        # joint_position_from_frame = pose_detector.get_joint_positions_from_frame(frame)
        # if joint_position_from_frame:
        #     # Extract rules from detected pose
        #     arm_data = arm_rules.extract_arm_rules(joint_position_from_frame)
        #     leg_data = leg_rules.extract_leg_rules(joint_position_from_frame)
        #     torso_data = torso_rules.extract_torso_rules(joint_position_from_frame)

        #     # Prepare frame data for logging
        #     frame_pose_data = {
        #         'arms': arm_data,
        #         'legs': leg_data,
        #         'torso': torso_data
        #     }

        #     all_frames_landmarks.append({
        #         'frame': frame_count,
        #         'pose_landmarks': frame_pose_data,
        #         'detected_equipment': detected_equipment
        #     })

        #     # Visualize: Draw pose landmarks and equipment on the frame
        #     visualize_pose_landmarks(frame, joint_position_from_frame)
        #     visualize_equipment(frame, detected_equipment)

        #     # Display the frame with overlay
        #     cv2.imshow("Video with Pose and Equipment Detection", frame)

        #     # Optionally log extracted rules to console
        #     print(f"Frame {frame_count} - Detected Equipment: {detected_equipment}")
        #     print(f"Extracted Rules: {frame_pose_data}")

        # Update the equipment counter with detected equipment from this frame
        equipment_counter.update(detected_equipment)

        if cv2.waitKey(10) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

    # Optionally save the landmarks and equipment detection logs to a file
    if log_to_file:
        with open('pose_and_equipment_log.json', 'w') as f:
            json.dump(all_frames_landmarks, f, indent=4)

    return pose_dict, equipment_counter, {}

def visualize_pose_landmarks(frame, pose_landmarks):
    """ Draws pose landmarks on the frame. """
    for landmark, position in pose_landmarks.items():
        x = int(position['x'] * frame.shape[1])
        y = int(position['y'] * frame.shape[0])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

def visualize_equipment(frame, equipment):
    """ Display detected equipment names on the frame. """
    text = "Detected Equipment: " + ', '.join(equipment)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)


