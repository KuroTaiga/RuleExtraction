import mediapipe as mp
import cv2
import json

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def extract_rules(landmarks):
    rules = {
        "arms": {"left": {}, "right": {}},
        "legs": {"left": {}, "right": {}},
        "torso": {}
    }

    # Arm positions and states
    def classify_arm(shoulder, elbow, wrist):
        # Calculate angles
        arm_angle = elbow[1] - shoulder[1]  # Y-difference
        arm_position = "horizontal" if abs(arm_angle) < 0.1 else ("up" if arm_angle < 0 else "down")
        elbow_angle = abs(wrist[0] - elbow[0])  # Difference in X-coordinates
        arm_state = "bent" if elbow_angle > 0.1 else "straight"
        return arm_position, arm_state

    # Left arm
    left_arm = classify_arm(
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST],
    )
    rules["arms"]["left"] = {"position": left_arm[0], "state": left_arm[1]}

    # Right arm
    right_arm = classify_arm(
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST],
    )
    rules["arms"]["right"] = {"position": right_arm[0], "state": right_arm[1]}

    # Leg states
    def classify_leg(hip, knee, ankle):
        # Calculate knee angle
        knee_angle = abs(ankle[0] - knee[0])
        leg_state = "bent" if knee_angle > 0.1 else "straight"
        return leg_state

    # Left leg
    left_leg = classify_leg(
        landmarks[mp_pose.PoseLandmark.LEFT_HIP],
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE],
    )
    rules["legs"]["left"] = {"state": left_leg}

    # Right leg
    right_leg = classify_leg(
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE],
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE],
    )
    rules["legs"]["right"] = {"state": right_leg}

    # Torso position
    def classify_torso(hip, shoulder):
        torso_angle = abs(hip[1] - shoulder[1])  # Y-difference
        if torso_angle > 0.5:
            return "standing"
        elif 0.1 < torso_angle <= 0.5:
            return "sitting"
        else:
            return "laying down"

    torso_position = classify_torso(
        landmarks[mp_pose.PoseLandmark.LEFT_HIP], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    )
    rules["torso"] = {"position": torso_position}

    return rules

# Video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe Pose
    results = pose.process(image)
    if results.pose_landmarks:
        landmarks = {
            id: (lm.x, lm.y, lm.z) for id, lm in enumerate(results.pose_landmarks.landmark)
        }

        # Extract rules
        rules = extract_rules(landmarks)

        # Output as JSON
        rules_json = json.dumps(rules, indent=4)
        print(rules_json)

        # Draw Pose landmarks on the image
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show the frame
    cv2.imshow("Pose Detection", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
