# constants.py
MATCH_THRESHOLD = 0.5
# constants.py


# Define the keys for the rules dictionary
RULE_KEYS = [
    'left_foot',
    'right_foot',
    'left_hip',
    'right_hip',
    'torso',
    'left_hand',
    'right_hand',
    'left_elbow',
    'right_elbow',
    'left_shoulder',
    'right_shoulder',
    'left_knee',
    'right_knee'
]
MIRRORED = ['true','false']
EQUIPMENTS = ['dumbbell','barbell','kettlebell','ball']
INCLINE = ['0','15','45','60','90','none']
# List of joint keys used in the system
JOINT_KEYS = [
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
# List of keys from arm and leg rules
ARM_KEYS = ['left_hand', 'right_hand', 'left_elbow','right_elbow']
LEG_KEYS = ['left_foot', 'right_foot', 'left_knee','right_knee']
BODY_KEYS = ['torso','left_hip','right_hip','left_shoulder','right_shoulder']
# List of keys from specific exercise rules

# Threshold constants for position detection (e.g., distances in pixels or units used)
CLOSE = 10.0          # Threshold to determine if two points are 'close' to each other
SPREAD = 50.0         # Threshold to determine if arms are 'spread' apart
OUTWARD = 15.0
INWARD = 5.0
FLEXED = 90.0
BENT_RANGE = 5.0
SHOULDER_RANGE = 1.1
# Constants for angle thresholds (in degrees)
STRAIGHT = 8.0                        # Degrees; threshold to consider a limb 'straight'
HIP_HINGE_ANGLE_THRESHOLD = 30.0      # Degrees; threshold for detecting 'hip_hinge' posture

# Constants for posture detection (distance thresholds)
STANDING = 20.0                       # Vertical distance threshold for 'standing' posture
UPRIGHT = 15.0                        # Vertical distance threshold for 'upright' posture
NEUTRAL_SPINE = 5.0
LEAN = 10.0
BENT = 40.0                           # Vertical distance threshold for 'bent' posture

# Constant for core engagement detection (distance threshold)
CORE_ENGAGEMENT_THRESHOLD = 50.0      # Threshold for detecting 'core_engaged' posture

# Thresholds for Kettlebell Goblet Squat
SHOULDER_WIDTH_THRESHOLD = 50.0               # Adjust based on your coordinate system
KETTLEBELL_DISTANCE_THRESHOLD = 20.0          # Threshold for hands close to chest
# Thresholds for Side Lunges
WIDE_STANCE_THRESHOLD = 80.0                 # Adjust based on your coordinate system

# Thresholds for Single Arm Arnold Press
ARM_EXTENSION_ANGLE_THRESHOLD = 160           # Degrees; angle at elbow for extended arm
ELBOW_OUT_ANGLE_THRESHOLD = 10                # Threshold for elbow pointing outward
ELBOW_FRONT_THRESHOLD = 15                    # Threshold for elbow in front of torso
# Simulate getting joint positions from a video feed

# def get_joint_positions_from_video() -> dict:
#     # joint_positions = {
#     #     'left_shoulder': (80.0, 200.0),
#     #     'right_shoulder': (120.0, 200.0),
#     #     'left_elbow': (70.0, 250.0),
#     #     'right_elbow': (130.0, 250.0),
#     #     'left_wrist': (60.0, 300.0),
#     #     'right_wrist': (140.0, 300.0),
#     #     'left_hip': (90.0, 270.0),
#     #     'right_hip': (110.0, 270.0),
#     #     'left_knee': (85.0, 350.0),
#     #     'right_knee': (115.0, 350.0),
#     #     'left_ankle': (80.0, 400.0),
#     #     'right_ankle': (120.0, 400.0)
#     # }
#     return joint_positions