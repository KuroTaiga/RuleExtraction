# Rules genereation for specific excercises
from constants import *
from helper import calculate_distance, calculate_angle
def generate_exercise_rules(joint_positions:dict)->dict:
    """
    Generates rules specific to certain exercises based on joint positions.

    Args:
        joint_positions (dict): A dictionary containing the coordinates of various joints.

    Returns:
        dict: A dictionary containing lists of strings representing detected exercise-specific rules.
    """
    # Initialize the rules dictionary with empty lists
    exercise_rules = {key: [] for key in EXERCISE_KEYS}
    goblet_squat_rules = kettlebell_goblet_squat_rules(joint_positions)
    for key in goblet_squat_rules:
        if key in exercise_rules:
            exercise_rules[key].extend(goblet_squat_rules[key])
        else:
            exercise_rules[key] = goblet_squat_rules[key]

    return exercise_rules


def kettlebell_goblet_squat_rules(joint_positions):
    """
    Generates rules for the Kettlebell Goblet Squat exercise.

    Args:
        joint_positions (dict): A dictionary containing the coordinates of various joints.

    Returns:
        dict: A dictionary containing detected rules for the Kettlebell Goblet Squat exercise.
    """
    rules = {key: [] for key in RULE_KEYS}

    # Check for feet shoulder-width apart
    left_hip = joint_positions.get('left_hip')
    right_hip = joint_positions.get('right_hip')

    if left_hip and right_hip:
        hip_distance = abs(left_hip[0] - right_hip[0])
        if hip_distance >= SHOULDER_WIDTH_THRESHOLD:
            rules['legs'].append('shoulder_width_apart')

    # Check for bent knees (deep squat)
    left_knee_angle = None
    right_knee_angle = None

    left_hip = joint_positions.get('left_hip')
    left_knee = joint_positions.get('left_knee')
    left_ankle = joint_positions.get('left_ankle')

    right_hip = joint_positions.get('right_hip')
    right_knee = joint_positions.get('right_knee')
    right_ankle = joint_positions.get('right_ankle')

    if left_hip and left_knee and left_ankle:
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        if left_knee_angle < BENT:
            rules['left_leg_position'].append('bent')
            rules['left_leg_position'].append('knees_over_toes')

    if right_hip and right_knee and right_ankle:
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        if right_knee_angle < BENT:
            rules['right_leg_position'].append('bent')
            rules['right_leg_position'].append('knees_over_toes')

    # Combine leg positions
    if 'bent' in rules.get('left_leg_position', []) and 'bent' in rules.get('right_leg_position', []):
        rules['leg_position'].append('bent')
    
    if 'knees_over_toes' in rules.get('left_leg_position', []) and 'knees_over_toes' in rules.get('right_leg_position', []):
        rules['leg_position'].append('knees_over_toes')


    # Check for upright torso
    left_shoulder = joint_positions.get('left_shoulder')
    right_shoulder = joint_positions.get('right_shoulder')
    left_hip = joint_positions.get('left_hip')

    if left_shoulder and left_hip:
        torso_angle = abs(left_shoulder[0] - left_hip[0])
        if torso_angle <= UPRIGHT:
            rules['posture'].append('upright')

    # Check for kettlebell close to chest
    left_wrist = joint_positions.get('left_wrist')
    right_wrist = joint_positions.get('right_wrist')
    chest_x = (left_shoulder[0] + right_shoulder[0]) / 2

    if left_wrist and right_wrist:
        left_hand_distance = abs(left_wrist[0] - chest_x)
        right_hand_distance = abs(right_wrist[0] - chest_x)
        if left_hand_distance <= KETTLEBELL_DISTANCE_THRESHOLD and right_hand_distance <= KETTLEBELL_DISTANCE_THRESHOLD:
            rules['arm_position'].append('close_to_chest')

    # Add 'squat' to actions if knees are bent, the whole movement need to be added
    #TODO: is there a better way to address the movement?
    if 'bent' in rules.get('legs', []):
        rules['action'].append('squat', 'descent', 'ascent', 'maintain_position')

    return rules

def side_lunges_rules_generator(joint_positions):
    """
    Generates rules for the Side Lunges exercise.

    Args:
        joint_positions (dict): A dictionary containing the coordinates of various joints.

    Returns:
        dict: A dictionary containing detected rules for the Side Lunges exercise.
    """
    rules = {key: [] for key in RULE_KEYS}

    # Check for wide stance
    left_ankle = joint_positions.get('left_ankle')
    right_ankle = joint_positions.get('right_ankle')

    if left_ankle and right_ankle:
        ankle_distance = abs(left_ankle[0] - right_ankle[0])
        if ankle_distance >= WIDE_STANCE_THRESHOLD:
            rules['leg_position'].append('wide_stance')

    # Check for bent knees (low position)
    left_knee_angle = None
    right_knee_angle = None

    left_hip = joint_positions.get('left_hip')
    left_knee = joint_positions.get('left_knee')
    left_ankle = joint_positions.get('left_ankle')

    right_hip = joint_positions.get('right_hip')
    right_knee = joint_positions.get('right_knee')
    right_ankle = joint_positions.get('right_ankle')

    if left_hip and left_knee and left_ankle:
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        if left_knee_angle < BENT:
            rules['left_leg_position'].append('bent')

    if right_hip and right_knee and right_ankle:
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        if right_knee_angle < BENT:
            rules['right_leg_position'].append('bent')

    # Combine leg positions
    if 'bent' in rules.get('left_leg_position', []) and 'bent' in rules.get('right_leg_position', []):
        rules['leg_position'].append('bent')

    # Check for upright torso
    left_shoulder = joint_positions.get('left_shoulder')
    left_hip = joint_positions.get('left_hip')

    if left_shoulder and left_hip:
        torso_angle = abs(left_shoulder[0] - left_hip[0])
        if torso_angle <= UPRIGHT:
            rules['posture'].append('upright')
        else:
            rules['posture'].append('leaning')

    # Check for hands clasped at chest level
    left_wrist = joint_positions.get('left_wrist')
    right_wrist = joint_positions.get('right_wrist')
    left_shoulder = joint_positions.get('left_shoulder')
    right_shoulder = joint_positions.get('right_shoulder')

    if left_wrist and right_wrist and left_shoulder and right_shoulder:
        chest_y = (left_shoulder[1] + right_shoulder[1]) / 2
        left_hand_distance = abs(left_wrist[1] - chest_y)
        right_hand_distance = abs(right_wrist[1] - chest_y)
        if left_hand_distance <= CLOSE and right_hand_distance <= CLOSE:
            rules['arm_position'].append('close_to_chest')

    # Check for core engagement (simplified as upright posture)
    if 'upright' in rules.get('posture', []):
        rules['posture'].append('core_engaged')

    # Add actions based on movement
    rules['action'].extend(['ascend', 'descend', 'repeat_sequence'])

    return rules

def generate_exercise_rules(joint_positions):
    """
    Generates rules specific to certain exercises based on joint positions.

    Args:
        joint_positions (dict): A dictionary containing the coordinates of various joints.

    Returns:
        dict: A dictionary containing lists of strings representing detected exercise-specific rules.
    """
    # Initialize the rules dictionary with empty lists for each key
    exercise_rules = {key: [] for key in RULE_KEYS}

    # Call exercise-specific functions
    single_arm_arnold_press_rules = single_arm_arnold_press_rules_generator(joint_positions)
    for key in single_arm_arnold_press_rules:
        if key in exercise_rules:
            exercise_rules[key].extend(single_arm_arnold_press_rules[key])
        else:
            exercise_rules[key] = single_arm_arnold_press_rules[key]

    return exercise_rules

def single_arm_arnold_press_rules_generator(joint_positions):
    """
    Generates rules for the Single Arm Arnold Press exercise.

    Args:
        joint_positions (dict): A dictionary containing the coordinates of various joints.

    Returns:
        dict: A dictionary containing detected rules for the Single Arm Arnold Press exercise.
    """
    rules = {key: [] for key in RULE_KEYS}

    # Check for feet shoulder-width apart
    left_hip = joint_positions.get('left_hip')
    right_hip = joint_positions.get('right_hip')

    if left_hip and right_hip:
        hip_distance = abs(left_hip[0] - right_hip[0])
        if hip_distance >= SHOULDER_WIDTH_THRESHOLD:
            rules['leg_position'].append('shoulder_width_apart')

    # Check for upright posture
    left_shoulder = joint_positions.get('left_shoulder')
    left_hip = joint_positions.get('left_hip')

    if left_shoulder and left_hip:
        torso_angle = abs(left_shoulder[0] - left_hip[0])
        if torso_angle <= UPRIGHT:
            rules['posture'].append('upright')

    # Since it's a single arm exercise, we need to determine which arm is active
    active_arm = None
    if 'left_wrist' in joint_positions and 'left_elbow' in joint_positions and 'left_shoulder' in joint_positions:
        active_arm = 'left'
    elif 'right_wrist' in joint_positions and 'right_elbow' in joint_positions and 'right_shoulder' in joint_positions:
        active_arm = 'right'

    if active_arm:
        # Check for arm bent and positioned in front of torso
        shoulder = joint_positions.get(f'{active_arm}_shoulder')
        elbow = joint_positions.get(f'{active_arm}_elbow')
        wrist = joint_positions.get(f'{active_arm}_wrist')

        if shoulder and elbow and wrist:
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            if elbow_angle < BENT:
                rules[f'{active_arm}_arm_position'].append('bent')

            # Check if the elbow is in front of the torso (assuming x-axis represents left-right)
            if elbow[0] > shoulder[0] - ELBOW_FRONT_THRESHOLD and elbow[0] < shoulder[0] + ELBOW_FRONT_THRESHOLD:
                rules[f'{active_arm}_arm_position'].append('in_front_of_torso')

            # Check for kettlebell close to chest (distance between wrist and chest)
            chest_y = (joint_positions['left_shoulder'][1] + joint_positions['right_shoulder'][1]) / 2
            chest_x = (joint_positions['left_shoulder'][0] + joint_positions['right_shoulder'][0]) / 2
            wrist_distance = calculate_distance(wrist, (chest_x, chest_y))
            if wrist_distance <= CLOSE:
                rules[f'{active_arm}_arm_position'].append('close_to_chest')

            # Check for elbow pointing outward slightly (angle between shoulder, elbow, and horizontal plane)
            elbow_out_angle = abs(elbow[0] - shoulder[0])
            if elbow_out_angle > ELBOW_OUT_ANGLE_THRESHOLD:
                rules[f'{active_arm}_arm_position'].append('elbow_outward')

            # Add actions
            rules['action'].extend(['curl_upwards', 'controlled_motion', 'lowering_weight'])

        # Indicate that the other arm is inactive or neutral
        inactive_arm = 'left' if active_arm == 'right' else 'right'
        rules[f'{inactive_arm}_arm_position'].append('neutral')

    else:
        # If we cannot determine the active arm, add a general note
        rules['arm_position'].append('single_arm')

    return rules
###
    # Specific exercise rules for 30 different movements
    #

    
    # Abdominals Stretch
    # if abs(joint_positions['left_hip'][1] - joint_positions['left_ankle'][1]) < 50 and abs(joint_positions['left_shoulder'][1] - joint_positions['back'][1]) < 20:
    #     rules['posture'] = 'lying_flat'
    # if joint_positions['left_wrist'][1] < joint_positions['left_shoulder'][1] and joint_positions['right_wrist'][1] < joint_positions['right_shoulder'][1]:
    #     rules['arm_position'] = 'arms_overhead'

    # # Alternating Overhead Press
    # if joint_positions['left_wrist'][1] < joint_positions['left_shoulder'][1] or joint_positions['right_wrist'][1] < joint_positions['right_shoulder'][1]:
    #     rules['arm_position'] = 'one_arm_raised'

    # # Arnold Press
    # if joint_positions['left_wrist'][1] < joint_positions['left_shoulder'][1] and joint_positions['right_wrist'][1] < joint_positions['right_shoulder'][1]:
    #     rules['arm_position'] = 'both_arms_raised'

    # # Assisted Bulgarian Split Squat
    # if abs(joint_positions['left_knee'][0] - joint_positions['right_knee'][0]) > 50 and joint_positions['left_knee'][1] > joint_positions['left_hip'][1]:
    #     rules['legs'] = 'staggered_with_knee_bent'

    # # Ball Hamstring Curl
    # if abs(joint_positions['left_hip'][1] - joint_positions['left_shoulder'][1]) < 50 and joint_positions['left_foot'][1] > joint_positions['left_knee'][1]:
    #     rules['posture'] = 'lying_with_knees_bent'

    # # Band Bayesian Hammer Curl
    # if joint_positions['left_wrist'][1] > joint_positions['left_elbow'][1] and joint_positions['right_wrist'][1] > joint_positions['right_elbow'][1]:
    #     rules['arm_position'] = 'arms_bent'

    # # Band Front Raise
    # if joint_positions['left_wrist'][1] < joint_positions['left_shoulder'][1] and joint_positions['right_wrist'][1] < joint_positions['right_shoulder'][1]:
    #     rules['arm_position'] = 'arms_raised_to_shoulder'

    # # Band Glute Kickback
    # if abs(joint_positions['left_knee'][1] - joint_positions['right_knee'][1]) > 50 and joint_positions['left_foot'][1] < joint_positions['left_knee'][1]:
    #     rules['leg_position'] = 'leg_extended_back'

    # # Band Hammer Curl
    # if joint_positions['left_wrist'][1] > joint_positions['left_elbow'][1] and joint_positions['right_wrist'][1] > joint_positions['right_elbow'][1]:
    #     rules['arm_position'] = 'arms_bent'

    # # Band High Hammer Curl
    # if joint_positions['left_wrist'][1] < joint_positions['left_shoulder'][1] and joint_positions['right_wrist'][1] < joint_positions['right_shoulder'][1]:
    #     rules['arm_position'] = 'high_arms_bent'

    # # Band Lateral Raise
    # if joint_positions['left_wrist'][1] < joint_positions['left_shoulder'][1] and joint_positions['right_wrist'][1] < joint_positions['right_shoulder'][1]:
    #     rules['arm_position'] = 'arms_extended_to_sides'

    # # Band Skullcrusher
    # if joint_positions['left_wrist'][1] > joint_positions['left_shoulder'][1] and joint_positions['right_wrist'][1] > joint_positions['right_shoulder'][1]:
    #     rules['arm_position'] = 'arms_behind_head'

    # # Band Squat
    # if joint_positions['left_knee'][1] > joint_positions['left_hip'][1] and joint_positions['right_knee'][1] > joint_positions['right_hip'][1]:
    #     rules['legs'] = 'squat_position'

    # # Band Wood Chopper
    # if abs(joint_positions['left_wrist'][1] - joint_positions['right_wrist'][1]) > 50:
    #     rules['arm_position'] = 'diagonal_pull'

    # # Barbell Curtsy Lunge
    # if joint_positions['left_knee'][1] > joint_positions['left_hip'][1] and abs(joint_positions['left_knee'][0] - joint_positions['right_knee'][0]) > 50:
    #     rules['posture'] = 'curtsy_lunge_with_barbell'

    # # Barbell Bench Press
    # if abs(joint_positions['left_shoulder'][1] - joint_positions['back'][1]) < 20 and joint_positions['left_wrist'][1] < joint_positions['left_shoulder'][1]:
    #     rules['posture'] = 'lying_with_arms_raised'

    # # Barbell Bulgarian Split Squat
    # if abs(joint_positions['left_knee'][0] - joint_positions['right_knee'][0]) > 50 and joint_positions['left_knee'][1] > joint_positions['left_hip'][1]:
    #     rules['legs'] = 'staggered_with_knee_bent'

    # # Barbell Close Grip Bench Press
    # if joint_positions['left_wrist'][0] - joint_positions['right_wrist'][0] < 20 and joint_positions['left_wrist'][1] < joint_positions['left_shoulder'][1]:
    #     rules['posture'] = 'lying_with_close_grip'

    # # Barbell Front Raise
    # if joint_positions['left_wrist'][1] < joint_positions['left_shoulder'][1] and joint_positions['right_wrist'][1] < joint_positions['right_shoulder'][1]:
    #     rules['arm_position'] = 'barbell_front_raise'

    # # Barbell High Bar Squat
    # if joint_positions['left_knee'][1] > joint_positions['left_hip'][1] and joint_positions['left_hip'][1] < joint_positions['left_shoulder'][1]:
    #     rules['posture'] = 'squat_with_barbell_high_bar'

    # # Barbell Low Bar Squat
    # if joint_positions['left_knee'][1] > joint_positions['left_hip'][1] and joint_positions['left_hip'][1] < joint_positions['left_shoulder'][1]:
    #     rules['posture'] = 'squat_with_barbell_low_bar'

    # # Barbell Seated Calf Raise
    # if joint_positions['left_knee'][1] > joint_positions['left_hip'][1] and joint_positions['left_foot'][1] > joint_positions['left_knee'][1]:
    #     rules['posture'] = 'seated_calf_raise'

    # # Kettlebell Hip Thrust
    # if joint_positions['left_hip'][1] < joint_positions['left_shoulder'][1] and joint_positions['left_foot'][1] > joint_positions['left_knee'][1]:
    #     rules['posture'] = 'hip_thrust_with_kettlebell'

    # # Neutral Seated Overhead Press
    # if joint_positions['left_wrist'][1] < joint_positions['left_shoulder'][1] and joint_positions['right_wrist'][1] < joint_positions['right_shoulder'][1]:
    #     rules['arm_position'] = 'seated_overhead_press'

    # # Staggered Hip Thrust
    # if joint_positions['left_foot'][1] < joint_positions['right_foot'][1] and joint_positions['left_hip'][1] < joint_positions['left_shoulder'][1]:
    #     rules['posture'] = 'staggered_hip_thrust'

    # # Supermans
    # if abs(joint_positions['nose'][1] - joint_positions['back'][1]) < 20 and joint_positions['left_wrist'][1] < joint_positions['left_shoulder'][1] and joint_positions['left_foot'][1] < joint_positions['left_knee'][1]:
    #     rules['posture'] = 'supermans_pose'

    # # Windmill
    # if joint_positions['right_hand'][1] < joint_positions['right_shoulder'][1] and abs(joint_positions['left_foot'][0] - joint_positions['right_foot'][0]) > 50:
    #     rules['exercise'] = 'windmill_pose'
