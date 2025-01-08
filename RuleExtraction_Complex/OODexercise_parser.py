# exercise_parser.py

import re
from constants import RULE_KEYS

def parse_exercise_description(description:str)->dict:
    # Initialize rules with keys from RULE_KEYS, each key corresponds to an empty list
    rules = {key: [] for key in RULE_KEYS}

    description = description.lower()

    # Posture parsing

    #TODO: there should be a faster match and check when it comes to checking matching strings
    posture_phrases = {
        'standing': ['standing'],
        'lying': ['lying flat', 'lying face down'],
        'upright': ['upright'],
        'seated': ['seated'],
        'hip_hinge': ['hinge at the hips', 'hip hinge'],
        'core_engaged': ['core engaged', 'engage the core'],
        'balance':['maintaining balance']
    }

    for posture, phrases in posture_phrases.items():
        for phrase in phrases:
            if phrase in description:
                rules['posture'].append(posture)
                break  # Avoid adding duplicate postures
    # TODO: Address each side of leg and arms
    # Leg positioning and actions
    leg_phrases = {
        'shoulder_width_apart': ['shoulder-width apart', 'feet shoulder width apart'],
        'bent': ['squat', 'lunge', 'curtsy'],
        'staggered': ['staggered', 'split squat'],
        'knees_over_toes':['knees align over the toes']
    }

    for leg_position, phrases in leg_phrases.items():
        for phrase in phrases:
            if phrase in description:
                rules['leg_position'].append(leg_position)
                break

    # Leg extension parsing
    leg_extension_phrases = {
        'extended': ['leg extended', 'kickback'],
        'calf_raise': ['calf raise']
    }

    for extension, phrases in leg_extension_phrases.items():
        for phrase in phrases:
            if phrase in description:
                rules['leg_extension'].append(extension)
                break

    # Arm positioning and actions
    #TODO: update close to chest phrases
    arm_phrases = {
        'spread': ['spread the arms', 'arms spread'],
        'extended': ['extend the arms', 'arms extended', 'outstretched arms'],
        'arms_bent': ['bend in the elbow', 'arms bent'],
        'arms_overhead': ['arms raised overhead', 'arms overhead'],
        'front_raise': ['front raise'],
        'lateral_raise': ['lateral raise'],
        'hammer_curl': ['hammer curl'],
        'skullcrusher': ['skullcrusher'],
        'press': ['press'],
        'wood_chopper': ['wood chopper'],
        'elbows_back': ['elbows back', 'driving the elbows back'],
        'close_to_chest':['close to the chest','weight at the chest','hands at chest level'],
        'elbow_bent_front': ['elbows bent and positioned in front of the torso'],
        'single_arm': ['single arm', 'one arm']
    }

    for arm_position, phrases in arm_phrases.items():
        for phrase in phrases:
            if phrase in description:
                rules['arm_position'].append(arm_position)
                break

    # Parsing complex actions
    #TODO: figure out how to do continued motions
    action_phrases = {
        'push': ['push', 'press'],
        'pull': ['pull'],
        'rotation': ['rotate', 'rotation', 'twist'],
        'descent':['descent','descend','lowering'],
        'ascent':['ascent','drive the body upward','rising'],
        'maintain_position':[''],
        'curl_upwards': ['curl the kettlebell upwards', 'curl upwards'],
        'controlled_motion': ['with a controlled motion', 'controlled motion'],
        'lowering_weight': ['lowering the weight back down', 'gradually lowering'],

    }

    for action, phrases in action_phrases.items():
        for phrase in phrases:
            if phrase in description:
                rules['action'].append(action)
                break

    # Parsing combination of movements
    combined_phrases = {
        'squat_press': ['squat while pressing', 'squat and press'],
        'lunge_raise': ['lunge while raising'],
        'rotation_press': ['twist and press', 'rotation and press']
    }

    for combined_movement, phrases in combined_phrases.items():
        for phrase in phrases:
            if phrase in description:
                rules['combined_movement'].append(combined_movement)
                break

    # Item-related exercise (kettlebell)
    if 'kettlebell' in description:
        if 'close to the chest' in description:
            rules['left_arm_position'].append('close_to_chest')
            rules['right_arm_position'].append('close_to_chest')

    # Side-specific
    if 'single arm' in description or 'one arm' in description:
        # Determine which arm, if specified
        if 'left' in description:
            rules['left_arm_position'].append('active')
        elif 'right' in description:
            rules['right_arm_position'].append('active')
        else:
            # If not specified, assume either arm could be active
            rules['arm_position'].append('single_arm')

    # Clean up empty lists in rules
    rules = {k: v for k, v in rules.items() if v}

    return rules
