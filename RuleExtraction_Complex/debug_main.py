from exercise_analyzer import process_video_with_rules
from GYMDetector import YOLOv7EquipmentDetector, PoseDetector
from helper import get_video_path, calculate_rule_similarity
from constants import EQUIPMENTS
from exercise_rules import build_exercise_rules_json, get_exercise_names
import json
import os

def analyze_single_video(video_path: str, 
                        equipment_detector: YOLOv7EquipmentDetector,
                        pose_detector: PoseDetector,
                        exercise_rules: list):
    """
    Analyze a single video and print detailed debugging information.
    
    Args:
        video_path: Path to the video file
        equipment_detector: Initialized equipment detector
        pose_detector: Initialized pose detector
        exercise_rules: List of exercise rules from exercise_rules.py
    """
    # Get video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0].lower()
    print(f"\n{'='*80}\nAnalyzing video: {video_name}\n{'='*80}\n")
    
    # Process video and extract rules
    print("Extracting rules from video...")
    extracted_rules = process_video_with_rules(
        video_path, 
        equipment_detector, 
        pose_detector
    )
    
    # Print extracted rules
    print("\nExtracted Rules:")
    print(json.dumps(extracted_rules, indent=2))
    
    # Compare with each exercise in exercise_rules
    print("\nComparing with all known exercises:")
    similarities = []
    for exercise in exercise_rules:
        exercise_name = exercise['activity'].lower()
        similarity = calculate_rule_similarity(
            extracted_rules['body_landmarks'],
            exercise['body_landmarks']
        )
        similarities.append((exercise_name, similarity))
        print(f"\nSimilarity with {exercise_name}: {similarity:.3f}")
        
    # Sort and get top 3 matches
    top_matches = sorted(similarities, key=lambda x: x[1], reverse=True)[:3]
    sorted_similarities = sorted(similarities,key=lambda x: x[1], reverse=True)
    print("\nTop 3 Matches:")
    for i, (exercise, score) in enumerate(top_matches, 1):
        print(f"{i}. {exercise}: {score:.3f}")
    
    # Check if correct exercise was in top matches
    correct_rank = None
    for i, (exercise, _) in enumerate(sorted_similarities, 1):
        if exercise == video_name:
            correct_rank = i
            break
    
    if correct_rank:
        print(f"\nCorrect exercise ({video_name}) was ranked: {correct_rank}")
    else:
        print(f"\nWarning: Correct exercise ({video_name}) not found in exercise rules!")
        
    return extracted_rules, similarities

def main():
    # Initialize paths and constants
    video_root_dir = "../blender_mp4/"  # Adjust this path
    model_path = "./assets/best-v2.pt"
    
    # Load exercise rules
    exercise_rules = build_exercise_rules_json()
    exercise_names = get_exercise_names(exercise_rules)
    
    print("Available exercises in rules:")
    for name in exercise_names:
        print(f"- {name}")
    
    # Initialize detectors
    equipment_detector = YOLOv7EquipmentDetector(model_path, EQUIPMENTS)
    pose_detector = PoseDetector()
    
    # Get available videos that match exercise rules
    video_list, target_exercise_names = get_video_path(video_root_dir, exercise_names)
    
    if not video_list:
        print("\nNo matching videos found!")
        return
        
    print(f"\nFound {len(video_list)} matching videos:")
    for i, (video, exercise) in enumerate(zip(video_list, target_exercise_names), 1):
        print(f"{i}. {exercise} ({os.path.basename(video)})")
    
    while True:
        try:
            choice = input("\nEnter the number of the video to analyze (or 'q' to quit): ")
            if choice.lower() == 'q':
                break
                
            idx = int(choice) - 1
            if 0 <= idx < len(video_list):
                analyze_single_video(
                    video_list[idx],
                    equipment_detector,
                    pose_detector,
                    exercise_rules
                )
            else:
                print("Invalid video number!")
        except ValueError:
            print("Please enter a valid number or 'q'")
        except Exception as e:
            print(f"Error analyzing video: {str(e)}")

if __name__ == "__main__":
    main()