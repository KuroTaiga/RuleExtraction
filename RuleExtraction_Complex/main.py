from exercise_analyzer import process_video_with_rules
from GYMDetector import YOLOv7EquipmentDetector, PoseDetector
from helper import get_video_path, calculate_rule_similarity, calculate_similarity_with_details, calculate_equipment_similarity
from constants import EQUIPMENTS
from exercise_rules import build_exercise_rules_json, get_exercise_names
import pandas as pd
import json
import os

def detect_equipment_from_filename(filename: str) -> list:
    """Extract equipment type from filename."""
    filename = filename.lower()
    equipment_found = []
    for eqpt in EQUIPMENTS:
        if eqpt in filename:
            equipment_found.append(eqpt)  # Fixed to append actual equipment name
    return equipment_found if equipment_found else ['none']

def analyze_video(video_path: str, 
                 equipment_detector: YOLOv7EquipmentDetector,
                 pose_detector: PoseDetector,
                 exercise_rules: list,
                 expected_activity: str):
    """
    Analyze a single video using debug_main's working logic with added reporting.
    """
    # Get video name and equipment
    video_name = os.path.splitext(os.path.basename(video_path))[0].lower()
    detected_equipment = detect_equipment_from_filename(video_name)
    
    print(f"\n{'='*80}\nAnalyzing video: {video_name}\n{'='*80}\n")
    
    # Process video and extract rules
    print("Extracting rules from video...")
    extracted_rules = process_video_with_rules(
        video_path, 
        equipment_detector, 
        pose_detector
    )

    #todo: delete this after fixing equiptment detection:
    extracted_rules['equipment'] = {'type': detected_equipment}
    
    # Print extracted rules
    print("\nExtracted Rules:")
    print(json.dumps(extracted_rules, indent=2))
    
    # Compare with each exercise
    print("\nComparing with all known exercises:")
    similarities = []
    detailed_scores = {}
    
    for exercise in exercise_rules:
        exercise_name = exercise['activity'].lower()

        similarity_landmark = calculate_rule_similarity(
            extracted_rules['body_landmarks'],
            exercise['body_landmarks']
        )
        print(f"landmark: {similarity_landmark}")
        similarity_equipment = calculate_equipment_similarity(
            extracted_rules['equipment'],
            exercise['equipment']
        )
        print(f"equipment{similarity_equipment}")
        similarity = similarity_landmark+similarity_equipment
        similarities.append((exercise_name, similarity))
        print(f"\nSimilarity with {exercise_name}: {similarity:.3f}")
        
        # Store detailed scores
        detailed_scores[exercise_name] = {
            'similarity': similarity,
            'rules_matched': True  # Add more details if needed
        }
    
    # Sort and get top 3 matches
    matches = sorted(similarities, key=lambda x: x[1], reverse=True)[:3]
    print("\nTop 3 Matches:")
    for i, (exercise, score) in enumerate(matches, 1):
        print(f"{i}. {exercise}: {score:.3f}")
    
    return {
        'equipment': detected_equipment,
        'matches': matches,
        'extracted_rules': extracted_rules,
        'detailed_scores': detailed_scores
    }

def main():
    # Initialize paths
    video_root_dir = "../blender_mp4/"
    model_path = "./assets/best-v2.pt"
    output_dir = "./analysis_results2"
    os.makedirs(output_dir, exist_ok=True)

    # Load exercise rules
    exercise_rules = build_exercise_rules_json()
    exercise_names = get_exercise_names(exercise_rules)
    print(f"Looking for these exercises: {exercise_names}")

    # Initialize detectors
    equipment_detector = YOLOv7EquipmentDetector(model_path, EQUIPMENTS)
    pose_detector = PoseDetector()

    # Get video paths
    video_list, target_exercise_names = get_video_path(video_root_dir, exercise_names)
    if not video_list:
        print("No matching videos found!")
        return
        
    print(f"Found {len(video_list)} matching videos")

    # Prepare report data
    report_data = []
    correct_in_top3 = 0
    
    # Process each video
    for video_path, expected_activity in zip(video_list, target_exercise_names):
        print(f"\nProcessing video: {expected_activity}")
        
        try:
            # Analyze video
            results = analyze_video(
                video_path,
                equipment_detector,
                pose_detector,
                exercise_rules,
                expected_activity
            )
            
            # Check if expected activity is in top 3
            matched_exercises = [match[0].lower() for match in results['matches']]
            is_in_top3 = expected_activity.lower() in matched_exercises
            if is_in_top3:
                correct_in_top3 += 1
            
            # Prepare row for Excel report
            report_row = {
                'Activity': expected_activity,
                'Equipment': ', '.join(results['equipment']),
                'In Top 3': is_in_top3,
                'Rank': matched_exercises.index(expected_activity.lower()) + 1 if is_in_top3 else 'Not Found',
                'Top Match': results['matches'][0][0],
                'Top Match Score': f"{results['matches'][0][1]:.3f}",
                '2nd Match': results['matches'][1][0] if len(results['matches']) > 1 else '',
                '2nd Match Score': f"{results['matches'][1][1]:.3f}" if len(results['matches']) > 1 else '',
                '3rd Match': results['matches'][2][0] if len(results['matches']) > 2 else '',
                '3rd Match Score': f"{results['matches'][2][1]:.3f}" if len(results['matches']) > 2 else '',
                'Detailed Scores': json.dumps(results['detailed_scores'], indent=2)
            }
            
            report_data.append(report_row)
            
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create DataFrame
    df = pd.DataFrame(report_data)
    
    # Add summary row
    accuracy = correct_in_top3 / len(video_list) if video_list else 0
    summary_row = pd.DataFrame([{
        'Activity': 'SUMMARY',
        'Equipment': '',
        'In Top 3': f'{correct_in_top3}/{len(video_list)}',
        'Rank': f'{accuracy:.2%} accuracy',
        'Top Match': '',
        'Top Match Score': '',
        '2nd Match': '',
        '2nd Match Score': '',
        '3rd Match': '',
        '3rd Match Score': '',
        'Detailed Scores': ''
    }])
    
    df = pd.concat([df, summary_row], ignore_index=True)
    
    # Save reports
    try:
        # Save Excel report
        excel_path = os.path.join(output_dir, 'exercise_analysis_report.xlsx')
        df.to_excel(excel_path, index=False)
        print(f"Excel report saved to: {excel_path}")
    except Exception as e:
        print(f"Error saving Excel report: {e}")
        # Fallback to CSV
        csv_path = os.path.join(output_dir, 'exercise_analysis_report.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved report as CSV instead: {csv_path}")
    
    # Save detailed JSON
    json_path = os.path.join(output_dir, 'detailed_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\nAnalysis complete!")
    print(f"Detailed JSON results saved to: {json_path}")
    print(f"Overall accuracy: {accuracy:.2%} ({correct_in_top3}/{len(video_list)} correct in top 3)")

if __name__ == "__main__":
    main()