from helper import *
from GYMDetector import YOLOv7EquipmentDetector, PoseDetector

def detect_equipment_from_filename(filename: str) -> list:
    """Extract equipment type from filename."""
    filename = filename.lower()
    equipment_found = []
    for eqpt in EQUIPMENTS:
        if eqpt in filename:
            equipment_found.append(eqpt)  # Fixed to append actual equipment name
    return equipment_found if equipment_found else ['none']


def main():
    video_source = 'test/videos'
    model_path = 'assests/best.pt'
    output_path = 'test/results'
    os.makedirs(output_path, exist_ok=True)

    exercise_rules = load_exercise_rules('test/exercise_rules.json')
    exercise_names = [exercise['activity'] for exercise in exercise_rules]

    equipment_detector = YOLOv7EquipmentDetector(model_path)
    pose_detector = PoseDetector()

    
