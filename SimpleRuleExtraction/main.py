from helper import *
from GYMDetector import YOLOv7EquipmentDetector, PoseDetector
from constants import EQUIPMENTS

def detect_equipment_from_filename(filename: str) -> list:
    """Extract equipment type from filename."""
    filename = filename.lower()
    equipment_found = []
    for eqpt in EQUIPMENTS:
        if eqpt in filename:
            equipment_found.append(eqpt)  # Fixed to append actual equipment name
    return equipment_found if equipment_found else ['none']

def extract_videos(folder_path):
    """Extract video files from a folder."""
    video_files = []
    activity_names = []
    for file in os.listdir(folder_path):
        if file.endswith('.mp4'):
            curr_activity = file.split('.')[0]
            video_files.append(file)
            activity_names.append(curr_activity)
    return video_files,activity_names

def main():
    video_source = 'test/videos'
    model_path = 'assests/best-v2.pt'
    output_path = 'test/results'
    os.makedirs(output_path, exist_ok=True)
    video_files,activity_names = extract_videos(video_source)

    # print(activity_names)

    equiptment_detector = YOLOv7EquipmentDetector(model_path)
    
main()
    
