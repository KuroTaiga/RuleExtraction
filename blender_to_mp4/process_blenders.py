import os
import subprocess
def run_setup_materials(input_folder,output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".blend"):
            input_path = os.path.join(input_folder, file_name)
            # output_file_name = f"{os.path.splitext(file_name)[0]}.mp4"
            # output_path = os.path.join(output_folder, output_file_name)
            command = [
                "blender",
                "-b",
                input_path,
                "-P", "setup_materials.py"
            ]
            print(f"Processing {file_name}...")
            try:
                # Run the Blender command
                subprocess.run(command, check=True)
                print(f"Successfully added material {file_name}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to add material {file_name}: {e}")

def run_add_cameras(input_folder,output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through all .blend files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".blend"):
            input_path = os.path.join(input_folder, file_name)
            # output_file_name = f"{os.path.splitext(file_name)[0]}.mp4"
            # output_path = os.path.join(output_folder, output_file_name)
            
            command = [
                "blender",
                "-b", input_path,
                "-P", "blender_add_cameras.py",
                "--", output_folder #change this if you don't want the default
            ]

            print(f"Processing {file_name}...")
            try:
                # Run the Blender command
                subprocess.run(command, check=True)
                print(f"Successfully added cameras {file_name}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to add cameras {file_name}: {e}")

def run_blender_to_mp4(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through all .blend files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".blend"):
            input_path = os.path.join(input_folder, file_name)
            # output_file_name = f"{os.path.splitext(file_name)[0]}.mp4"
            # output_path = os.path.join(output_folder, output_file_name)
            
            command = [
                "blender",
                "-b", input_path,
                "-P", "multi_camera_blender_to_video.py",
                "--", output_folder #change this if you don't want the default
            ]

            print(f"Processing {file_name}...")
            try:
                # Run the Blender command
                subprocess.run(command, check=True)
                print(f"Successfully converted {file_name}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to convert {file_name}: {e}")

# Input and Output Folder Paths
input_folder = "Blenders"  # Replace with your input folder path
blender_with_material = "Blender_With_Material"
multiple_camera_blender_folder = "Blenders_Different_Angles"
multicamera_and_color_folder = "Blenders_Different_Angles_w_Material"

video_output_folder = "MP4s/MultiCameras"  # Replace with your output folder path

# run_setup_materials(input_folder, blender_with_material)
# run_add_cameras(input_folder,multicamera_and_color_folder)
run_blender_to_mp4(multiple_camera_blender_folder, video_output_folder)
