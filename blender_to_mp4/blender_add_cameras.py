#created by Xiang Zhang
#modified by Jiankun Dong
import bpy
import math
import time
import mathutils
import os
import sys

argv = sys.argv
try:
    # Find the custom output folder argument (after "--")
    output_folder_index = argv.index("--") + 1
    output_folder = argv[output_folder_index]
except (ValueError, IndexError):
    # Default to the .blend file's directory if no folder is provided
    output_folder = "Blenders_Different_Angles"

blend_file_path = bpy.data.filepath
blend_file_name = os.path.basename(blend_file_path)
blend_name = os.path.splitext(blend_file_name)[0]  
blend_name = blend_name.replace(" ", "_")
output_path = os.path.join(output_folder,blend_file_name)

# Ensure that existing cameras are deleted in the correct context
def delete_all_cameras():
    # Ensure all objects are deselected
    bpy.ops.object.select_all(action='DESELECT')
    # Select all cameras
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            obj.select_set(True)
    # Delete all selected objects
    bpy.ops.object.delete()

# Must ensure this operation runs in the correct context
if bpy.context.object and bpy.context.object.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')

delete_all_cameras()

# Find objects whose names contain 'Armature'
armature = None
for obj in bpy.data.objects:
    if 'Armature' in obj.name and obj.type == 'ARMATURE':
        armature = obj
        break

if armature is None:
    print("Error: 'Armature' object not found")
else:
    # Get the location of mixamorig:Spine2
    spine2_bone = armature.pose.bones.get('mixamorig:Spine2')
    if spine2_bone is None:
        print("Error: 'mixamorig:Spine2' bone not found")
    else:
        spine2_location = armature.matrix_world @ spine2_bone.head  # Global coordinates
        print(f"mixamorig:Spine2 location: {spine2_location}")

        # Arrange cameras at heights of 1.5m, 2m, and 2.5m on the Z-axis
        z_positions = [1.5, 2.0, 2.5]

        # Rotate cameras every 90 degrees, for a total of 360 degrees
        rotation_steps = 4
        rotation_step_degree = 90

        # Radius of 2.5m
        radii = [2.5]

        camera_counter = 1
        start_time = time.time()

        for z_pos in z_positions:
            for radius in radii:
                for step in range(rotation_steps):
                    creation_start_time = time.time()

                    # Calculate camera position on the circle
                    angle = math.radians(step * rotation_step_degree)
                    x = spine2_location.x + radius * math.cos(angle)
                    y = spine2_location.y + radius * math.sin(angle)

                    # Create camera
                    bpy.ops.object.select_all(action='DESELECT')
                    bpy.ops.object.camera_add(location=(x, y, z_pos))
                    camera = bpy.context.object
                    camera.name = f"{blend_name}_Camera_R{int(radius*100)}_Z{int(z_pos*100)}_A{step*rotation_step_degree}"
                    camera_counter += 1

                    # Set camera rotation to face mixamorig:Spine2
                    direction = mathutils.Vector((x, y, z_pos)) - spine2_location
                    rot_quat = direction.to_track_quat('Z', 'Y')
                    camera.rotation_euler = rot_quat.to_euler()

                    creation_end_time = time.time()
                    print(f"Camera {camera.name} created in {creation_end_time - creation_start_time:.4f} seconds")
                    print(f"Location: {camera.location}")
                    print(f"Rotation: {camera.rotation_euler}")

        end_time = time.time()
        total_time = end_time - start_time
        average_time_per_camera = total_time / (camera_counter - 1)

        print(f"Total {camera_counter - 1} cameras created in {total_time:.2f} seconds")
        print(f"Average time per camera: {average_time_per_camera:.4f} seconds")

        # Save the file
        bpy.ops.wm.save_as_mainfile(filepath=output_path)
