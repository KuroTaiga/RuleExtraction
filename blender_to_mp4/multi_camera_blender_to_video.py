
#created by Xiang Zhang
#modified by Jiankun Dong
import bpy
import os
import json
import sys

# 设置输出路径和文件名格式
argv = sys.argv
try:
    # Find the custom output folder argument (after "--")
    output_folder_index = argv.index("--") + 1
    output_folder = argv[output_folder_index]
except (ValueError, IndexError):
    # Default to the .blend file's directory if no folder is provided
    output_folder = "MP4s/MultiCameras"

blend_file_path = bpy.data.filepath
blend_name = os.path.splitext(os.path.basename(blend_file_path))[0]
blend_name = blend_name.replace(" ", "_")
output_path = os.path.join(output_folder,f'{blend_name}.mp4')

processed_cameras_file = os.path.join(output_folder, "processed_cameras.json")
video_output_format = "video_{camera_name}.mp4"

# 获取所有摄像机对象，按名称排序
cameras = sorted([obj for obj in bpy.data.objects if obj.type == 'CAMERA'], key=lambda cam: cam.name)

# 设置渲染分辨率和帧范围
bpy.context.scene.render.resolution_x = 1280
bpy.context.scene.render.resolution_y = 960
bpy.context.scene.cycles.samples = 50
bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
bpy.context.scene.render.ffmpeg.format = 'MPEG4'
bpy.context.scene.render.ffmpeg.codec = 'H264'
bpy.context.scene.render.ffmpeg.constant_rate_factor = 'PERC_LOSSLESS'
bpy.context.scene.render.ffmpeg.ffmpeg_preset = 'REALTIME'
bpy.context.scene.render.ffmpeg.gopsize = 12
bpy.context.scene.render.ffmpeg.max_b_frames = 2

start_frame = bpy.context.scene.frame_start
end_frame = bpy.context.scene.frame_end
frame_step = 10

# 确保世界背景颜色的节点设置正确
bpy.context.scene.world.use_nodes = True
nodes = bpy.context.scene.world.node_tree.nodes
links = bpy.context.scene.world.node_tree.links

background_node = nodes.get('Background')
if background_node:
    # 清除颜色链接
    while background_node.inputs['Color'].links:
        link = background_node.inputs['Color'].links[0]
        links.remove(link)
else:
    # 如果没有Background节点，则创建一个
    background_node = nodes.new(type='ShaderNodeBackground')
    links.new(background_node.outputs['Background'], nodes.get('World Output').inputs['Surface'])

# 创建输出路径
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 加载已处理的摄像机列表
if os.path.exists(processed_cameras_file):
    with open(processed_cameras_file, 'r') as file:
        processed_cameras = json.load(file)
else:
    processed_cameras = []

# 渲染每个摄像机的动画
for camera in cameras:
    video_output_file = os.path.join(output_path, video_output_format.format(camera_name=camera.name))
    
    if camera.name in processed_cameras or os.path.exists(video_output_file):
        print(f"Skipping {camera.name}, already processed or output exists: {video_output_file}")
        continue
    
    # 设置摄像机的焦距
    camera.data.lens = 10
    
    # 设置当前摄像机
    bpy.context.scene.camera = camera
    
    # 设置输出文件路径
    bpy.context.scene.render.filepath = video_output_file
    
    # 设置帧步长
    bpy.context.scene.frame_step = frame_step
    
    # 渲染动画
    bpy.ops.render.render(animation=True)
    
    # 将已处理的摄像机记录到文件
    processed_cameras.append(camera.name)
    with open(processed_cameras_file, 'w') as file:
        json.dump(processed_cameras, file)

print("All cameras processed.")