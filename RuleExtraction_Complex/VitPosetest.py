import cv2
import torch
import numpy as np
from mmcv import Config
from mmpose.models import build_posenet
from mmcv.runner import load_checkpoint
import torchvision.transforms as transforms

# Load the model
# cfg = Config.fromfile('ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ochu/vitpose_base_coco_256x192.py')
# cfg = Config.fromfile('ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitPose+_small_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py')
cfg = Config.fromfile('ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitPose+_base_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py')
# cfg = Config.fromfile('ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitPose+_large_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py')

model = build_posenet(cfg.model)
# load_checkpoint(model, './ViTPose/weights/vitpose_large_coco_aic_mpii.pth', map_location='cpu')
# load_checkpoint(model, './ViTPose/weights/vitpose+_small.pth', map_location='cpu')
load_checkpoint(model,'./ViTPose/weights/vitpose+_base.pth',map_location='cpu')
# try:
#     load_checkpoint(model,'./ViTPose/weights/vitpose+_large.pth',map_location='cuda')
# except Exception as e:
#     load_checkpoint(model,'./ViTPose/weights/vitpose+_large.pth',map_location='cpu')
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Define COCO keypoint connections
COCO_CONNECTIONS = [
    (15, 13), (13, 11), (16, 14), (14, 12),  # Limbs
    (11, 12), (5, 11), (6, 12),  # Hip to shoulders
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # Neck and arms
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)  # Face and shoulders
]

# Function to preprocess a frame
def preprocess_frame(frame):
    original_height, original_width = frame.shape[:2]
    
    # Resize to model input dimensions (256x192)
    frame_resized = cv2.resize(frame, (192, 256))
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    # Apply transformation
    input_tensor = transform(frame_rgb).unsqueeze(0)  # Add batch dimension
    img_metas = [{
        'original_shape': (original_height, original_width, 3),
        'img_shape': (256, 192, 3),
        'center': np.array([96, 128]),  # Center of the image (in model coordinates)
        'scale': np.array([1.0, 1.0]),  # Scale (no scaling applied)
        'rotation': 0,
        'flip_pairs': None,
        'image_file': None,
        'dataset_idx':0
    }]

    return input_tensor, img_metas

# Function to draw the skeleton on a frame
def draw_skeleton(frame, keypoints, connections, confidence_threshold=0.3,original_shape = (256,192)):
    original_height, original_width = original_shape[:2]
    input_height, input_width = 256, 192  # Model input dimensions
    for i, (x, y, confidence) in enumerate(keypoints):
        if confidence > confidence_threshold:
            # Draw the keypoint
            x = int(x * original_width / input_width)
            y = int(y * original_height / input_height)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    
    for connection in connections:
        start_idx, end_idx = connection
        if (
            keypoints[start_idx][2] > confidence_threshold and
            keypoints[end_idx][2] > confidence_threshold
        ):
            # Draw the connection
            start_point = (
                int(keypoints[start_idx][0] * original_width / input_width),
                int(keypoints[start_idx][1] * original_height / input_height)
            )
            end_point = (
                int(keypoints[end_idx][0] * original_width / input_width),
                int(keypoints[end_idx][1] * original_height / input_height)
            )
            cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

# Open video file
# video_path = '../blender_mp4/Dumbbell Bench Press 2.mp4'
video_path = '../real_video/Alternating dumbbell forward lunge.mp4'
cap = cv2.VideoCapture(video_path)

# Process video frame-by-frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_tensor, img_metas = preprocess_frame(frame)

    # Run inference
    with torch.no_grad():
        output = model(img=input_tensor, img_metas=img_metas, return_loss=False)

    # Get keypoints (assuming single person for simplicity)
    preds = output['preds']  # Shape: (1, num_keypoints, 3)
    keypoints = preds[0]  # Extract the keypoints for the first person

    # Draw the skeleton
    draw_skeleton(frame, keypoints, COCO_CONNECTIONS,original_shape=frame.shape)

    # Display the frame
    cv2.imshow('Pose Estimation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
