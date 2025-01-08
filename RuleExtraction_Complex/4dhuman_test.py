# Installation steps for 4D-Humans:
"""
1. Clone the repository:
git clone https://github.com/shubham-goel/4D-Humans.git
cd 4D-Humans

2. Create conda environment and install dependencies:
conda create -n 4dhumans python=3.8
conda activate 4dhumans

# Install PyTorch with CUDA (adjust cuda version as needed)
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch

# Install other dependencies
pip install -r requirements.txt

3. Download pre-trained models:
# Create a directory for pretrained models
mkdir pretrained_models
cd pretrained_models

# Download from the project's Google Drive:
# - SMPL model files (SMPL_MALE.pkl, SMPL_FEMALE.pkl, SMPL_NEUTRAL.pkl)
# - Pretrained model weights
# Place them in the pretrained_models directory

4. Install SMPL:
# Clone SMPL repository
git clone https://github.com/vchoutas/smplx.git
cd smplx
pip install -e .
"""

import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from utils.geometry import perspective_projection
from models.hmr import HMR
from utils.imutils import crop

class FourDHumansTest:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.setup_model()
        
    def setup_model(self):
        model = HMR(pretrained=True)
        checkpoint = torch.load('pretrained_models/checkpoint.pt', map_location=self.device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        return model.to(self.device)
        
    def process_frame(self, frame):
        # Prepare input
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect person and crop
        # Note: You might want to use a person detector here
        # For testing, we'll use the whole frame
        center = np.array([frame.shape[1]//2, frame.shape[0]//2])
        scale = min(frame.shape[0], frame.shape[1]) / 200
        
        # Process image
        img_processed = crop(img, center, scale, 
                           [constants.IMG_RES, constants.IMG_RES])
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_processed.transpose(2,0,1)).float()
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = self.model(img_tensor)
            
            # Convert rotation matrices to angles
            pred_pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3))
            pred_pose = pred_pose.reshape(1, -1)
            
            # Get SMPL vertices
            smpl_output = self.smpl(betas=pred_betas, 
                                  body_pose=pred_pose[:, 3:],
                                  global_orient=pred_pose[:, :3])
            vertices = smpl_output.vertices
            
            # Project 3D points to 2D
            pred_keypoints_3d = smpl_output.joints
            pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                     pred_camera)
            
        return pred_keypoints_2d, vertices

def test_4dhumans():
    # Initialize model
    model = FourDHumansTest()
    
    # Open video
    video_path = "path/to/your/test/video.mp4"
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        keypoints_2d, vertices = model.process_frame(frame)
        
        # Visualize results
        display_frame = frame.copy()
        
        # Draw skeleton
        for i, kp in enumerate(keypoints_2d[0]):
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(display_frame, (x, y), 3, (0, 255, 0), -1)
            
        # Draw connections (simplified skeleton)
        skeleton = [
            # Torso
            (2, 3), (3, 4),
            # Left arm
            (5, 6), (6, 7),
            # Right arm
            (8, 9), (9, 10),
            # Left leg
            (11, 12), (12, 13),
            # Right leg
            (14, 15), (15, 16)
        ]
        
        for connection in skeleton:
            pt1 = keypoints_2d[0][connection[0]]
            pt2 = keypoints_2d[0][connection[1]]
            cv2.line(display_frame,
                    (int(pt1[0]), int(pt1[1])),
                    (int(pt2[0]), int(pt2[1])),
                    (0, 255, 0), 2)
        
        cv2.imshow('4D-Humans Output', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_4dhumans()