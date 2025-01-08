import gradio as gr
import cv2
import numpy as np
import mediapipe as mp
import torch
import torchvision.transforms as transforms
from collections import OrderedDict
import logging
import sys
from PIL import Image

# Increase recursion limit
sys.setrecursionlimit(10000)

class VitPoseWrapper:
    """Wrapper class for VitPose model initialization and inference"""
    def __init__(self, weights_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        
        self.weights_path = weights_path
        self.model = self.load_model()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Define COCO keypoint connections
        self.keypoint_connections = [
            (15, 13), (13, 11), (16, 14), (14, 12),  # limbs
            (11, 12), (5, 11), (6, 12),  # hip
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # spine and neck
            (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)  # face and shoulders
        ]
        
    def load_model(self):
        """Load VitPose model"""
        try:
            from transformers import ViTPose
            
            model_cfg = {
                'backbone': {
                    'type': 'ViT',
                    'img_size': [256, 192],
                    'patch_size': 16,
                    'embed_dim': 768,
                    'depth': 12,
                    'num_heads': 12,
                    'ratio': 1,
                    'use_checkpoint': False,
                    'mlp_ratio': 4,
                    'qkv_bias': True,
                    'drop_path_rate': 0.3,
                }
            }
            
            model = ViTPose(model_cfg)
            
            if self.weights_path:
                logging.info(f"Loading weights from {self.weights_path}")
                state_dict = torch.load(self.weights_path, map_location=self.device)
                
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        k = k[7:]
                    new_state_dict[k] = v
                    
                model.load_state_dict(new_state_dict)
            
            model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            logging.error(f"Failed to load VitPose model: {e}")
            return None

    def preprocess_image(self, frame):
        """Preprocess image for VitPose"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (192, 256))
        frame_tensor = self.transform(frame_resized)
        frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
        return frame_tensor
        
    def postprocess_keypoints(self, heatmaps, original_shape):
        """Convert heatmaps to keypoint coordinates"""
        height, width = original_shape[:2]
        heatmap_height, heatmap_width = heatmaps.shape[2:]
        
        keypoints = []
        confidences = []
        
        for heatmap in heatmaps[0]:
            flat_id = torch.argmax(heatmap).item()
            y = flat_id // heatmap_width
            x = flat_id % heatmap_width
            
            confidence = heatmap[y, x].item()
            
            x_coord = int((x / heatmap_width) * width)
            y_coord = int((y / heatmap_height) * height)
            
            keypoints.append((x_coord, y_coord))
            confidences.append(confidence)
            
        return np.array(keypoints), np.array(confidences)

class PoseCompare:
    def __init__(self, vitpose_weights_path=None):
        self.vitpose_weights_path = vitpose_weights_path
        self.setup_models()
        
    def setup_models(self):
        """Initialize pose estimation models"""
        try:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mediapipe_pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                min_detection_confidence=0.5
            )
        except Exception as e:
            logging.error(f"Failed to initialize MediaPipe: {e}")
            self.mediapipe_pose = None
            
        self.vitpose = None
        # if self.vitpose_weights_path:
        #     try:
        #         self.vitpose = VitPoseWrapper(self.vitpose_weights_path)
        #         logging.info("VitPose model initialized successfully")
        #     except Exception as e:
        #         logging.error(f"Failed to initialize VitPose: {e}")

    def process_frame(self, frame, show_background):
        """Process a single frame with all models"""
        if frame is None:
            return None, None, None, None
            
        results = {
            'original': frame,
            'mediapipe': self.process_mediapipe(frame.copy(), show_background),
            'vitpose': self.process_vitpose(frame.copy(), show_background),
            'wham': self.create_blank_frame(frame) if not show_background else frame.copy(),
            '4dhuman': self.create_blank_frame(frame) if not show_background else frame.copy()
        }
        
        return (
            results['original'],
            results['mediapipe'],
            results['vitpose'],
            results['wham'],
            results['4dhuman']
        )

    def create_blank_frame(self, frame):
        """Create a blank frame with same dimensions"""
        return np.zeros(frame.shape, dtype=np.uint8)

    def process_mediapipe(self, frame, show_background):
        """Process frame with MediaPipe"""
        try:
            if not self.mediapipe_pose:
                return frame
                
            display_frame = frame if show_background else self.create_blank_frame(frame)
            results = self.mediapipe_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                color = (255, 255, 255) if not show_background else (0, 255, 0)
                self.mp_drawing.draw_landmarks(
                    display_frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=color,
                        thickness=2,
                        circle_radius=2
                    ),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=color,
                        thickness=2
                    )
                )
            
            return display_frame
        except Exception as e:
            logging.error(f"Error in MediaPipe processing: {e}")
            return frame

    def process_vitpose(self, frame, show_background):
        """Process frame with VitPose"""
        try:
            if not self.vitpose or not self.vitpose.model:
                return frame
                
            display_frame = frame if show_background else self.create_blank_frame(frame)
            
            with torch.no_grad():
                frame_tensor = self.vitpose.preprocess_image(frame)
                heatmaps = self.vitpose.model(frame_tensor)
            
            keypoints, confidences = self.vitpose.postprocess_keypoints(heatmaps, frame.shape)
            
            confidence_threshold = 0.3
            color = (255, 255, 255) if not show_background else (0, 255, 0)
            
            for connection in self.vitpose.keypoint_connections:
                if (confidences[connection[0]] > confidence_threshold and 
                    confidences[connection[1]] > confidence_threshold):
                    pt1 = tuple(map(int, keypoints[connection[0]]))
                    pt2 = tuple(map(int, keypoints[connection[1]]))
                    cv2.line(display_frame, pt1, pt2, color, 2)
            
            for i, (x, y) in enumerate(keypoints):
                if confidences[i] > confidence_threshold:
                    cv2.circle(display_frame, (int(x), int(y)), 4, color, -1)
            
            return display_frame
        except Exception as e:
            logging.error(f"Error in VitPose processing: {e}")
            return frame

def process_video(video_path, show_background):
    """Process video file and return frames"""
    try:
        pose_compare = PoseCompare(vitpose_weights_path="./ViTPose/weights/vitpose_large_coco_aic_mpii.pth")
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frames = pose_compare.process_frame(frame, show_background)
            frames.append(processed_frames)
            
        cap.release()
        return frames
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        return None

def create_gradio_interface():
    """Create Gradio interface"""
    with gr.Blocks() as demo:
        gr.Markdown("# Pose Estimation Comparison")
        
        with gr.Row():
            video_input = gr.Video()
            show_background = gr.Checkbox(label="Show Background", value=True)
        
        with gr.Row():
            original_output = gr.Video(label="Original")
            mediapipe_output = gr.Video(label="MediaPipe")
            # vitpose_output = gr.Video(label="VitPose")
            # wham_output = gr.Video(label="WHAM")
            # fourdhuman_output = gr.Video(label="4DHuman")
            
        process_btn = gr.Button("Process Video")
        
        process_btn.click(
            fn=process_video,
            inputs=[video_input, show_background],
            outputs=[original_output, mediapipe_output]
            # outputs=[original_output, mediapipe_output, vitpose_output, wham_output, fourdhuman_output]
        )
    
    return demo

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        demo = create_gradio_interface()
        demo.launch()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()