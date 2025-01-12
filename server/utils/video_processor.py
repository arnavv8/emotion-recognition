import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from typing import List
from config import Config

class VideoProcessor:
    def __init__(self):
        self.frame_size = Config.VIDEO_FRAME_SIZE
        self.frame_rate = Config.VIDEO_FRAME_RATE
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.frame_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_frames(self, video_path: str, max_frames: int = 32) -> torch.Tensor:
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        while len(frames) < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            frame = self.transform(frame)
            frames.append(frame)
        
        cap.release()
        
        # Pad if necessary
        while len(frames) < max_frames:
            frames.append(torch.zeros_like(frames[0]))
        
        # Stack frames
        frames = torch.stack(frames)
        
        return frames.unsqueeze(0)  # Add batch dimension
    
    def process_frame(self, frame: np.ndarray) -> torch.Tensor:
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        frame = self.transform(frame)
        
        return frame.unsqueeze(0)  # Add batch dimension