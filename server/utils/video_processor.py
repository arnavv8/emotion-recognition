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
        
        # Enhanced video preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.frame_size),
            transforms.RandomHorizontalFlip(p=0.3),  # Data augmentation
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Lighting augmentation
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_frames(self, video_path: str, max_frames: int = 32) -> torch.Tensor:
        """Extract and process video frames with enhanced features."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        # Calculate frame sampling rate
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError("No frames found in video")
            
        sample_rate = max(1, total_frames // max_frames)
        frame_count = 0
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        while len(frames) < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Only process every nth frame
            if frame_count % sample_rate == 0:
                # Detect face
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    # Get the largest face
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    
                    # Add padding around face
                    padding = int(min(w, h) * 0.2)
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(frame.shape[1] - x, w + 2 * padding)
                    h = min(frame.shape[0] - y, h + 2 * padding)
                    
                    # Extract face region
                    face = frame[y:y+h, x:x+w]
                    
                    # Convert BGR to RGB
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    
                    # Apply transforms
                    face_tensor = self.transform(face)
                    frames.append(face_tensor)
            
            frame_count += 1
        
        cap.release()
        
        # Handle cases where no faces were detected
        if len(frames) == 0:
            raise ValueError("No faces detected in video")
        
        # Pad if necessary
        while len(frames) < max_frames:
            frames.append(torch.zeros_like(frames[0]))
        
        # Stack frames
        frames = torch.stack(frames)
        
        return frames.unsqueeze(0)  # Add batch dimension
    
    def process_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Process a single frame for real-time prediction."""
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms (without augmentation for inference)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.frame_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        frame = transform(frame)
        return frame.unsqueeze(0)  # Add batch dimension