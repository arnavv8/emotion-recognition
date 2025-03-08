import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import random
from typing import List
from config import Config

class VideoProcessor:
    def __init__(self):
        self.frame_size = Config.VIDEO_FRAME_SIZE
        self.frame_rate = Config.VIDEO_FRAME_RATE
        self.augmentation_enabled = Config.AUGMENTATION_ENABLED
        
        # Enhanced video preprocessing for training
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.frame_size),
            transforms.RandomHorizontalFlip(p=0.3),  # Data augmentation
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Lighting augmentation
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # Geometric augmentation
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Inference transform without augmentation
        self.inference_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.frame_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def apply_augmentation(self, frames_tensor):
        """Apply additional augmentations to video frames tensor."""
        if not self.augmentation_enabled or random.random() < 0.5:
            return frames_tensor
            
        # Get tensor dimensions
        C, T, H, W = frames_tensor.shape
        
        # Convert to numpy for OpenCV operations
        frames_np = frames_tensor.permute(1, 2, 3, 0).numpy()  # T, H, W, C
        augmented_frames = []
        
        for t in range(T):
            frame = frames_np[t]  # H, W, C
            
            # Random brightness adjustment
            if random.random() < 0.3:
                brightness = random.uniform(0.7, 1.3)
                frame = frame * brightness
                frame = np.clip(frame, 0, 1)
            
            # Random contrast adjustment
            if random.random() < 0.3:
                contrast = random.uniform(0.7, 1.3)
                mean = np.mean(frame, axis=(0, 1), keepdims=True)
                frame = (frame - mean) * contrast + mean
                frame = np.clip(frame, 0, 1)
            
            # Random rotation
            if random.random() < 0.3:
                angle = random.uniform(-15, 15)
                M = cv2.getRotationMatrix2D((W/2, H/2), angle, 1)
                frame = cv2.warpAffine(frame, M, (W, H))
            
            augmented_frames.append(frame)
        
        # Convert back to tensor
        augmented_frames = np.stack(augmented_frames)  # T, H, W, C
        augmented_tensor = torch.from_numpy(augmented_frames).permute(3, 0, 1, 2)  # C, T, H, W
        
        return augmented_tensor
    
    def extract_frames(self, video_path: str, max_frames: int = 32, is_training: bool = False) -> torch.Tensor:
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
                    
                    # Apply augmentation during training
                    if is_training and self.augmentation_enabled:
                        face = self.apply_augmentation_to_frame(face)
                    
                    # Convert BGR to RGB
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    
                    # Apply transforms
                    transform = self.train_transform if is_training else self.inference_transform
                    face_tensor = transform(face)
                    frames.append(face_tensor)
            
            frame_count += 1
        
        cap.release()
        
        # Handle cases where no faces were detected
        if len(frames) == 0:
            raise ValueError("No faces detected in video")
        
        # Pad if necessary
        while len(frames) < max_frames:
            # Duplicate the last frame instead of using zeros
            if frames:
                frames.append(frames[-1])
            else:
                # This should not happen given the check above, but just in case
                dummy_frame = torch.zeros(3, self.frame_size[0], self.frame_size[1])
                frames.append(dummy_frame)
        
        # Stack frames
        frames_tensor = torch.stack(frames)
        
        # Reshape to [C, T, H, W] for 3D CNN
        frames_tensor = frames_tensor.permute(1, 0, 2, 3)
        
        return frames_tensor.unsqueeze(0)  # Add batch dimension
    
    def apply_augmentation_to_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply augmentation to a single frame."""
        if not self.augmentation_enabled or random.random() < 0.5:
            return frame
            
        augmented = frame.copy()
        
        # Random brightness adjustment
        if random.random() < 0.3:
            brightness = random.uniform(0.7, 1.3)
            augmented = cv2.convertScaleAbs(augmented, alpha=brightness, beta=0)
        
        # Random contrast adjustment
        if random.random() < 0.3:
            contrast = random.uniform(0.7, 1.3)
            mean = np.mean(augmented)
            augmented = cv2.convertScaleAbs(augmented, alpha=contrast, beta=(1-contrast)*mean)
        
        # Random blur
        if random.random() < 0.2:
            blur_size = random.choice([3, 5])
            augmented = cv2.GaussianBlur(augmented, (blur_size, blur_size), 0)
            
        # Random noise
        if random.random() < 0.2:
            noise = np.random.normal(0, 10, augmented.shape).astype(np.uint8)
            augmented = cv2.add(augmented, noise)
            
        # Random rotation
        if random.random() < 0.2:
            angle = random.uniform(-15, 15)
            h, w = augmented.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            augmented = cv2.warpAffine(augmented, M, (w, h))
            
        return augmented
    
    def process_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Process a single frame for real-time prediction."""
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms (without augmentation for inference)
        frame = self.inference_transform(frame)
        
        # Add temporal and batch dimensions for model compatibility
        frame = frame.unsqueeze(1)  # Add temporal dimension [C, T, H, W]
        return frame.unsqueeze(0)  # Add batch dimension [B, C, T, H, W]