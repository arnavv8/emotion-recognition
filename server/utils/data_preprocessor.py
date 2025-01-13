import torch
import numpy as np
from typing import List, Tuple, Dict
from .audio_processor import AudioProcessor
from .video_processor import VideoProcessor
from .dataset_loader import RAVDESSLoader, CREMADLoader
from torch.nn.utils.rnn import pad_sequence
import gc

class DataPreprocessor:
    def __init__(self, process_video: bool = False):
        self.audio_processor = AudioProcessor()
        self.video_processor = VideoProcessor()
        self.ravdess_loader = RAVDESSLoader()
        self.cremad_loader = CREMADLoader()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 8  # Reduced batch size to manage memory
        self.process_video = process_video  # Flag to control video processing
        print(f"Using device: {self.device}")
        
        # Unified emotion mapping
        self.emotion_to_idx = {
            'angry': 0,
            'disgust': 1,
            'fearful': 2,
            'happy': 3,
            'neutral': 4,
            'sad': 5,
            'surprised': 6
        }
    
    def cleanup_memory(self):
        """Clean up GPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def prepare_audio_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare audio dataset from both RAVDESS and CREMA-D."""
        features = []
        labels = []
        
        # Load RAVDESS audio
        print("Processing RAVDESS audio...")
        ravdess_audio = self.ravdess_loader.load_audio_data()
        for i, (file_path, metadata) in enumerate(ravdess_audio):
            try:
                if metadata['emotion'] == 'calm':
                    continue
                    
                audio_features = self.audio_processor.preprocess_audio(file_path)
                features.append(audio_features)
                
                label = self.emotion_to_idx[metadata['emotion']]
                labels.append(label)
                
                # Clean up memory periodically
                if i % 100 == 0:
                    self.cleanup_memory()
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

        # Load CREMA-D audio
        print("Processing CREMA-D audio...")
        cremad_audio = self.cremad_loader.load_audio_data()
        for i, (file_path, metadata) in enumerate(cremad_audio):
            try:
                audio_features = self.audio_processor.preprocess_audio(file_path)
                features.append(audio_features)
                
                label = self.emotion_to_idx[metadata['emotion']]
                labels.append(label)
                
                # Clean up memory periodically
                if i % 100 == 0:
                    self.cleanup_memory()
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

        print("Processing audio features...")
        # Convert to tensors with padding
        features = [feat.squeeze(0) for feat in features]  # Remove extra dimensions
        max_length = max([feat.shape[-1] for feat in features])  # Get the max time step length

        # Pad all features to max_length
        features_padded = []
        for i, feat in enumerate(features):
            padded = torch.nn.functional.pad(feat, (0, max_length - feat.shape[-1]))
            features_padded.append(padded)
            
            # Clean up memory periodically
            if i % 100 == 0:
                self.cleanup_memory()
        
        features_padded = torch.stack(features_padded)  # Convert to a single tensor
        labels = torch.tensor(labels, dtype=torch.long)

        self.cleanup_memory()
        return features_padded, labels
    
    def prepare_video_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare video dataset from both RAVDESS and CREMA-D."""
        if not self.process_video:
            print("Skipping video processing.")
            return torch.tensor([]), torch.tensor([])

        features = []
        labels = []
        
        # Load RAVDESS video
        print("Processing RAVDESS video...")
        ravdess_video = self.ravdess_loader.load_video_data()
        for i, (file_path, metadata) in enumerate(ravdess_video):
            try:
                if metadata['emotion'] == 'calm':
                    continue
                    
                video_features = self.video_processor.extract_frames(file_path)
                features.append(video_features.squeeze(0))  # Remove batch dimension
                
                label = self.emotion_to_idx[metadata['emotion']]
                labels.append(label)
                
                # Clean up memory more frequently for video processing
                if i % 50 == 0:
                    self.cleanup_memory()
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        # Load CREMA-D video
        print("Processing CREMA-D video...")
        cremad_video = self.cremad_loader.load_video_data()
        for i, (file_path, metadata) in enumerate(cremad_video):
            try:
                video_features = self.video_processor.extract_frames(file_path)
                features.append(video_features.squeeze(0))  # Remove batch dimension
                
                label = self.emotion_to_idx[metadata['emotion']]
                labels.append(label)
                
                # Clean up memory more frequently for video processing
                if i % 50 == 0:
                    self.cleanup_memory()
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

        print("Processing video features...")
        # Ensure all feature tensors have the same shape using padding
        features_padded = pad_sequence(features, batch_first=True, padding_value=0)
        labels = torch.tensor(labels, dtype=torch.long)

        self.cleanup_memory()
        return features_padded, labels
