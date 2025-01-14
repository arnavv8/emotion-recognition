import torch
import numpy as np
import gc
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from .audio_processor import AudioProcessor
from .video_processor import VideoProcessor
from .dataset_loader import RAVDESSLoader, CREMADLoader
from torch.nn.utils.rnn import pad_sequence

class DataPreprocessor:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.video_processor = VideoProcessor()
        self.ravdess_loader = RAVDESSLoader()
        self.cremad_loader = CREMADLoader()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 8  # Reduced batch size to manage memory
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

    def process_audio_file(self, file_path: str, metadata: dict) -> Tuple[torch.Tensor, int]:
        """
        Processes an individual audio file.
        """
        try:
            if metadata['emotion'] == 'calm':
                return None, None
            
            audio_features = self.audio_processor.preprocess_audio(file_path)
            label = self.emotion_to_idx[metadata['emotion']]
            return audio_features.squeeze(0), label
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None, None

    def prepare_audio_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare audio dataset from both RAVDESS and CREMA-D."""
        ravdess_audio = self.ravdess_loader.load_audio_data()
        cremad_audio = self.cremad_loader.load_audio_data()
        total_files = len(ravdess_audio) + len(cremad_audio)

        print(f"Processing {total_files} audio files in parallel...")

        processed_files = 0
        features, labels = [], []

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {
                executor.submit(self.process_audio_file, f[0], f[1]): f[0] 
                for f in ravdess_audio + cremad_audio
            }

            for future in as_completed(future_to_file):
                result = future.result()
                if result and result[0] is not None:
                    features.append(result[0])
                    labels.append(result[1])
                processed_files += 1
                print(f"Processed {processed_files}/{total_files} audio files...", end="\r")

        print("\nAudio processing complete.")

        # Padding
        max_length = max([feat.shape[-1] for feat in features])
        features_padded = [torch.nn.functional.pad(feat, (0, max_length - feat.shape[-1])) for feat in features]

        features_padded = torch.stack(features_padded)
        labels = torch.tensor(labels, dtype=torch.long)

        self.cleanup_memory()
        return features_padded, labels

    def process_video_file(self, file_path: str, metadata: dict) -> Tuple[torch.Tensor, int]:
        """
        Processes an individual video file.
        """
        try:
            if metadata['emotion'] == 'calm':
                return None, None
            
            video_features = self.video_processor.extract_frames(file_path)
            label = self.emotion_to_idx[metadata['emotion']]
            return video_features.squeeze(0), label
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None, None

    def prepare_video_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare video dataset from both RAVDESS and CREMA-D."""
        ravdess_video = self.ravdess_loader.load_video_data()
        cremad_video = self.cremad_loader.load_video_data()
        total_files = len(ravdess_video) + len(cremad_video)

        print(f"Processing {total_files} video files in parallel...")

        processed_files = 0
        features, labels = [], []

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {
                executor.submit(self.process_video_file, f[0], f[1]): f[0] 
                for f in ravdess_video + cremad_video
            }

            for future in as_completed(future_to_file):
                result = future.result()
                if result and result[0] is not None:
                    features.append(result[0])
                    labels.append(result[1])
                processed_files += 1
                print(f"Processed {processed_files}/{total_files} video files...", end="\r")

        print("\nVideo processing complete.")

        # Ensure uniform shape using padding
        features_padded = pad_sequence(features, batch_first=True, padding_value=0)
        labels = torch.tensor(labels, dtype=torch.long)

        self.cleanup_memory()
        return features_padded, labels
