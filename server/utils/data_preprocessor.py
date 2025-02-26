import torch
import numpy as np
from typing import List, Tuple, Dict
from .audio_processor import AudioProcessor
from .video_processor import VideoProcessor
from .dataset_loader import RAVDESSLoader, CREMADLoader
from torch.nn.utils.rnn import pad_sequence
import gc

class DataPreprocessor:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.video_processor = VideoProcessor()
        self.ravdess_loader = RAVDESSLoader()
        self.cremad_loader = CREMADLoader()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Dataset-specific emotion mappings
        self.ravdess_emotions = {
            'neutral': 0,
            'calm': 1,
            'happy': 2,
            'sad': 3,
            'angry': 4,
            'fearful': 5,
            'disgust': 6,
            'surprised': 7
        }
        
        self.cremad_emotions = {
            'ANG': 0,  # angry
            'DIS': 1,  # disgust
            'FEA': 2,  # fearful
            'HAP': 3,  # happy
            'NEU': 4,  # neutral
            'SAD': 5   # sad
        }
        
        # Current emotion mapping (will be set based on dataset)
        self.emotion_to_idx = {}
    
    def cleanup_memory(self):
        """Clean up GPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def prepare_audio_dataset(self, dataset: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare audio dataset with dataset-specific processing."""
        if dataset not in ['ravdess', 'cremad']:
            raise ValueError("dataset must be 'ravdess' or 'cremad'")
            
        # Set emotion mapping based on selected dataset
        self.emotion_to_idx = (
            self.ravdess_emotions if dataset == 'ravdess' 
            else self.cremad_emotions
        )
        
        print(f"\nUsing emotion mapping for {dataset.upper()}:")
        for emotion, idx in sorted(self.emotion_to_idx.items()):
            print(f"{emotion}: {idx}")

        features = []
        labels = []
        processed_count = 0
        total_files = 0
        skipped_files = 0
        emotion_counts = {emotion: 0 for emotion in self.emotion_to_idx.keys()}
        
        # Load dataset-specific audio
        loader = self.ravdess_loader if dataset == 'ravdess' else self.cremad_loader
        audio_files = loader.load_audio_data()
        total_files = len(audio_files)
        print(f"\nProcessing {dataset.upper()} audio...")
        
        for file_path, metadata in audio_files:
            try:
                # Process audio file
                audio_features = self.audio_processor.preprocess_audio(
                    file_path,
                    dataset_type=dataset
                )
                
                # Get emotion label
                emotion_key = metadata['emotion']
                if dataset == 'cremad':
                    emotion_key = metadata['emotion'][:3].upper()
                
                if emotion_key not in self.emotion_to_idx:
                    print(f"\nWarning: Unknown emotion {emotion_key} in {file_path}")
                    skipped_files += 1
                    continue
                
                label = self.emotion_to_idx[emotion_key]
                emotion_counts[emotion_key] += 1
                
                features.append(audio_features.squeeze(0))
                labels.append(label)
                
                # Update progress
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"\rProcessed {processed_count}/{total_files} files | "
                          f"Skipped: {skipped_files}", end="", flush=True)
                    print("\nCurrent emotion distribution:")
                    for emotion, count in emotion_counts.items():
                        print(f"{emotion}: {count}")
                    self.cleanup_memory()
                    
            except Exception as e:
                print(f"\nError processing {file_path}: {str(e)}")
                skipped_files += 1

        print(f"\nProcessing complete!")
        print(f"Total files processed: {processed_count}")
        print(f"Files skipped: {skipped_files}")
        print(f"Success rate: {(processed_count / (processed_count + skipped_files)) * 100:.2f}%")
        print("\nFinal emotion distribution:")
        for emotion, count in emotion_counts.items():
            print(f"{emotion}: {count}")

        if not features:
            raise ValueError("No audio files were successfully processed")

        print("\nPreparing final dataset...")
        
        try:
            # Convert to tensors
            features_tensor = torch.stack(features)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            
            # Verify data
            unique_labels = torch.unique(labels_tensor)
            print("\nUnique labels in dataset:", unique_labels.tolist())
            print("\nLabel counts:")
            for label in unique_labels:
                count = (labels_tensor == label).sum().item()
                emotion = [k for k, v in self.emotion_to_idx.items() if v == label.item()][0]
                print(f"{emotion} (label {label.item()}): {count}")
            
            # Normalize features
            features_mean = features_tensor.mean(dim=0, keepdim=True)
            features_std = features_tensor.std(dim=0, keepdim=True)
            features_tensor = (features_tensor - features_mean) / (features_std + 1e-6)
            
            print(f"\nDataset shape: {features_tensor.shape}")
            print(f"Labels shape: {labels_tensor.shape}")
            print(f"Number of emotion classes: {len(self.emotion_to_idx)}")
            
            # Verify tensor device placement
            features_tensor = features_tensor.to(self.device)
            labels_tensor = labels_tensor.to(self.device)
            print(f"Features device: {features_tensor.device}")
            print(f"Labels device: {labels_tensor.device}")
            
            return features_tensor, labels_tensor
            
        except Exception as e:
            print(f"Error preparing final dataset: {str(e)}")
            raise

    def prepare_video_dataset(self, dataset: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare video dataset (CREMA-D only)."""
        if dataset != 'cremad':
            raise ValueError("Video dataset preparation is only supported for CREMA-D")
        
        self.emotion_to_idx = self.cremad_emotions
        print("\nUsing CREMA-D emotion mapping:")
        for emotion, idx in sorted(self.emotion_to_idx.items()):
            print(f"{emotion}: {idx}")
        
        features = []
        labels = []
        processed_count = 0
        total_files = 0
        skipped_files = 0
        emotion_counts = {emotion: 0 for emotion in self.emotion_to_idx.keys()}
        
        # Load CREMA-D video files
        video_files = self.cremad_loader.load_video_data()
        total_files = len(video_files)
        print("\nProcessing CREMA-D video...")
        
        for file_path, metadata in video_files:
            try:
                # Process video file
                video_features = self.video_processor.extract_frames(file_path)
                
                # Get emotion label
                emotion_key = metadata['emotion'][:3].upper()
                if emotion_key not in self.emotion_to_idx:
                    print(f"\nWarning: Unknown emotion {emotion_key} in {file_path}")
                    skipped_files += 1
                    continue
                
                label = self.emotion_to_idx[emotion_key]
                emotion_counts[emotion_key] += 1
                
                features.append(video_features.squeeze(0))
                labels.append(label)
                
                # Update progress
                processed_count += 1
                if processed_count % 50 == 0:
                    print(f"\rProcessed {processed_count}/{total_files} files | "
                          f"Skipped: {skipped_files}", end="", flush=True)
                    print("\nCurrent emotion distribution:")
                    for emotion, count in emotion_counts.items():
                        print(f"{emotion}: {count}")
                    self.cleanup_memory()
                    
            except Exception as e:
                print(f"\nError processing {file_path}: {str(e)}")
                skipped_files += 1
        
        print(f"\nProcessing complete!")
        print(f"Total files processed: {processed_count}")
        print(f"Files skipped: {skipped_files}")
        print(f"Success rate: {(processed_count / (processed_count + skipped_files)) * 100:.2f}%")
        print("\nFinal emotion distribution:")
        for emotion, count in emotion_counts.items():
            print(f"{emotion}: {count}")

        if not features:
            raise ValueError("No video files were successfully processed")

        print("\nPreparing final dataset...")
        
        try:
            # Convert to tensors
            features_tensor = torch.stack(features)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            
            # Verify data
            unique_labels = torch.unique(labels_tensor)
            print("\nUnique labels in dataset:", unique_labels.tolist())
            print("\nLabel counts:")
            for label in unique_labels:
                count = (labels_tensor == label).sum().item()
                emotion = [k for k, v in self.emotion_to_idx.items() if v == label.item()][0]
                print(f"{emotion} (label {label.item()}): {count}")
            
            # Normalize features
            features_mean = features_tensor.mean(dim=0, keepdim=True)
            features_std = features_tensor.std(dim=0, keepdim=True)
            features_tensor = (features_tensor - features_mean) / (features_std + 1e-6)
            
            print(f"\nDataset shape: {features_tensor.shape}")
            print(f"Labels shape: {labels_tensor.shape}")
            print(f"Number of emotion classes: {len(self.emotion_to_idx)}")
            
            # Verify tensor device placement
            features_tensor = features_tensor.to(self.device)
            labels_tensor = labels_tensor.to(self.device)
            print(f"Features device: {features_tensor.device}")
            print(f"Labels device: {labels_tensor.device}")
            
            return features_tensor, labels_tensor
            
        except Exception as e:
            print(f"Error preparing final dataset: {str(e)}")
            raise

    def get_emotion_mapping(self) -> Dict[str, int]:
        """Return the current emotion to index mapping."""
        return self.emotion_to_idx

    def get_emotion_labels(self) -> List[str]:
        """Return the list of emotion labels in order."""
        return [emotion for emotion, _ in sorted(self.emotion_to_idx.items(), key=lambda x: x[1])]