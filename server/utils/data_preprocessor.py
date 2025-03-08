import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from utils.audio_processor import AudioProcessor
from utils.video_processor import VideoProcessor
from utils.dataset_loader import RAVDESSLoader, CREMADLoader
from torch.nn.utils.rnn import pad_sequence
import gc
import random
from sklearn.model_selection import train_test_split
from config import Config

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
        
        # Standardized emotion mapping for the API
        self.api_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        
        # Mapping from dataset-specific emotions to API emotions
        self.ravdess_to_api = {
            'neutral': 'neutral',
            'calm': 'neutral',  # Map calm to neutral for API
            'happy': 'happy',
            'sad': 'sad',
            'angry': 'angry',
            'fearful': 'fearful',
            'disgust': 'disgusted',
            'surprised': 'surprised'
        }
        
        self.cremad_to_api = {
            'ANG': 'angry',
            'DIS': 'disgusted',
            'FEA': 'fearful',
            'HAP': 'happy',
            'NEU': 'neutral',
            'SAD': 'sad'
        }
    
    def cleanup_memory(self):
        """Clean up GPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def prepare_audio_dataset(self, dataset: str) -> Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
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

        features_list = []
        labels = []
        processed_count = 0
        total_files = 0
        skipped_files = 0
        emotion_counts = {emotion: 0 for emotion in self.emotion_to_idx.keys()}
        
        # Load dataset-specific audio
        loader = self.ravdess_loader if dataset == 'ravdess' else self.cremad_loader
        
        try:
            audio_files = loader.load_audio_data()
            total_files = len(audio_files)
            print(f"\nProcessing {dataset.upper()} audio...")
            
            # Shuffle files to ensure random distribution
            random.shuffle(audio_files)
            
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
                    
                    # Ensure consistent dimensions [batch, channels, time]
                    if audio_features.dim() == 2:
                        audio_features = audio_features.unsqueeze(0)
                    elif audio_features.dim() == 4:
                        audio_features = audio_features.squeeze(0)
                    
                    features_list.append(audio_features)
                    labels.append(label)
                    
                    # Update progress
                    processed_count += 1
                    if processed_count % 100 == 0:
                        print(f"\rProcessed {processed_count}/{total_files} files | "
                              f"Skipped: {skipped_files}", end="", flush=True)
                        self.cleanup_memory()
                        
                except Exception as e:
                    print(f"\nError processing {file_path}: {str(e)}")
                    skipped_files += 1

            if not features_list:
                print("No audio files were successfully processed")
                return None

            print("\nPreparing final dataset...")
            
            try:
                # Stack features and ensure consistent dimensions
                features_tensor = torch.stack(features_list)
                labels_tensor = torch.tensor(labels, dtype=torch.long)
                
                # Verify data
                unique_labels = torch.unique(labels_tensor)
                print("\nUnique labels in dataset:", unique_labels.tolist())
                print("\nLabel counts:")
                for label in unique_labels:
                    count = (labels_tensor == label).sum().item()
                    emotion = [k for k, v in self.emotion_to_idx.items() if v == label.item()][0]
                    print(f"{emotion} (label {label.item()}): {count}")
                
                # Split data into train, validation, and test sets
                train_idx, temp_idx = train_test_split(
                    range(len(labels_tensor)),
                    test_size=0.3,
                    random_state=42,
                    stratify=labels_tensor.numpy()
                )
                
                val_idx, test_idx = train_test_split(
                    temp_idx,
                    test_size=0.5,
                    random_state=42,
                    stratify=labels_tensor[temp_idx].numpy()
                )
                
                # Create data splits
                train_data = {
                    'features': features_tensor[train_idx],
                    'labels': labels_tensor[train_idx]
                }
                
                val_data = {
                    'features': features_tensor[val_idx],
                    'labels': labels_tensor[val_idx]
                }
                
                test_data = {
                    'features': features_tensor[test_idx],
                    'labels': labels_tensor[test_idx]
                }
                
                # Calculate class weights for handling imbalanced data
                if Config.CLASS_WEIGHTS_ENABLED:
                    class_counts = torch.bincount(labels_tensor)
                    class_weights = 1.0 / class_counts.float()
                    class_weights = class_weights / class_weights.sum() * len(class_counts)
                    train_data['class_weights'] = class_weights
                
                return train_data, val_data, test_data
                
            except Exception as e:
                print(f"Error preparing final dataset: {str(e)}")
                return None
                
        except Exception as e:
            print(f"Error loading audio dataset: {str(e)}")
            return None

    def prepare_video_dataset(self, dataset: str) -> Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
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
        
        try:
            # Load CREMA-D video files
            video_files = self.cremad_loader.load_video_data()
            total_files = len(video_files)
            print("\nProcessing CREMA-D video...")
            
            # Shuffle files to ensure random distribution
            random.shuffle(video_files)
            
            for file_path, metadata in video_files:
                try:
                    # Process video file
                    video_features = self.video_processor.extract_frames(file_path, is_training=True)
                    
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
                print("No video files were successfully processed")
                return None

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
                features_std = features_tensor.std(dim=0, keepdim=True) + 1e-6  # Avoid division by zero
                features_tensor = (features_tensor - features_mean) / features_std
                
                # Apply data augmentation to balance classes
                if Config.AUGMENTATION_ENABLED:
                    augmented_features = []
                    augmented_labels = []
                    
                    # Find the maximum count across all emotions
                    max_count = max(emotion_counts.values())
                    
                    # Augment data for underrepresented classes
                    for label in unique_labels:
                        label_idx = (labels_tensor == label).nonzero(as_tuple=True)[0]
                        count = len(label_idx)
                        
                        # Skip if this class already has enough samples
                        if count >= max_count * 0.8:  # Allow some imbalance (80% of max)
                            continue
                        
                        # Calculate how many augmented samples we need
                        num_augment = int(max_count * 0.8) - count
                        
                        # Generate augmented samples
                        for _ in range(num_augment):
                            # Randomly select a sample from this class
                            sample_idx = label_idx[random.randint(0, count - 1)]
                            sample = features_tensor[sample_idx]
                            
                            # Apply strong augmentation (for video, this would be frame transformations)
                            augmented_sample = self.video_processor.apply_augmentation(sample)
                            
                            # Add to augmented data
                            augmented_features.append(augmented_sample)
                            augmented_labels.append(label.item())
                    
                    # Add augmented data to original data if we have any
                    if augmented_features:
                        augmented_features_tensor = torch.stack(augmented_features)
                        augmented_labels_tensor = torch.tensor(augmented_labels, dtype=torch.long)
                        
                        features_tensor = torch.cat([features_tensor, augmented_features_tensor], dim=0)
                        labels_tensor = torch.cat([labels_tensor, augmented_labels_tensor], dim=0)
                        
                        print("\nAfter augmentation:")
                        for label in torch.unique(labels_tensor):
                            count = (labels_tensor == label).sum().item()
                            emotion = [k for k, v in self.emotion_to_idx.items() if v == label.item()][0]
                            print(f"{emotion} (label {label.item()}): {count}")
                
                # Split data into train, validation, and test sets
                # Use stratified split to maintain class distribution
                train_idx, temp_idx = train_test_split(
                    range(len(labels_tensor)),
                    test_size=0.3,
                    random_state=42,
                    stratify=labels_tensor.numpy()
                )
                
                val_idx, test_idx = train_test_split(
                    temp_idx,
                    test_size=0.5,
                    random_state=42,
                    stratify=labels_tensor[temp_idx].numpy()
                )
                
                # Create data splits - keep on CPU initially to avoid CUDA pinned memory issues
                train_data = {
                    'features': features_tensor[train_idx],
                    'labels': labels_tensor[train_idx],
                    'mean': features_mean,
                    'std': features_std
                }
                
                val_data = {
                    'features': features_tensor[val_idx],
                    'labels': labels_tensor[val_idx],
                    'mean': features_mean,
                    'std': features_std
                }
                
                test_data = {
                    'features': features_tensor[test_idx],
                    'labels': labels_tensor[test_idx],
                    'mean': features_mean,
                    'std': features_std
                }
                
                # Print split sizes
                print(f"\nTrain set: {len(train_idx)} samples")
                print(f"Validation set: {len(val_idx)} samples")
                print(f"Test set: {len(test_idx)} samples")
                
                # Calculate class weights for handling imbalanced data
                if Config.CLASS_WEIGHTS_ENABLED:
                    class_counts = torch.bincount(labels_tensor)
                    class_weights = 1.0 / class_counts.float()
                    class_weights = class_weights / class_weights.sum() * len(class_counts)
                    train_data['class_weights'] = class_weights
                    
                    print("\nClass weights:")
                    for label, weight in enumerate(class_weights):
                        if label in unique_labels:
                            emotion = [k for k, v in self.emotion_to_idx.items() if v == label][0]
                            print(f"{emotion} (label {label}): {weight.item():.4f}")
                
                return train_data, val_data, test_data
                
            except Exception as e:
                print(f"Error preparing final dataset: {str(e)}")
                return None
                
        except Exception as e:
            print(f"Error loading video dataset: {str(e)}")
            return None

    def get_emotion_mapping(self) -> Dict[str, int]:
        """Return the current emotion to index mapping."""
        return self.emotion_to_idx

    def get_emotion_labels(self) -> List[str]:
        """Return the list of emotion labels in order."""
        return [emotion for emotion, _ in sorted(self.emotion_to_idx.items(), key=lambda x: x[1])]
        
    def map_to_api_emotion(self, dataset: str, emotion: str) -> str:
        """Map dataset-specific emotion to standardized API emotion."""
        if dataset == 'ravdess':
            return self.ravdess_to_api.get(emotion, 'neutral')
        elif dataset == 'cremad':
            return self.cremad_to_api.get(emotion, 'neutral')
        else:
            return 'neutral'  # Default fallback