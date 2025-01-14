import torch
import numpy as np
from typing import List, Tuple, Dict
from .audio_processor import AudioProcessor
from .dataset_loader import RAVDESSLoader, CREMADLoader
from torch.nn.utils.rnn import pad_sequence
import gc

class DataPreprocessor:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.ravdess_loader = RAVDESSLoader()
        self.cremad_loader = CREMADLoader()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Unified emotion mapping for both datasets
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

    def prepare_audio_dataset(self, dataset: str = 'both') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare audio dataset with dataset selection.
        dataset: 'ravdess', 'cremad', or 'both'
        """
        if dataset not in ['ravdess', 'cremad', 'both']:
            raise ValueError("dataset must be 'ravdess', 'cremad', or 'both'")

        features = []
        labels = []
        processed_count = 0
        total_files = 0
        skipped_files = 0
        
        # Load RAVDESS audio if selected
        if dataset in ['ravdess', 'both']:
            ravdess_audio = self.ravdess_loader.load_audio_data()
            total_files += len(ravdess_audio)
            print("\nProcessing RAVDESS audio...")
            
            for file_path, metadata in ravdess_audio:
                try:
                    # Skip 'calm' emotion as it's not in our unified emotion set
                    if metadata['emotion'] == 'calm':
                        skipped_files += 1
                        continue
                        
                    # Process audio file
                    audio_features = self.audio_processor.preprocess_audio(file_path)
                    features.append(audio_features.squeeze(0))
                    
                    # Get emotion label
                    label = self.emotion_to_idx[metadata['emotion']]
                    labels.append(label)
                    
                    # Update progress
                    processed_count += 1
                    if processed_count % 10 == 0:  # Show progress every 10 files
                        print(f"\rProcessed {processed_count}/{total_files} files | Skipped: {skipped_files}", end="", flush=True)
                        self.cleanup_memory()
                        
                except Exception as e:
                    print(f"\nError processing {file_path}: {str(e)}")
                    skipped_files += 1

        # Load CREMA-D audio if selected
        if dataset in ['cremad', 'both']:
            cremad_audio = self.cremad_loader.load_audio_data()
            if dataset == 'cremad':
                total_files = len(cremad_audio)
                processed_count = 0
            else:
                total_files += len(cremad_audio)
            
            print("\nProcessing CREMA-D audio...")
            
            for file_path, metadata in cremad_audio:
                try:
                    # Map CREMA-D emotions to our unified set
                    if metadata['emotion'] not in self.emotion_to_idx:
                        skipped_files += 1
                        continue
                        
                    # Process audio file
                    audio_features = self.audio_processor.preprocess_audio(file_path)
                    features.append(audio_features.squeeze(0))
                    
                    # Get emotion label
                    label = self.emotion_to_idx[metadata['emotion']]
                    labels.append(label)
                    
                    # Update progress
                    processed_count += 1
                    if processed_count % 10 == 0:  # Show progress every 10 files
                        print(f"\rProcessed {processed_count}/{total_files} files | Skipped: {skipped_files}", end="", flush=True)
                        self.cleanup_memory()
                        
                except Exception as e:
                    print(f"\nError processing {file_path}: {str(e)}")
                    skipped_files += 1

        print(f"\nProcessing complete!")
        print(f"Total files processed: {processed_count}")
        print(f"Files skipped: {skipped_files}")
        print(f"Success rate: {(processed_count / (processed_count + skipped_files)) * 100:.2f}%")

        if not features:
            raise ValueError("No audio files were successfully processed")

        print("\nPreparing final dataset...")
        
        # Convert to tensors with padding
        try:
            # Find maximum length across all features
            max_length = max(feat.shape[-1] for feat in features)
            
            # Pad all features to max_length
            features_padded = []
            for i, feat in enumerate(features):
                padded = torch.nn.functional.pad(feat, (0, max_length - feat.shape[-1]))
                features_padded.append(padded)
                
                # Clean up memory periodically
                if i % 100 == 0:
                    self.cleanup_memory()
            
            # Stack features and convert labels to tensor
            features_tensor = torch.stack(features_padded)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            
            print(f"Dataset shape: {features_tensor.shape}")
            print(f"Labels shape: {labels_tensor.shape}")
            
            return features_tensor, labels_tensor
            
        except Exception as e:
            print(f"Error preparing final dataset: {str(e)}")
            raise

    def get_emotion_mapping(self) -> Dict[str, int]:
        """Return the emotion to index mapping."""
        return self.emotion_to_idx

    def get_emotion_labels(self) -> List[str]:
        """Return the list of emotion labels in order."""
        return [emotion for emotion, _ in sorted(self.emotion_to_idx.items(), key=lambda x: x[1])]