import os
import pandas as pd
from typing import Dict, List, Tuple
from config import Config

class RAVDESSLoader:
    def __init__(self):
        self.dataset_path = Config.RAVDESS_PATH
        self.emotion_map = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }
        
    def parse_filename(self, filename: str) -> Dict:
        """Parse RAVDESS filename to extract metadata."""
        try:
            parts = filename.split('.')[0].split('-')
            
            if len(parts) != 7:
                raise ValueError(f"Invalid filename format: {filename}")
            
            emotion_code = parts[2]
            if emotion_code not in self.emotion_map:
                raise ValueError(f"Unknown emotion code: {emotion_code}")
            
            return {
                'modality': parts[0],  # 01=full-AV, 02=video-only, 03=audio-only
                'channel': parts[1],   # 01=speech, 02=song
                'emotion': self.emotion_map[emotion_code],
                'intensity': 'normal' if parts[3] == '01' else 'strong',
                'statement': parts[4], # 01="Kids...", 02="Dogs..."
                'repetition': parts[5],
                'actor': parts[6],
                'gender': 'female' if int(parts[6]) % 2 == 0 else 'male'
            }
        except Exception as e:
            raise ValueError(f"Error parsing filename {filename}: {str(e)}")
    
    def load_audio_data(self) -> List[Tuple[str, Dict]]:
        """Load audio-only files from RAVDESS dataset."""
        audio_files = []
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"RAVDESS dataset not found at {self.dataset_path}")
        
        # Iterate through actor folders
        for actor_folder in sorted(os.listdir(self.dataset_path)):
            if actor_folder.startswith('Actor_'):
                actor_path = os.path.join(self.dataset_path, actor_folder)
                
                # Get all audio files
                for file in os.listdir(actor_path):
                    if file.startswith('03-') and file.endswith('.wav'):  # Audio-only files
                        try:
                            file_path = os.path.join(actor_path, file)
                            metadata = self.parse_filename(file)
                            audio_files.append((file_path, metadata))
                        except ValueError as e:
                            print(f"Warning: Skipping file {file}: {str(e)}")
        
        if not audio_files:
            raise ValueError("No valid audio files found in RAVDESS dataset")
        
        return audio_files
    
    def load_video_data(self) -> List[Tuple[str, Dict]]:
        """Load video files from RAVDESS dataset."""
        video_files = []
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"RAVDESS dataset not found at {self.dataset_path}")
        
        # Iterate through actor folders
        for actor_folder in sorted(os.listdir(self.dataset_path)):
            if actor_folder.startswith('Actor_'):
                actor_path = os.path.join(self.dataset_path, actor_folder)
                
                # Get all video files
                for file in os.listdir(actor_path):
                    if file.startswith('01-') and file.endswith('.mp4'):  # Full AV files
                        try:
                            file_path = os.path.join(actor_path, file)
                            metadata = self.parse_filename(file)
                            video_files.append((file_path, metadata))
                        except ValueError as e:
                            print(f"Warning: Skipping file {file}: {str(e)}")
        
        if not video_files:
            raise ValueError("No valid video files found in RAVDESS dataset")
        
        return video_files

class CREMADLoader:
    def __init__(self):
        self.dataset_path = Config.CREMA_D_PATH
        self.emotion_map = {
            'ANG': 'ANG',  # Keep original codes for CREMA-D
            'DIS': 'DIS',
            'FEA': 'FEA',
            'HAP': 'HAP',
            'NEU': 'NEU',
            'SAD': 'SAD'
        }
        
        # Load demographics data
        demographics_path = os.path.join(self.dataset_path, 'VideoDemographics.csv')
        if not os.path.exists(demographics_path):
            raise FileNotFoundError(f"Demographics file not found at {demographics_path}")
            
        self.demographics = pd.read_csv(demographics_path)
        
    def parse_filename(self, filename: str) -> Dict:
        """Parse CREMA-D filename to extract metadata."""
        try:
            # Example: 1012_IEO_ANG_XX.wav
            parts = filename.split('.')[0].split('_')
            
            if len(parts) != 4:
                raise ValueError(f"Invalid filename format: {filename}")
            
            actor_id = parts[0]
            emotion_code = parts[2]
            
            if emotion_code not in self.emotion_map:
                raise ValueError(f"Unknown emotion code: {emotion_code}")
            
            # Get actor demographics
            actor_demo = self.demographics[
                self.demographics['ActorID'] == int(actor_id)
            ]
            
            if actor_demo.empty:
                raise ValueError(f"No demographics found for actor {actor_id}")
            
            actor_demo = actor_demo.iloc[0]
            
            return {
                'actor_id': actor_id,
                'sentence': parts[1],
                'emotion': self.emotion_map[emotion_code],
                'intensity': parts[3],
                'age': actor_demo['Age'],
                'sex': actor_demo['Sex'],
                'race': actor_demo['Race'],
                'ethnicity': actor_demo['Ethnicity']
            }
        except Exception as e:
            raise ValueError(f"Error parsing filename {filename}: {str(e)}")
    
    def load_audio_data(self) -> List[Tuple[str, Dict]]:
        """Load audio files from CREMA-D dataset."""
        audio_files = []
        audio_dir = os.path.join(self.dataset_path, 'AudioWAV')
        
        if not os.path.exists(audio_dir):
            raise FileNotFoundError(f"CREMA-D audio directory not found at {audio_dir}")
        
        for file in os.listdir(audio_dir):
            if file.endswith('.wav'):
                try:
                    file_path = os.path.join(audio_dir, file)
                    metadata = self.parse_filename(file)
                    audio_files.append((file_path, metadata))
                except ValueError as e:
                    print(f"Warning: Skipping file {file}: {str(e)}")
        
        if not audio_files:
            raise ValueError("No valid audio files found in CREMA-D dataset")
        
        return audio_files
    
    def load_video_data(self) -> List[Tuple[str, Dict]]:
        """Load video files from CREMA-D dataset."""
        video_files = []
        video_dir = os.path.join(self.dataset_path, 'VideoFlash')
        
        if not os.path.exists(video_dir):
            raise FileNotFoundError(f"CREMA-D video directory not found at {video_dir}")
        
        for file in os.listdir(video_dir):
            if file.endswith('.flv'):
                try:
                    file_path = os.path.join(video_dir, file)
                    metadata = self.parse_filename(file.replace('.flv', ''))
                    video_files.append((file_path, metadata))
                except ValueError as e:
                    print(f"Warning: Skipping file {file}: {str(e)}")
        
        if not video_files:
            raise ValueError("No valid video files found in CREMA-D dataset")
        
        return video_files