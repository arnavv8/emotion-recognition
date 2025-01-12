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
        parts = filename.split('.')[0].split('-')
        
        return {
            'modality': parts[0],  # 01=full-AV, 02=video-only, 03=audio-only
            'channel': parts[1],   # 01=speech, 02=song
            'emotion': self.emotion_map[parts[2]],
            'intensity': 'normal' if parts[3] == '01' else 'strong',
            'statement': parts[4], # 01="Kids...", 02="Dogs..."
            'repetition': parts[5],
            'actor': parts[6],
            'gender': 'female' if int(parts[6]) % 2 == 0 else 'male'
        }
    
    def load_audio_data(self) -> List[Tuple[str, Dict]]:
        """Load audio-only files from RAVDESS dataset."""
        audio_files = []
        
        # Iterate through actor folders
        for actor_folder in sorted(os.listdir(self.dataset_path)):
            if actor_folder.startswith('Actor_'):
                actor_path = os.path.join(self.dataset_path, actor_folder)
                
                # Get all audio files
                for file in os.listdir(actor_path):
                    if file.startswith('03-') and file.endswith('.wav'):  # Audio-only files
                        file_path = os.path.join(actor_path, file)
                        metadata = self.parse_filename(file)
                        audio_files.append((file_path, metadata))
        
        return audio_files
    
    def load_video_data(self) -> List[Tuple[str, Dict]]:
        """Load video files from RAVDESS dataset."""
        video_files = []
        
        # Iterate through actor folders
        for actor_folder in sorted(os.listdir(self.dataset_path)):
            if actor_folder.startswith('Actor_'):
                actor_path = os.path.join(self.dataset_path, actor_folder)
                
                # Get all video files
                for file in os.listdir(actor_path):
                    if file.startswith('01-') and file.endswith('.mp4'):  # Full AV files
                        file_path = os.path.join(actor_path, file)
                        metadata = self.parse_filename(file)
                        video_files.append((file_path, metadata))
        
        return video_files

class CREMADLoader:
    def __init__(self):
        self.dataset_path = Config.CREMA_D_PATH
        self.emotion_map = {
            'ANG': 'angry',
            'DIS': 'disgust',
            'FEA': 'fearful',
            'HAP': 'happy',
            'NEU': 'neutral',
            'SAD': 'sad'
        }
        
        # Load demographics data
        self.demographics = pd.read_csv(
            os.path.join(self.dataset_path, 'VideoDemographics.csv')
        )
        
    def parse_filename(self, filename: str) -> Dict:
        """Parse CREMA-D filename to extract metadata."""
        # Example: 1012_IEO_ANG_XX.wav
        parts = filename.split('.')[0].split('_')
        actor_id = parts[0]
        
        # Get actor demographics
        actor_demo = self.demographics[
            self.demographics['ActorID'] == int(actor_id)
        ].iloc[0]
        
        return {
            'actor_id': actor_id,
            'sentence': parts[1],
            'emotion': self.emotion_map[parts[2]],
            'intensity': parts[3],
            'age': actor_demo['Age'],
            'sex': actor_demo['Sex'],
            'race': actor_demo['Race'],
            'ethnicity': actor_demo['Ethnicity']
        }
    
    def load_audio_data(self) -> List[Tuple[str, Dict]]:
        """Load audio files from CREMA-D dataset."""
        audio_files = []
        audio_dir = os.path.join(self.dataset_path, 'AudioWAV')
        
        for file in os.listdir(audio_dir):
            if file.endswith('.wav'):
                file_path = os.path.join(audio_dir, file)
                metadata = self.parse_filename(file)
                audio_files.append((file_path, metadata))
        
        return audio_files
    
    def load_video_data(self) -> List[Tuple[str, Dict]]:
        """Load video files from CREMA-D dataset."""
        video_files = []
        video_dir = os.path.join(self.dataset_path, 'VideoFlash')
        
        for file in os.listdir(video_dir):
            if file.endswith('.flv'):
                file_path = os.path.join(video_dir, file)
                metadata = self.parse_filename(file)
                video_files.append((file_path, metadata))
        
        return video_files