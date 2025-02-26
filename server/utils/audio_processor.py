import torch
import torchaudio
import numpy as np
from typing import Tuple
from config import Config

class AudioProcessor:
    def __init__(self):
        self.sample_rate = Config.AUDIO_SAMPLE_RATE
        self.n_mfcc = Config.N_MFCC
        self.n_mels = Config.N_MELS
        self.window_size = Config.WINDOW_SIZE
        self.hop_length = Config.HOP_LENGTH
        self.max_time_length = Config.MAX_TIME_LENGTH

    def load_audio(self, file_path: str) -> Tuple[torch.Tensor, int]:
        """Load and preprocess audio file."""
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Convert stereo to mono by averaging channels
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)
        
        # Apply pre-emphasis filter
        waveform = torch.cat([
            waveform[:, :1],
            waveform[:, 1:] - 0.97 * waveform[:, :-1]
        ], dim=1)
        
        return waveform, self.sample_rate

    def extract_features(self, waveform: torch.Tensor, dataset_type: str) -> torch.Tensor:
        """Extract audio features with dataset-specific processing."""
        # Compute MFCC features
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                'n_mels': self.n_mels,
                'n_fft': int(self.window_size * self.sample_rate),
                'hop_length': int(self.hop_length * self.sample_rate),
                'window_fn': torch.hann_window
            }
        )
        mfcc = mfcc_transform(waveform)

        # Compute delta and delta-delta features
        mfcc_delta = torchaudio.functional.compute_deltas(mfcc)
        mfcc_delta2 = torchaudio.functional.compute_deltas(mfcc_delta)

        # Extract additional features based on dataset
        if dataset_type == 'ravdess':
            # RAVDESS-specific: Add pitch and energy features
            spectral = torchaudio.transforms.Spectrogram(
                n_fft=int(self.window_size * self.sample_rate),
                hop_length=int(self.hop_length * self.sample_rate)
            )(waveform)
            
            # Compute pitch
            pitch = torch.log1p(
                torch.max(spectral, dim=1)[0].unsqueeze(1)
            )
            
            # Compute energy
            energy = torch.log1p(
                torch.sum(spectral, dim=1).unsqueeze(1)
            )
            
            # Concatenate all features
            features = torch.cat([
                mfcc,
                mfcc_delta,
                mfcc_delta2,
                pitch,
                energy
            ], dim=1)
            
        else:  # CREMA-D
            # CREMA-D-specific: Add spectral features
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=int(self.window_size * self.sample_rate),
                hop_length=int(self.hop_length * self.sample_rate)
            )(waveform)
            
            # Log-mel spectrogram
            mel_spectrogram = torch.log1p(mel_spectrogram)
            
            # Concatenate all features
            features = torch.cat([
                mfcc,
                mfcc_delta,
                mfcc_delta2,
                mel_spectrogram
            ], dim=1)

        return features

    def preprocess_audio(self, file_path: str, dataset_type: str = 'cremad') -> torch.Tensor:
        """Load and preprocess audio into fixed-length feature tensors."""
        waveform, _ = self.load_audio(file_path)
        features = self.extract_features(waveform, dataset_type)

        # Ensure fixed-length through adaptive padding/trimming
        time_steps = features.shape[-1]
        if time_steps > self.max_time_length:
            # Take center portion for long sequences
            start = (time_steps - self.max_time_length) // 2
            features = features[:, :, start:start + self.max_time_length]
        elif time_steps < self.max_time_length:
            # Pad shorter sequences with reflection padding
            pad_size = self.max_time_length - time_steps
            pad_left = pad_size // 2
            pad_right = pad_size - pad_left
            features = torch.nn.functional.pad(
                features,
                (pad_left, pad_right),
                mode='reflect'
            )

        return features.unsqueeze(0)  # Add batch dimension