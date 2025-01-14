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
        self.max_time_length = Config.MAX_TIME_LENGTH  # Fixed time steps for padding/trimming

    def load_audio(self, file_path: str) -> Tuple[torch.Tensor, int]:
        """
        Load an audio file, convert to mono, and resample if necessary.
        """
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample only if required
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate).to(waveform.device)
            waveform = resampler(waveform)
        
        return waveform, self.sample_rate

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute MFCC features along with delta and delta-delta features.
        """
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                'n_mels': self.n_mels,
                'n_fft': int(self.window_size * self.sample_rate),
                'hop_length': int(self.hop_length * self.sample_rate)
            }
        )
        mfcc = mfcc_transform(waveform)  # Shape: [1, n_mfcc, time_steps]

        # Compute delta and delta-delta features
        mfcc_delta = torchaudio.functional.compute_deltas(mfcc)
        mfcc_delta2 = torchaudio.functional.compute_deltas(mfcc_delta)

        # Concatenate features
        features = torch.cat([mfcc, mfcc_delta, mfcc_delta2], dim=1)  # Shape: [1, n_features, time_steps]

        return features

    def preprocess_audio(self, file_path: str) -> torch.Tensor:
        """
        Load and preprocess audio into fixed-length MFCC feature tensors.
        """
        waveform, _ = self.load_audio(file_path)
        features = self.extract_features(waveform)

        # Ensure fixed-length padding/trimming
        time_steps = features.shape[-1]
        if time_steps > self.max_time_length:
            features = features[:, :, :self.max_time_length]  # Trim
        elif time_steps < self.max_time_length:
            pad = torch.nn.ConstantPad1d((0, self.max_time_length - time_steps), 0)
            features = pad(features)  # Pad

        return features.unsqueeze(0)  # Add batch dimension for compatibility
