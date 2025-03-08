import torch
import torchaudio
import numpy as np
import random
from typing import Tuple
from config import Config

class AudioProcessor:
    def __init__(self):
        self.sample_rate = Config.AUDIO_SAMPLE_RATE
        self.n_mfcc = Config.N_MFCC
        self.n_mels = Config.N_MELS
        self.n_fft = Config.N_FFT
        self.window_size = int(Config.WINDOW_SIZE * self.sample_rate)  # Convert to samples
        self.hop_length = int(Config.HOP_LENGTH * self.sample_rate)    # Convert to samples
        self.max_time_length = Config.MAX_TIME_LENGTH
        self.augmentation_enabled = Config.AUGMENTATION_ENABLED

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
        
        # Normalize audio to prevent extreme values
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        return waveform, self.sample_rate

    def apply_augmentation(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply audio augmentation techniques."""
        if not self.augmentation_enabled:
            return waveform
            
        # Only apply augmentation with 50% probability
        if random.random() < 0.5:
            return waveform
            
        augmented = waveform.clone()
        
        # Random time shift (up to 10% of the signal)
        if random.random() < 0.3:
            shift_amount = int(random.random() * 0.1 * waveform.shape[1])
            if random.random() < 0.5:  # shift right
                augmented = torch.cat([
                    torch.zeros(1, shift_amount, device=waveform.device),
                    waveform[:, :-shift_amount]
                ], dim=1)
            else:  # shift left
                augmented = torch.cat([
                    waveform[:, shift_amount:],
                    torch.zeros(1, shift_amount, device=waveform.device)
                ], dim=1)
        
        # Add random noise
        if random.random() < 0.3:
            noise_factor = Config.NOISE_FACTOR * random.random()
            noise = torch.randn_like(augmented) * noise_factor
            augmented = augmented + noise
            
        # Time masking (mask out random segments)
        if random.random() < 0.3:
            mask_size = int(random.random() * 0.1 * waveform.shape[1])
            mask_start = int(random.random() * (waveform.shape[1] - mask_size))
            augmented[:, mask_start:mask_start+mask_size] = 0
            
        # Time stretching (speed up or slow down)
        if random.random() < 0.3:
            stretch_factor = random.uniform(
                Config.TIME_STRETCH_RANGE[0],
                Config.TIME_STRETCH_RANGE[1]
            )
            
            # Create complex spectrogram for time stretching
            window = torch.hann_window(self.n_fft).to(waveform.device)
            spec_complex = torch.stft(
                augmented.squeeze(0),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=window,
                return_complex=True
            ).unsqueeze(0)
            
            # Apply time stretching
            time_stretch = torchaudio.transforms.TimeStretch(
                hop_length=self.hop_length,
                n_freq=self.n_fft // 2 + 1,
                fixed_rate=stretch_factor
            )
            stretched_complex = time_stretch(spec_complex)
            
            # Convert back to waveform
            augmented = torch.istft(
                stretched_complex.squeeze(0),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=window,
                return_complex=False
            ).unsqueeze(0)
            
        # Ensure the augmented waveform has the same length as the original
        if augmented.shape[1] > waveform.shape[1]:
            augmented = augmented[:, :waveform.shape[1]]
        elif augmented.shape[1] < waveform.shape[1]:
            augmented = torch.nn.functional.pad(
                augmented,
                (0, waveform.shape[1] - augmented.shape[1])
            )
            
        # Normalize after augmentation
        augmented = augmented / (torch.max(torch.abs(augmented)) + 1e-8)
            
        return augmented

    def extract_features(self, waveform: torch.Tensor, dataset_type: str) -> torch.Tensor:
        """Extract audio features with dataset-specific processing."""
        # Apply augmentation during training
        if self.augmentation_enabled and dataset_type != 'inference':
            waveform = self.apply_augmentation(waveform)
        
        # Ensure minimum length for feature extraction
        min_length = self.n_fft + self.hop_length * (self.max_time_length - 1)
        if waveform.shape[1] < min_length:
            waveform = torch.nn.functional.pad(waveform, (0, min_length - waveform.shape[1]))

        
        # Create window for STFT
        window = torch.hann_window(self.n_fft).to(waveform.device)
        
        # Compute mel spectrogram
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            window_fn=torch.hann_window,
            power=2.0
        )(waveform)
        
        # Convert to log scale
        mel_spectrogram = torch.log1p(mel_spectrogram)
        
        # Compute MFCC features
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                'n_fft': self.n_fft,
                'n_mels': self.n_mels,
                'hop_length': self.hop_length,
                'window_fn': torch.hann_window
            }
        )
        mfcc = mfcc_transform(waveform)

        # Compute delta and delta-delta features
        mfcc_delta = torchaudio.functional.compute_deltas(mfcc)
        mfcc_delta2 = torchaudio.functional.compute_deltas(mfcc_delta)
        
        # Extract additional features based on dataset
        if dataset_type == 'ravdess':
            # Compute complex spectrogram
            spec_complex = torch.stft(
                waveform.squeeze(0),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=window,
                return_complex=True
            )
            
            # Compute magnitude spectrogram
            spec = torch.abs(spec_complex)
            
            # Compute pitch (fundamental frequency)
            pitch = torch.log1p(torch.max(spec, dim=0)[0].unsqueeze(0))
            
            # Compute energy
            energy = torch.log1p(torch.sum(spec, dim=0).unsqueeze(0))
            
            # Concatenate all features
            features = torch.cat([
                mfcc,
                mfcc_delta,
                mfcc_delta2,
                mel_spectrogram,
                pitch.unsqueeze(0),
                energy.unsqueeze(0)
            ], dim=1)
            
        else:  # CREMA-D or inference
            # Compute complex spectrogram
            spec_complex = torch.stft(
                waveform.squeeze(0),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=window,
                return_complex=True
            )
            
            # Compute magnitude spectrogram
            spec = torch.abs(spec_complex)
            
            # Compute spectral centroid
            freqs = torch.linspace(0, self.sample_rate/2, spec.size(0), device=spec.device)
            spectral_centroid = torch.sum(freqs.unsqueeze(1) * spec, dim=0) / (torch.sum(spec, dim=0) + 1e-8)
            
            # Concatenate all features
            features = torch.cat([
                mfcc,
                mfcc_delta,
                mfcc_delta2,
                mel_spectrogram,
                spectral_centroid.unsqueeze(0).unsqueeze(0)
            ], dim=1)

        pad_size=0

        # Ensure fixed time length through adaptive pooling
        if features.size(-1) > self.max_time_length:
            # Use adaptive pooling for longer sequences
            adaptive_pool = torch.nn.AdaptiveAvgPool1d(self.max_time_length)
            features = adaptive_pool(features)
        elif features.size(-1) < self.max_time_length:
            # Pad shorter sequences
            pad_size = self.max_time_length - features.size(-1)
            features = torch.nn.functional.pad(features, (0, pad_size))

        return features

    def preprocess_audio(self, file_path: str, dataset_type: str = 'inference') -> torch.Tensor:
        """Load and preprocess audio into fixed-length feature tensors."""
        waveform, _ = self.load_audio(file_path)
        features = self.extract_features(waveform, dataset_type)
        return features.unsqueeze(0)