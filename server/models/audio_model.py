import torch
import torch.nn as nn
import torchaudio

class AudioEmotionModel(nn.Module):
    def __init__(self, num_emotions=7):
        super().__init__()
        
        # CNN layers for MFCC features
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        # LSTM layers
        self.lstm = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_emotions)
        
        # Other layers
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # CNN layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Reshape for LSTM
        batch_size, channels, time, freq = x.size()
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, time, channels * freq)
        
        # LSTM layers
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take last output
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x