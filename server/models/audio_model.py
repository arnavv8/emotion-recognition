import torch
import torch.nn as nn
import torchaudio

class AudioEmotionModel(nn.Module):
    def __init__(self, num_emotions=7, dropout_rate=0.5):
        super().__init__()
        
        # Input shape: [batch_size, 1, n_mfcc * 3, time_steps]
        
        # First conv block with residual connection
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        
        # Second conv block with residual connection
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        
        # Third conv block with residual connection
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        )
        
        # Fourth conv block for deeper feature extraction
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )
        
        # Residual connections
        self.res1 = nn.Conv2d(1, 64, kernel_size=1)
        self.res2 = nn.Conv2d(64, 128, kernel_size=1)
        self.res3 = nn.Conv2d(128, 256, kernel_size=1)
        self.res4 = nn.Conv2d(256, 512, kernel_size=1)
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        
        # Fully connected layers with improved architecture
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),  # Added batch normalization
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),  # Added batch normalization
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),  # Added batch normalization
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_emotions)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Add channel dimension if not present
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        # Apply residual connections
        res1 = self.res1(x)
        x = self.conv1(x)
        x = x + res1[:, :, :x.size(2), :x.size(3)]
        
        res2 = self.res2(x)
        x = self.conv2(x)
        x = x + res2[:, :, :x.size(2), :x.size(3)]
        
        res3 = self.res3(x)
        x = self.conv3(x)
        x = x + res3[:, :, :x.size(2), :x.size(3)]
        
        res4 = self.res4(x)
        x = self.conv4(x)
        x = x + res4[:, :, :x.size(2), :x.size(3)]
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x_flat = x.view(x.size(0), -1)
        
        # Apply attention
        attention_weights = self.attention(x_flat)
        
        # Fully connected layers
        x = self.fc(x_flat)
        
        return x