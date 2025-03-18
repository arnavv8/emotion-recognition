import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioEmotionModel(nn.Module):
    def __init__(self, num_emotions=7, dropout_rate=0.5):
        super().__init__()

        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Mish(),  # Using Mish activation
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )

        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Mish(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Mish(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        )

        # Fourth convolutional block
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Mish(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )

        # Residual connections
        self.res1 = nn.Conv2d(1, 64, kernel_size=1)
        self.res2 = nn.Conv2d(64, 128, kernel_size=1)
        self.res3 = nn.Conv2d(128, 256, kernel_size=1)
        self.res4 = nn.Conv2d(256, 512, kernel_size=1)

        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),
            nn.Tanh(),
            nn.Linear(512, 512),  # Adjusted to apply per feature
            nn.Softmax(dim=-1)  # Normalize across feature dimension
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_emotions)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)  # Use Xavier for FC layers
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Add channel dimension if not present
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        # Apply first residual connection
        res1 = self.res1(x)
        x = self.conv1(x)
        x = x + F.interpolate(res1, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Apply second residual connection
        res2 = self.res2(x)
        x = self.conv2(x)
        x = x + F.interpolate(res2, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Apply third residual connection
        res3 = self.res3(x)
        x = self.conv3(x)
        x = x + F.interpolate(res3, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Apply fourth residual connection
        res4 = self.res4(x)
        x = self.conv4(x)
        x = x + F.interpolate(res4, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Adaptive pooling
        x = self.adaptive_pool(x)

        # Flatten
        x_flat = x.view(x.size(0), -1)

        attention_weights = self.attention(x_flat)  # Shape: (batch_size, 512)
        attention_weights = attention_weights.repeat_interleave(16, dim=1)  # Expand to (batch_size, 8192)
        x_flat = x_flat * attention_weights  # Now element-wise multiplication works


        # Fully connected layers
        x = self.fc(x_flat)

        return x
