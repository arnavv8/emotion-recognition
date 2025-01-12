import torch
import torch.nn as nn
import torchvision.models as models

class VideoEmotionModel(nn.Module):
    def __init__(self, num_emotions=7):
        super().__init__()
        
        # Load pretrained R3D_18
        self.base_model = models.video.r3d_18(pretrained=True)
        
        # Replace last fully connected layer
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_emotions)
        )
        
    def forward(self, x):
        return self.base_model(x)