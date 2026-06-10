import torch
from torch import nn

class Temporal(nn.Module):
    def __init__(self, input_channels=16, output_features=128):
        super(Temporal, self).__init__()

        # Input RIR - Room Impulse Response or time-domain waveforms  (B, 16, L)
        # Multi-scale Conv2D: kernels 3x3, 15x15, 31x31
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(input_channels, 32, 3, padding=1), 
            nn.BatchNorm1d(32), 
            nn.ReLU()
        )

        self.conv15 = nn.Sequential(
            nn.Conv1d(input_channels, 32, 15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        self.conv31 = nn.Sequential(
            nn.Conv1d(input_channels, 32, 31, padding=15),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.proj = nn.Conv1d(96, 128, kernel_size=1)

        self.resnet1d = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128)
        )

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   # (B, 128, 1)
            nn.Conv1d(128, 8, 1),
            nn.ReLU(),
            nn.Conv1d(8, 128, 1),
            nn.Sigmoid()
        )
 
    def forward(self, x):

        x1 = self.conv3(x)
        x2 = self.conv15(x)
        x3 = self.conv31(x)
        x = torch.cat([x1, x2, x3], dim=1)   # (B, 96, L)
 
        residual = self.proj(x)               # (B, 128, L)
        out = self.resnet1d(x)                # (B, 128, L)
        x = torch.relu(out + residual)        # residual connection
 
        se = self.se(x)                       # (B, 128, 1)
        x = x * se                            # channel attention
 
        return x   # (B, 128, L)

        # # Multi-scale Conv1D: kernels 3, 15, 31 
        # x1 = self.conv3(x)
        # x2 = self.conv15(x)
        # x3 = self.conv31(x)

        # # Concatenation
        # x = torch.cat([x1, x2, x3], dim=1) 

        # # Residuals
        # residual = self.proj(x)
        # out = self.resnet1d(x)
        # x = torch.relu(out + residual)
        # return x # (B, 128, L)
    