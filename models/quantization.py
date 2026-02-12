import torch.nn as nn


class QuantizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        # Disable quantization for now
        return self.conv(x)


class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # Disable quantization for now
        return self.linear(x)
