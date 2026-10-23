import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard Sinusoidal Positional Encoding[cite: 1]."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Time, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise Separable Convolution block used in Transformer-2 Encoders[cite: 1, 6]."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SpeechFrontend(nn.Module):
    """Convolutional frontend for acoustic feature downsampling[cite: 1, 6]."""
    def __init__(self, input_freq_bins: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            input_freq_bins, 64, kernel_size=11, stride=2, padding=5
        )
        self.conv2 = nn.Conv1d(64, 64, kernel_size=11, stride=2, padding=5)
        self.conv3 = nn.Conv1d(64, d_model, kernel_size=11, stride=2, padding=5)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch, Time, Freq_Bins) -> (Batch, Freq_Bins, Time)
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.dropout(x)
        # (Batch, d_model, Reduced_Time) -> (Batch, Reduced_Time, d_model)
        return x.transpose(1, 2)