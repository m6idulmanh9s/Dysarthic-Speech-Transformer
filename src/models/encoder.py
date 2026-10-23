import torch
import torch.nn as nn
from src.models.modules import DepthwiseSeparableConv1d


class Transformer2EncoderBlock(nn.Module):
    """
    Modified encoder block replacing the standard FFN with Depthwise Separable Convolutions
    and adding a second attention block for deeper acoustic feature extraction[cite: 1, 6, 7].
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        # First Attention Block
        self.attn1 = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Dual Depthwise Separable Convolutions[cite: 1, 6]
        self.dw_conv1 = DepthwiseSeparableConv1d(
            d_model, d_model, kernel_size=3, padding=1
        )
        self.dw_conv2 = DepthwiseSeparableConv1d(
            d_model, d_model, kernel_size=3, padding=1
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # Second Attention Block[cite: 1, 6]
        self.attn2 = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, src_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # --- First Attention Block ---
        attn1_out, _ = self.attn1(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + self.dropout1(attn1_out))

        # --- Depthwise Separable Block ---
        res2 = x
        x_conv = x.transpose(1, 2)
        x_conv = torch.relu(self.dw_conv1(x_conv))
        x_conv = torch.relu(self.dw_conv2(x_conv))
        x_conv = x_conv.transpose(1, 2)
        x = self.norm2(res2 + self.dropout2(x_conv))

        # --- Second Attention Block ---
        res3 = x
        attn2_out, _ = self.attn2(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.norm3(res3 + self.dropout3(attn2_out))

        return x