import math
import torch
import torch.nn as nn
from src.models.modules import PositionalEncoding, SpeechFrontend
from src.models.encoder import Transformer2EncoderBlock


class DysarthricSpeechTransformer(nn.Module):
    """
    Complete Sequence-to-Sequence architecture consisting of a CNN frontend,
    Transformer-2 Encoders, and a standard PyTorch Transformer Decoder[cite: 1, 6].
    """
    def __init__(
        self,
        input_freq_bins: int,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        num_encoder_layers: int = 5,
        num_decoder_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.frontend = SpeechFrontend(input_freq_bins, d_model, dropout=dropout)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        self.pos_decoder = PositionalEncoding(d_model, dropout=dropout)

        self.encoders = nn.ModuleList(
            [
                Transformer2EncoderBlock(d_model, n_heads, dropout=dropout)
                for _ in range(num_encoder_layers)
            ]
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        self.out = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_padding_mask: torch.Tensor = None,
        tgt_padding_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        
        # 1. Acoustic Encoding
        memory = self.frontend(src)
        memory = self.pos_encoder(memory)

        # Optional: Subsample the src_padding_mask if provided (stride 8 total)
        if src_padding_mask is not None:
            src_padding_mask = src_padding_mask[:, ::8]
            # Handle edge cases where integer division leaves a dimension mismatch
            if src_padding_mask.size(1) != memory.size(1):
                src_padding_mask = src_padding_mask[:, :memory.size(1)]

        for encoder in self.encoders:
            memory = encoder(memory, src_key_padding_mask=src_padding_mask)

        # 2. Target Text Decoding
        tgt_embed = self.tgt_emb(tgt) * math.sqrt(self.d_model)
        tgt_embed = self.pos_decoder(tgt_embed)

        output = self.decoder(
            tgt=tgt_embed,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )

        # 3. Projection & Stability Cast
        # We explicitly cast to float32 at the output projection to prevent NaN
        # in the CrossEntropyLoss when training with mixed precision (autocast)[cite: 1].
        logits = self.out(output)
        return logits.float()