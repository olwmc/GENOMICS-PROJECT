import torch.nn as nn
from src.rpe_transformer_block import RPETransformerBlock
class SequenceAutoencoder(nn.Module):
    """Unified autoencoder for DNA or epigenomic tracks."""
    def __init__(
        self,
        input_channels,      # n_tracks OR vocab_size
        is_dna=False,   # False = epigenomic, True = DNA tokens
        d_model=32,
        n_heads=8,
        n_encoder_layers=2,
        n_decoder_layers=3,
        d_ff=1024,
        pool_size=4,
        dropout=0.1,
        max_relative_position=32,
    ):
        super().__init__()

        self.is_dna = is_dna
        self.d_model = d_model
        self.d_bottleneck = d_model
        self.pool_size = pool_size

        if is_dna:
            self.input_proj = nn.Embedding(input_channels, d_model)
        else:
            self.input_proj = nn.Linear(input_channels, d_model)

        self.encoder_layers = nn.ModuleList([
            RPETransformerBlock(
                d_model, n_heads, d_ff, dropout, max_relative_position
            ) for _ in range(n_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            RPETransformerBlock(
                d_model, n_heads, d_ff, dropout, max_relative_position
            ) for _ in range(n_decoder_layers)
        ])

        self.pool_proj = nn.Linear(d_model * pool_size, d_model)
        self.upsample_proj = nn.Linear(d_model, d_model * pool_size)
        self.output_proj = nn.Linear(d_model, input_channels)

        self.dropout = nn.Dropout(dropout)

    def pool(self, x):
        # Arguably we don't need this but whatever it's nice to have for testing.
        B, L, D = x.shape
        pad_len = (self.pool_size - L % self.pool_size) % self.pool_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        x = x.view(B, (L + pad_len) // self.pool_size, D * self.pool_size)
        return self.pool_proj(x), L + pad_len

    def upsample(self, x, target_len):
        B, C, D = x.shape
        x = self.upsample_proj(x)
        x = x.view(B, C * self.pool_size, D)
        return x[:, :target_len]

    def encode(self, x):
        x = self.input_proj(x)
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x)
        compressed, padded_len = self.pool(x)
        return compressed, padded_len

    def decode(self, compressed, target_len):
        x = self.upsample(compressed, target_len)
        for layer in self.decoder_layers:
            x = layer(x)
        return self.output_proj(x)

    def forward(self, x):
        seq_len = x.shape[1]
        compressed, _ = self.encode(x)
        return self.decode(compressed, seq_len)

    def get_compressed_representation(self, x):
        compressed, _ = self.encode(x)
        return compressed
