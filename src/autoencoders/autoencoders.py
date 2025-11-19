class SequenceAutoencoder(nn.Module):
    """Unified autoencoder for DNA or epigenomic tracks."""
    def __init__(
        self,
        input_channels,      # n_tracks OR vocab_size
        is_discrete=False,   # False = epigenomic, True = DNA tokens
        d_model=256,
        n_heads=8,
        n_encoder_layers=2,
        n_decoder_layers=3,
        d_ff=1024,
        pool_size=4,
        dropout=0.1,
        max_relative_position=32,
    ):
        super().__init__()

        self.is_discrete = is_discrete
        self.d_model = d_model
        self.pool_size = pool_size

        if is_discrete:
            self.input_proj = nn.Embedding(input_channels, d_model)
        else:
            self.input_proj = nn.Linear(input_channels, d_model)

        self.encoder_layers = nn.ModuleList([
            TransformerBlockWithRPE(
                d_model, n_heads, d_ff, dropout, max_relative_position
            ) for _ in range(n_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            TransformerBlockWithRPE(
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

    def encode(self, x, mask=None):
        x = self.input_proj(x)
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        compressed, padded_len = self.pool(x)
        return compressed, padded_len

    def decode(self, compressed, target_len):
        x = self.upsample(compressed, target_len)
        for layer in self.decoder_layers:
            x = layer(x)
        return self.output_proj(x)

    def forward(self, x, mask=None):
        seq_len = x.shape[1]
        compressed, _ = self.encode(x, mask)
        return self.decode(compressed, seq_len)

    def get_compressed_representation(self, x, mask=None):
        compressed, _ = self.encode(x, mask)
        return compressed


# class EpigenomicAutoencoder(nn.Module):
#     """Epigenomic Autoencoder with shallow transformer encoder, pooling, and decoder transformer"""
#     def __init__(
#         self,
#         n_tracks=5,
#         d_model=256,
#         n_heads=8,
#         n_encoder_layers=2,
#         n_decoder_layers=3,
#         d_ff=1024,
#         max_seq_len=100,
#         pool_size=100,  # Compress entire sequence to 1 vector
#         dropout=0.1,
#         max_relative_position=32
#     ):
#         super().__init__()
        
#         self.d_model = d_model
#         self.pool_size = pool_size
#         self.max_seq_len = max_seq_len
#         self.n_tracks = n_tracks
        
#         # Project input features to d_model
#         self.input_proj = nn.Linear(n_tracks, d_model)
        
#         # Shallow encoder
#         self.encoder_layers = nn.ModuleList([
#             TransformerBlockWithRPE(
#                 d_model, n_heads, d_ff, dropout, max_relative_position
#             )
#             for _ in range(n_encoder_layers)
#         ])
        
#         # Pooling layer (learned pooling)
#         self.pool_proj = nn.Linear(d_model * pool_size, d_model)
        
#         # Decoder transformer
#         self.decoder_layers = nn.ModuleList([
#             TransformerBlockWithRPE(
#                 d_model, n_heads, d_ff, dropout, max_relative_position
#             )
#             for _ in range(n_decoder_layers)
#         ])
        
#         # Output projection back to n_tracks
#         self.output_proj = nn.Linear(d_model, n_tracks)
        
#         # Upsampling for decoder
#         self.upsample_proj = nn.Linear(d_model, d_model * pool_size)
        
#         self.dropout = nn.Dropout(dropout)
        
#     def pool(self, x):
#         """Pool sequence by combining every pool_size tokens"""
#         batch_size, seq_len, d_model = x.shape
        
#         # Pad sequence to be divisible by pool_size
#         pad_len = (self.pool_size - seq_len % self.pool_size) % self.pool_size
#         if pad_len > 0:
#             x = F.pad(x, (0, 0, 0, pad_len))
#             seq_len = seq_len + pad_len
        
#         # Reshape for pooling
#         x = x.view(batch_size, seq_len // self.pool_size, self.pool_size * d_model)
        
#         # Apply learned pooling projection
#         x = self.pool_proj(x)

#         return x, seq_len
    
#     def upsample(self, x, target_len):
#         """Upsample compressed representation back to original length"""
#         batch_size, compressed_len, d_model = x.shape
        
#         # Project to higher dimension
#         x = self.upsample_proj(x)
        
#         # Reshape to original sequence length
#         x = x.view(batch_size, compressed_len * self.pool_size, self.d_model)
        
#         # Trim to target length
#         x = x[:, :target_len, :]
        
#         return x
    
#     def encode(self, x, mask=None):
#         """Encode epigenomic sequence to compressed representation"""
#         # Project input to d_model
#         x = self.input_proj(x)
#         x = self.dropout(x)
        
#         # Apply encoder layers
#         for layer in self.encoder_layers:
#             x = layer(x, mask)
        
#         # Pool to compress
#         compressed, padded_len = self.pool(x)
        
#         return compressed, padded_len
    
#     def decode(self, compressed, target_len):
#         """Decode compressed representation back to epigenomic tracks"""
#         # Upsample compressed representation
#         x = self.upsample(compressed, target_len)
        
#         # Apply decoder layers
#         for layer in self.decoder_layers:
#             x = layer(x)
        
#         # Project to output features
#         output = self.output_proj(x)
        
#         return output
    
#     def forward(self, x, mask=None):
#         """Full forward pass through autoencoder"""
#         seq_len = x.shape[1]
        
#         # Encode
#         compressed, padded_len = self.encode(x, mask)
        
#         # Decode
#         output = self.decode(compressed, seq_len)
        
#         return output
    
#     def get_compressed_representation(self, x, mask=None):
#         """Get the compressed latent representation"""
#         compressed, _ = self.encode(x, mask)
#         return compressed


# class DNAAutoencoder(nn.Module):
#     """DNA Autoencoder with shallow transformer encoder, pooling, and decoder transformer"""
#     def __init__(
#         self,
#         vocab_size=5,
#         d_model=256,
#         n_heads=8,
#         n_encoder_layers=2,  # Shallow encoder
#         n_decoder_layers=4,  # Deeper decoder
#         d_ff=1024,
#         max_seq_len=512,
#         pool_size=4,  # Pooling factor
#         dropout=0.1,
#         max_relative_position=32
#     ):
#         super().__init__()
        
#         self.d_model = d_model
#         self.pool_size = pool_size
#         self.max_seq_len = max_seq_len
        
#         # Embedding layer (no global positional embeddings as requested)
#         self.embedding = nn.Embedding(vocab_size, d_model)
        
#         # Shallow encoder
#         self.encoder_layers = nn.ModuleList([
#             TransformerEncoderLayer(
#                 d_model, n_heads, d_ff, dropout, max_relative_position
#             )
#             for _ in range(n_encoder_layers)
#         ])
        
#         # Pooling layer (learned pooling)
#         self.pool_proj = nn.Linear(d_model * pool_size, d_model)
        
#         # Decoder transformer
#         self.decoder_layers = nn.ModuleList([
#             TransformerEncoderLayer(
#                 d_model, n_heads, d_ff, dropout, max_relative_position
#             )
#             for _ in range(n_decoder_layers)
#         ])
        
#         # Output projection
#         self.output_proj = nn.Linear(d_model, vocab_size)
        
#         # Upsampling for decoder
#         self.upsample_proj = nn.Linear(d_model, d_model * pool_size)
        
#         self.dropout = nn.Dropout(dropout)
        
#     def pool(self, x):
#         """Pool sequence by combining every pool_size tokens"""
#         batch_size, seq_len, d_model = x.shape
        
#         # Pad sequence to be divisible by pool_size
#         pad_len = (self.pool_size - seq_len % self.pool_size) % self.pool_size
#         if pad_len > 0:
#             x = F.pad(x, (0, 0, 0, pad_len))
#             seq_len = seq_len + pad_len
        
#         # Reshape for pooling
#         x = x.view(batch_size, seq_len // self.pool_size, self.pool_size * d_model)
        
#         # Apply learned pooling projection
#         x = self.pool_proj(x)

#         return x, seq_len
    
#     def upsample(self, x, target_len):
#         """Upsample compressed representation back to original length"""
#         batch_size, compressed_len, d_model = x.shape
        
#         # Project to higher dimension
#         x = self.upsample_proj(x)
        
#         # Reshape to original sequence length
#         x = x.view(batch_size, compressed_len * self.pool_size, self.d_model)
        
#         # Trim to target length
#         x = x[:, :target_len, :]
        
#         return x
    
#     def encode(self, input_ids, mask=None):
#         """Encode DNA sequence to compressed representation"""
#         # Embed input
#         x = self.embedding(input_ids)
#         x = self.dropout(x)
        
#         # Apply encoder layers
#         for layer in self.encoder_layers:
#             x = layer(x, mask)
        
#         # Pool to compress
#         compressed, padded_len = self.pool(x)
        
#         return compressed, padded_len
    
#     def decode(self, compressed, target_len):
#         """Decode compressed representation back to DNA sequence"""
#         # Upsample compressed representation
#         x = self.upsample(compressed, target_len)
        
#         # Apply decoder layers
#         for layer in self.decoder_layers:
#             x = layer(x)
        
#         # Project to vocabulary
#         logits = self.output_proj(x)
        
#         return logits
    
#     def forward(self, input_ids, mask=None):
#         """Full forward pass through autoencoder"""
#         seq_len = input_ids.shape[1]
        
#         # Encode
#         compressed, padded_len = self.encode(input_ids, mask)
        
#         # Decode
#         logits = self.decode(compressed, seq_len)
        
#         return logits
    
#     def get_compressed_representation(self, input_ids, mask=None):
#         """Get the compressed latent representation"""
#         compressed, _ = self.encode(input_ids, mask)
#         return compressed