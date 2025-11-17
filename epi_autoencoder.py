import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
import math
import random
from tqdm import tqdm


class RelativePositionalEncoding(nn.Module):
    """Relative Positional Encoding for inter-bin positions"""
    def __init__(self, d_model, max_relative_position=32):
        super().__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # Create relative position embeddings
        num_embeddings = 2 * max_relative_position + 1
        self.relative_embeddings = nn.Embedding(num_embeddings, d_model)
        
    def forward(self, seq_len):
        # Create relative position matrix
        positions = torch.arange(seq_len)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        
        # Clip to max relative position
        relative_positions = torch.clamp(
            relative_positions, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        # Shift to make indices positive
        relative_positions = relative_positions + self.max_relative_position
        
        # Get embeddings
        rel_embeddings = self.relative_embeddings(relative_positions.cuda())
        return rel_embeddings


class MultiHeadAttentionWithRPE(nn.Module):
    """Multi-head attention with relative positional encoding"""
    def __init__(self, d_model, n_heads, dropout=0.1, max_relative_position=32):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.rpe = RelativePositionalEncoding(self.d_k, max_relative_position)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # Linear transformations and split into heads
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Get relative position embeddings
        rel_pos_embeddings = self.rpe(seq_len).to(x.device)
        
        # Compute attention scores with relative positions
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add relative position biases
        rel_scores = torch.einsum('bhqd,qkd->bhqk', Q, rel_pos_embeddings)
        scores = scores + rel_scores / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, V)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.W_o(context)
        
        return output


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with RPE"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, max_relative_position=32):
        super().__init__()
        self.attention = MultiHeadAttentionWithRPE(
            d_model, n_heads, dropout, max_relative_position
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class EpigenomicAutoencoder(nn.Module):
    """Epigenomic Autoencoder with shallow transformer encoder, pooling, and decoder transformer"""
    def __init__(
        self,
        n_tracks=5,
        d_model=256,
        n_heads=8,
        n_encoder_layers=2,
        n_decoder_layers=3,
        d_ff=1024,
        max_seq_len=100,
        pool_size=100,  # Compress entire sequence to 1 vector
        dropout=0.1,
        max_relative_position=32
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pool_size = pool_size
        self.max_seq_len = max_seq_len
        self.n_tracks = n_tracks
        
        # Project input features to d_model
        self.input_proj = nn.Linear(n_tracks, d_model)
        
        # Shallow encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, n_heads, d_ff, dropout, max_relative_position
            )
            for _ in range(n_encoder_layers)
        ])
        
        # Pooling layer (learned pooling)
        self.pool_proj = nn.Linear(d_model * pool_size, d_model)
        
        # Decoder transformer
        self.decoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, n_heads, d_ff, dropout, max_relative_position
            )
            for _ in range(n_decoder_layers)
        ])
        
        # Output projection back to n_tracks
        self.output_proj = nn.Linear(d_model, n_tracks)
        
        # Upsampling for decoder
        self.upsample_proj = nn.Linear(d_model, d_model * pool_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def pool(self, x):
        """Pool sequence by combining every pool_size tokens"""
        batch_size, seq_len, d_model = x.shape
        
        # Pad sequence to be divisible by pool_size
        pad_len = (self.pool_size - seq_len % self.pool_size) % self.pool_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            seq_len = seq_len + pad_len
        
        # Reshape for pooling
        x = x.view(batch_size, seq_len // self.pool_size, self.pool_size * d_model)
        
        # Apply learned pooling projection
        x = self.pool_proj(x)

        return x, seq_len
    
    def upsample(self, x, target_len):
        """Upsample compressed representation back to original length"""
        batch_size, compressed_len, d_model = x.shape
        
        # Project to higher dimension
        x = self.upsample_proj(x)
        
        # Reshape to original sequence length
        x = x.view(batch_size, compressed_len * self.pool_size, self.d_model)
        
        # Trim to target length
        x = x[:, :target_len, :]
        
        return x
    
    def encode(self, x, mask=None):
        """Encode epigenomic sequence to compressed representation"""
        # Project input to d_model
        x = self.input_proj(x)
        x = self.dropout(x)
        
        # Apply encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # Pool to compress
        compressed, padded_len = self.pool(x)
        
        return compressed, padded_len
    
    def decode(self, compressed, target_len):
        """Decode compressed representation back to epigenomic tracks"""
        # Upsample compressed representation
        x = self.upsample(compressed, target_len)
        
        # Apply decoder layers
        for layer in self.decoder_layers:
            x = layer(x)
        
        # Project to output features
        output = self.output_proj(x)
        
        return output
    
    def forward(self, x, mask=None):
        """Full forward pass through autoencoder"""
        seq_len = x.shape[1]
        
        # Encode
        compressed, padded_len = self.encode(x, mask)
        
        # Decode
        output = self.decode(compressed, seq_len)
        
        return output
    
    def get_compressed_representation(self, x, mask=None):
        """Get the compressed latent representation"""
        compressed, _ = self.encode(x, mask)
        return compressed


class EpigenomicDataset(Dataset):
    """Dataset for epigenomic tracks from h5 file, aligned with DNA sequences"""
    def __init__(self, h5_file, seq_len=100, max_sequences=1_000, 
                 chromosomes=None, fasta_file=None, test_seq=1_000):
        self.seq_len = seq_len
        self.sequences = []
        
        if chromosomes is None:
            chromosomes = [f'chr{i}' for i in range(1, 20)]
        
                
        # Replace the entire data loading section in __init__:
        
        print(f"Loading epigenomic data from {h5_file}...")
        
        # Load epigenomic data
        target_total = max_sequences + test_seq  # Collect enough for train + test
        with h5py.File(h5_file, 'r') as f:
            for chrom in chromosomes:
                if chrom not in f:
                    print(f"Chromosome {chrom} not found in h5 file, skipping...")
                    continue
                    
                data = f[chrom][:]  # Shape: (5, N) where N is number of 100bp bins
                n_tracks, n_bins = data.shape
                
                print(f"Loading {chrom}: {n_tracks} tracks × {n_bins} bins")
                
                # Extract windows of seq_len consecutive bins
                for start_bin in range(0, n_bins - seq_len + 1, seq_len // 2):
                    if len(self.sequences) >= target_total:
                        break
                        
                    window = data[:, start_bin:start_bin + seq_len]  # Shape: (5, seq_len)
                    window = window.T  # Transpose to (seq_len, 5)
                    
                    # Check for NaN/Inf values and filter
                    if not np.isnan(window).any() and not np.isinf(window).any():
                        self.sequences.append(window)
                
                if len(self.sequences) >= target_total:
                    break
        
        print(f"Loaded {len(self.sequences)} windows before shuffling")
        
        # Shuffle and split
        random.shuffle(self.sequences)
        
        # Take first max_sequences for training, next 1000 for test
        self.test_sequences = self.sequences[max_sequences:max_sequences+test_seq]
        self.sequences = self.sequences[:max_sequences]

        print(f"Training set: {len(self.sequences)} sequences")
        print(f"Test set: {len(self.test_sequences)} sequences")

        # Compute normalization statistics on training set
        all_data = np.concatenate(self.sequences, axis=0)
        self.mean = np.mean(all_data, axis=0)
        self.std = np.std(all_data, axis=0) + 1e-8
        
        print(f"Normalization - Mean: {self.mean}, Std: {self.std}")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        window = self.sequences[idx]
        # Normalize
        window = (window - self.mean) / self.std
        return torch.tensor(window, dtype=torch.float32)


def train_autoencoder(
    model,
    dataloader,
    test_data,
    epochs=25,
    lr=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """Train the epigenomic autoencoder"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = batch.to(device)
            
            # Forward pass
            output = model(batch)
            
            # Calculate loss
            loss = criterion(output, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        
        # Evaluation on test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i in range(0, len(test_data), 64):
                batch = torch.stack([torch.tensor(seq, dtype=torch.float32) 
                                   for seq in test_data[i:i+64]]).to(device)
                output = model(batch)
                test_loss += criterion(output, batch).item()
        
        test_loss /= (len(test_data) // 64 + 1)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_loss:.6f}")
        print(f"  Test Loss:  {test_loss:.6f}")
    
    return model


def main():
    # Configuration
    H5_FILE = "/oscar/scratch/omclaugh/GM12878_X.h5"  # Update this path
    SEQ_LEN = 100  # Number of consecutive 100bp bins
    BATCH_SIZE = 64
    EPOCHS = 25
    MAX_SEQUENCES = 300_000
    
    # Create dataset and dataloader
    chromosomes = [f'chr{i}' for i in range(1, 20)]
    dataset = EpigenomicDataset(
        H5_FILE, 
        seq_len=SEQ_LEN, 
        max_sequences=MAX_SEQUENCES,
        chromosomes=chromosomes,
        test_seq = MAX_SEQUENCES//10
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Create model with same architecture as DNA autoencoder
    model = EpigenomicAutoencoder(
        n_tracks=5,
        d_model=24,
        n_heads=8,
        n_encoder_layers=2,
        n_decoder_layers=3,
        d_ff=256,
        max_seq_len=SEQ_LEN,
        pool_size=100,  # Compress 100 bins -> 1 vector
        dropout=0.1,
        max_relative_position=32
    )
    
    print(model)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Normalize test sequences
    test_sequences = []
    for seq in dataset.test_sequences:
        normalized = (seq - dataset.mean) / dataset.std
        test_sequences.append(normalized)
    
    # Train model
    trained_model = train_autoencoder(
        model, 
        dataloader, 
        test_sequences,
        epochs=EPOCHS
    )
    
    # Save model
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'mean': dataset.mean,
        'std': dataset.std
    }, 'epigenomic_autoencoder.pth')
    print("Model saved to epigenomic_autoencoder.pth")
    
    # Test reconstruction
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with torch.no_grad():
        # Get sample sequences
        sample = torch.stack([torch.tensor(seq, dtype=torch.float32) 
                             for seq in test_sequences[:100]]).to(device)
        print(f"Input shape: {sample.shape}")
        
        # Reconstruct
        output = model(sample)
        print(f"Output shape: {output.shape}")
        
        # Get compressed representation
        compressed, _ = model.encode(sample)
        print(f"Compressed shape: {compressed.shape}")
        
        # Calculate reconstruction error (MSE)
        mse = F.mse_loss(output, sample).item()
        
        # Calculate per-track correlation
        sample_np = sample.cpu().numpy()
        output_np = output.cpu().numpy()
        
        print(f"\nReconstruction MSE: {mse:.6f}")
        print("\nPer-track correlations:")
        for track_idx in range(5):
            sample_track = sample_np[:, :, track_idx].flatten()
            output_track = output_np[:, :, track_idx].flatten()
            corr = np.corrcoef(sample_track, output_track)[0, 1]
            print(f"  Track {track_idx}: {corr:.4f}")


if __name__ == "__main__":
    main()
