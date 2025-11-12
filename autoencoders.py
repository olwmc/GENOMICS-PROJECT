import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
import math
from Bio import SeqIO
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


class DNATokenizer:
    """Simple DNA tokenizer for ACGT nucleotides"""
    def __init__(self):
        self.vocab = {
            'N': 0,  # Unknown/padding
            'A': 1,
            'C': 2,
            'G': 3,
            'T': 4,

            'a': 1,
            'c': 2,
            't': 3,
            'g': 4,
        }
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
    
    def encode(self, sequence):
        return [self.vocab.get(c.upper(), 0) for c in sequence]
    
    def decode(self, tokens):
        return ''.join([self.inverse_vocab.get(t, 'N') for t in tokens])


class DNAAutoencoder(nn.Module):
    """DNA Autoencoder with shallow transformer encoder, pooling, and decoder transformer"""
    def __init__(
        self,
        vocab_size=5,
        d_model=256,
        n_heads=8,
        n_encoder_layers=2,  # Shallow encoder
        n_decoder_layers=4,  # Deeper decoder
        d_ff=1024,
        max_seq_len=512,
        pool_size=4,  # Pooling factor
        dropout=0.1,
        max_relative_position=32
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pool_size = pool_size
        self.max_seq_len = max_seq_len
        
        # Embedding layer (no global positional embeddings as requested)
        self.embedding = nn.Embedding(vocab_size, d_model)
        
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
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
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
    
    def encode(self, input_ids, mask=None):
        """Encode DNA sequence to compressed representation"""
        # Embed input
        x = self.embedding(input_ids)
        x = self.dropout(x)
        
        # Apply encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # Pool to compress
        compressed, padded_len = self.pool(x)
        
        return compressed, padded_len
    
    def decode(self, compressed, target_len):
        """Decode compressed representation back to DNA sequence"""
        # Upsample compressed representation
        x = self.upsample(compressed, target_len)
        
        # Apply decoder layers
        for layer in self.decoder_layers:
            x = layer(x)
        
        # Project to vocabulary
        logits = self.output_proj(x)
        
        return logits
    
    def forward(self, input_ids, mask=None):
        """Full forward pass through autoencoder"""
        seq_len = input_ids.shape[1]
        
        # Encode
        compressed, padded_len = self.encode(input_ids, mask)
        
        # Decode
        logits = self.decode(compressed, seq_len)
        
        return logits
    
    def get_compressed_representation(self, input_ids, mask=None):
        """Get the compressed latent representation"""
        compressed, _ = self.encode(input_ids, mask)
        return compressed


class DNADataset(Dataset):
    """Dataset for DNA sequences from FASTA file"""
    def __init__(self, fasta_file, seq_len=512, max_sequences=10000):
        self.seq_len = seq_len
        self.tokenizer = DNATokenizer()
        self.sequences = []
        
        print(f"Loading sequences from {fasta_file}...")

        
        valid = [f'chr{i}' for i in range(0,20)]
        # Load sequences from FASTA file
        try:
            for i, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
                if record.id not in valid:
                    break

                seq = str(record.seq)
                
                # Split long sequences into chunks
                for j in range(0, len(seq) - seq_len + 1, seq_len // 2):
                    chunk = seq[j:j + seq_len]
                    if len(chunk) == seq_len:
                        # Make sure it's not just all NNNNN
                        if len(set(chunk)) >= 2:
                            self.sequences.append(chunk)
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1} sequences...")

        
        except FileNotFoundError:
            print(f"File {fasta_file} not found. Generating random sequences for demo...")
            # Generate random sequences for demonstration
            nucleotides = ['A', 'C', 'G', 'T']
            for _ in range(max_sequences):
                seq = ''.join(random.choices(nucleotides, k=seq_len))
                self.sequences.append(seq)


        random.shuffle(self.sequences)
        self.test_sequences = self.sequences[max_sequences: max_sequences + 100]
        assert(len(self.test_sequences) >= 100)

        self.sequences = self.sequences[:max_sequences]
        print(f"Loaded {len(self.sequences)} sequences of length {seq_len}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        tokens = self.tokenizer.encode(seq)
        return torch.tensor(tokens, dtype=torch.long)


def train_autoencoder(
    model,
    dataloader,
    epochs=1,
    lr=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """Train the DNA autoencoder"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        print("Length of dataloader:",len(dataloader))
        for batch_idx, batch in tqdm(enumerate(dataloader)):
            batch = batch.to(device)
            
            # Forward pass
            logits = model(batch)
            
            # Calculate loss
            loss = criterion(
                logits.reshape(-1, logits.shape[-1]),
                batch.reshape(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    return model


def main():
    # Configuration
    FASTA_FILE = "/oscar/scratch/omclaugh/hg38.fa"
    SEQ_LEN = 100
    BATCH_SIZE = 32
    EPOCHS = 15
    
    # Create dataset and dataloader
    dataset = DNADataset(FASTA_FILE, seq_len=SEQ_LEN, max_sequences=100_000)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Create model
    model = DNAAutoencoder(
        vocab_size=5,
        d_model=32,
        n_heads=4,
        n_encoder_layers=2,  # Shallow encoder
        n_decoder_layers=2,  # Deeper decoder
        d_ff=128,
        max_seq_len=SEQ_LEN,
        pool_size=100,
        dropout=0.1,
        max_relative_position=32
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    trained_model = train_autoencoder(model, dataloader, epochs=EPOCHS)
    
    # Save model
    torch.save(trained_model.state_dict(), 'dna_autoencoder.pth')
    print("Model saved to dna_autoencoder.pth")
    
    # Test reconstruction
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = DNATokenizer()
    
    with torch.no_grad():
        # Get a sample sequence
        sample = torch.Tensor([tokenizer.encode(seq) for seq in dataset.test_sequences]).long().to(device)#.to(device)#[-1].unsqueeze(0).to(device)
        print(sample.shape)
        
        # Reconstruct
        logits = model(sample)

        # Print out the middle shape
        middle, _ = model.encode(sample)
        print("Middle shape:", middle.shape)

        predictions = torch.argmax(logits, dim=-1)
        
        # Decode
        original = [tokenizer.decode(s) for s in sample.cpu().numpy()]
        reconstructed = [tokenizer.decode(s) for s in predictions.cpu().numpy()]
        
        # Calculate accuracy
        accuracy = (sample == predictions).float().mean().item()
        print(f"Reconstruction Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
