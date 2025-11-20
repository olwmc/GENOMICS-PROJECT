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
from .autoencoders import SequenceAutoencoder

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
    SEQ_LEN = 100 # 100bp segments
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
    
    model = SequenceAutoencoder(
        input_channels=5,      # vocab_size
        is_dna=False,
        pool_size=100
    )

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
