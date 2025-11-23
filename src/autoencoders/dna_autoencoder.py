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
from .autoencoders import SequenceAutoencoder

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

class DNADataset(Dataset):
    """Dataset for DNA sequences from FASTA file"""
    def __init__(self, fasta_file, seq_len=512, max_sequences=10_000):
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
        self.test_sequences = self.sequences[max_sequences: max_sequences + 1_000]
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
    lr=3e-4,
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
    BATCH_SIZE = 128
    EPOCHS = 30
    OUTPUT_DIR = "/users/omclaugh/owm/GENOMICS-PROJECT/trained_models/"

    model = SequenceAutoencoder(
        input_channels=5,      # vocab_size
        is_dna=True,
        pool_size=100
    )

    # Create dataset and dataloader
    dataset = DNADataset(FASTA_FILE, seq_len=SEQ_LEN, max_sequences=5_000_000)
    print("Number of test seqs:",len(dataset.test_sequences))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    trained_model = train_autoencoder(model, dataloader, epochs=EPOCHS)
    
    # Save model
    torch.save(trained_model.state_dict(), OUTPUT_DIR+'dna_autoencoder.pth')
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
