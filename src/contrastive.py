# Contrastive model for genomic sequences with epigenomic features

from src.autoencoders.autoencoders import SequenceAutoencoder
import torch.nn as nn
import torch
import torch.nn.functional as F


class ContrastiveModel(nn.Module):
    def __init__(
        self, 
        dna_autoencoder: SequenceAutoencoder,
        d_embed: int = 768,
        aggregation_hidden_dim: int = None,
        normalize_output: bool = True
    ):
        """
        Args:
            dna_autoencoder: Pre-trained DNA autoencoder (100bp resolution)
            d_embed: Dimension of final embedding
            aggregation_hidden_dim: If provided, adds MLP layer before final projection
            normalize_output: Whether to L2-normalize output embeddings (recommended for contrastive learning)
        """
        super().__init__()
        
        self.dna_autoencoder = dna_autoencoder
        self.d_embed = d_embed
        self.normalize_output = normalize_output
        
        # Input: DNA bottleneck + 5 epigenomic tracks
        self.input_dim = dna_autoencoder.d_bottleneck + 5
        
        # Aggregation: 50 bins → 25 bins → single embedding
        if aggregation_hidden_dim is not None:
            self.aggregation_projection = nn.Sequential(
                nn.Linear(25 * self.input_dim, aggregation_hidden_dim),
                nn.LayerNorm(aggregation_hidden_dim),
                nn.ReLU(),
                nn.Linear(aggregation_hidden_dim, d_embed)
            )
        else:
            self.aggregation_projection = nn.Linear(25 * self.input_dim, d_embed)
        
        self.layer_norm = nn.LayerNorm(d_embed)

    def _aggregate(self, encoding):
        """
        Aggregate 100bp bins to single embedding.
        
        Args:
            encoding: [B, 50, d_bottleneck+5]
            
        Returns:
            [B, d_embed]
        """
        B, n_bins, d = encoding.shape
        assert n_bins == 50 and d == self.input_dim
        
        # Merge adjacent pairs: 50 → 25 bins
        reduced = encoding.view(B, 25, 2, self.input_dim).mean(dim=2)  # [B, 25, d]
        
        # Flatten and project
        flattened = reduced.view(B, -1)  # [B, 25*d]
        projected = self.aggregation_projection(flattened)  # [B, d_embed]
        projected = self.layer_norm(projected)
        
        return projected

    def encode(self, dna, epi):
        """
        Encode DNA sequence with epigenomic features.
        
        Args:
            dna: [B, 50, 100, 5] DNA sequences at 100bp resolution (5kb total)
            epi: [B, 50, 5] Epigenomic tracks at 100bp resolution
            
        Returns:
            [B, d_embed] Normalized embeddings
        """
        B = dna.shape[0]
        # Encode DNA
        dna_embeds, _ = self.dna_autoencoder.encode(dna.reshape(B*50, -1))  # [B, 50, d_bottleneck]

        dna_embeds = dna_embeds.squeeze().reshape(B, 50, -1)
        
        # Concatenate with epigenomic features
        encoding = torch.cat((dna_embeds.squeeze(), epi), dim=-1)  # [B, 50, d_bottleneck+5]
        
        # Aggregate to single embedding
        aggregated = self._aggregate(encoding)  # [B, d_embed]
        
        # Normalize for contrastive learning
        if self.normalize_output:
            aggregated = F.normalize(aggregated, p=2, dim=-1)
        
        return aggregated

    def forward(self, dna, epi):
        """Alias for encode()"""
        return self.encode(dna, epi)
    

if __name__ == "__main__":
    ae = SequenceAutoencoder(
        input_channels=5,      # vocab_size
        is_dna=True,
        pool_size=100
    )

    weights = torch.load("trained_models/dna_autoencoder.pth")
    ae.load_state_dict(weights)

    model = ContrastiveModel(ae)
    for p in model.parameters():
        p.requires_grad = False

    model.cuda()

    print("Model was instantiated!")

    # Test input
    B=32*10

    dna_ex = torch.randint(0, 5, (B, 50, 100)).cuda()
    epi_ex = torch.randn(B, 50, 5).cuda()

    print(model(dna_ex, epi_ex).shape)
