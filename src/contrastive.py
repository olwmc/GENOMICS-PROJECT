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
    ):
        """
        Args:
            dna_autoencoder: Pre-trained DNA autoencoder (100bp resolution)
            d_embed: Dimension of final embedding
            aggregation_hidden_dim: If provided, adds MLP layer before final projection
        """
        super().__init__()
        
        self.dna_autoencoder = dna_autoencoder
        self.d_embed = d_embed
        
        self.input_dim = dna_autoencoder.d_bottleneck + 5
        
        if aggregation_hidden_dim is not None:
            self.aggregation_projection = nn.Sequential(
                nn.Linear(25 * self.input_dim, aggregation_hidden_dim),
                nn.LayerNorm(aggregation_hidden_dim),
                nn.ReLU(),
                nn.Linear(aggregation_hidden_dim, aggregation_hidden_dim),
                nn.ReLU(),
                nn.Linear(aggregation_hidden_dim, d_embed),
                nn.ReLU(),
                nn.Linear(d_embed, 64)
            )
        else:
            self.aggregation_projection = nn.Linear(25 * self.input_dim, d_embed)
        
        self.layer_norm = nn.LayerNorm(64)

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
        
        # Merge adjacent pairs: 50 -> 25 bins
        reduced = encoding.view(B, 25, 2, self.input_dim).mean(dim=2)  # [B, 25, d]
        
        # Flatten and project
        flattened = reduced.view(B, -1)  # [B, 25*d]
        projected = self.aggregation_projection(flattened)  # [B, d_embed]
        projected = self.layer_norm(projected)
        
        return projected

    def encode(self, dna, epi):
        B = dna.shape[0]
        
        with torch.no_grad():
            dna_embeds, _ = self.dna_autoencoder.encode(dna.reshape(B*50, -1))
        
        dna_embeds = dna_embeds.reshape(B, 50, -1)
        
        encoding = torch.cat((dna_embeds, epi), dim=-1)
        aggregated = self._aggregate(encoding)

        return aggregated
    
    def forward(self, dna, epi):
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
