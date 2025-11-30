# src/contrastive_new.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveModel(nn.Module):
    """
    Joint DNA + epigenomic encoder for 5kb loci.

    Expects:
      dna_tokens: [B, 50, 100]  (per-locus, 50 patches × 100bp tokens, values in {0..4})
      epi_tokens: [B, 50, 5]    (5 epigenomic tracks pooled to 50 positions)
    """

    def __init__(
        self,
        d_base: int = 32,      # per-base embedding dim
        d_epi: int = 16,       # epigenomic per-position embedding dim
        d_model: int = 128,    # transformer hidden size
        d_embed: int = 512,    # final embedding dim for InfoNCE
        n_layers: int = 2,
        n_heads: int = 4,
        ff_dim: int = 512,
        max_tokens: int = 50,
    ):
        super().__init__()

        self.max_tokens = max_tokens

        # 1) DNA: embed per-base, then pool within each 100bp patch
        self.dna_embedding = nn.Embedding(5, d_base)  # A,C,G,T,N → d_base

        # 2) Epigenomic tracks: small MLP per position
        self.epi_proj = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, d_epi),
            nn.ReLU(),
        )

        # 3) Combine DNA+epi at each of 50 positions, then project to d_model
        self.input_proj = nn.Linear(d_base + d_epi, d_model)

        # 4) Positional encoding for 50 tokens (5kb / 100bp = 50)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_tokens, d_model))

        # 5) Transformer encoder over sequence of 50 fused tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 6) Projection head → final embedding
        self.projection_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_embed),
        )

    def _encode_dna(self, dna_tokens: torch.Tensor) -> torch.Tensor:
        """
        dna_tokens: [B, 50, 100] ints in {0..4}
        Returns: [B, 50, d_base] pooled over the 100bp patch.
        """
        B, L, S = dna_tokens.shape  # L should be 50, S = 100
        # [B, L, S, d_base]
        base_emb = self.dna_embedding(dna_tokens)  
        # average over 100bp within each patch → [B, L, d_base]
        patch_emb = base_emb.mean(dim=2)
        return patch_emb

    def _encode_epi(self, epi_tokens: torch.Tensor) -> torch.Tensor:
        """
        epi_tokens: [B, 50, 5]
        Returns: [B, 50, d_epi]
        """
        return self.epi_proj(epi_tokens)

    def encode(self, dna_tokens: torch.Tensor, epi_tokens: torch.Tensor) -> torch.Tensor:
        """
        Returns: [B, d_embed] embeddings suitable for InfoNCE.
        """
        B, L, S = dna_tokens.shape
        assert L <= self.max_tokens, f"Got {L} tokens, max_tokens={self.max_tokens}"
        assert epi_tokens.shape[:2] == (B, L)

        dna_emb = self._encode_dna(dna_tokens)   # [B, L, d_base]
        epi_emb = self._encode_epi(epi_tokens)   # [B, L, d_epi]

        # Scale-normalize each modality before fusion
        dna_emb = F.layer_norm(dna_emb, dna_emb.shape[-1:])
        epi_emb = F.layer_norm(epi_emb, epi_emb.shape[-1:])

        # Fuse: [B, L, d_base+d_epi] → [B, L, d_model]
        x = torch.cat([dna_emb, epi_emb], dim=-1)
        x = self.input_proj(x)

        # Add positional encoding
        pos = self.pos_emb[:, :L, :]
        x = x + pos

        # Transformer over 5kb locus (50 tokens)
        h = self.encoder(x)  # [B, L, d_model]

        # Global pooling over tokens (could also use CLS token)
        pooled = h.mean(dim=1)  # [B, d_model]

        # Projection head → contrastive embedding
        z = self.projection_head(pooled)  # [B, d_embed]
        z = F.normalize(z, p=2, dim=-1)
        return z

    def forward(self, dna_tokens: torch.Tensor, epi_tokens: torch.Tensor) -> torch.Tensor:
        return self.encode(dna_tokens, epi_tokens)

