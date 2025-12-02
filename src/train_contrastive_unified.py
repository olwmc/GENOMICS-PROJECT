import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.dataset import (
    GenomicsContrastiveDataset,
    DistanceBinBatchSampler,
    distance_binned_collate,
)

class ContrastiveModel(nn.Module):
    """
    Joint DNA + epigenomic encoder for 5kb loci.

    Inputs:
      dna_tokens: [B, 50, 100]  integers in {0..4}
      epi_tokens: [B, 50, 5]
    """

    def __init__(
        self,
        d_base=32,     # per-base embedding dim
        d_epi=16,      # epigenomic per-position embedding dim
        d_model=128,   # Transformer hidden size
        d_embed=512,   # final vector for InfoNCE
        n_layers=2,
        n_heads=4,
        ff_dim=512,
        max_tokens=50,
    ):
        super().__init__()

        self.max_tokens = max_tokens

        # 1) DNA base embedding
        self.dna_embedding = nn.Embedding(5, d_base)

        # 2) Epigenomic projection
        self.epi_proj = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, d_epi),
            nn.ReLU(),
        )

        # 3) Combine DNA+epi, project to d_model
        self.input_proj = nn.Linear(d_base + d_epi, d_model)

        # 4) CLS + Positional encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, max_tokens + 1, d_model))

        # 5) Transformer encoder (over 50 tokens)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        # 6) Projection head → final embedding for InfoNCE
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_embed),
            nn.BatchNorm1d(d_embed)
        )

    def _encode_dna(self, dna_tokens):
        """
        dna_tokens: [B, 50, 100]
        Average 100bp into a single patch embedding.
        """
        B, L, S = dna_tokens.shape
        base_emb = self.dna_embedding(dna_tokens)   # [B, L, S, d_base]
        patch_emb = base_emb.mean(dim=2)            # [B, L, d_base]
        return F.layer_norm(patch_emb, patch_emb.shape[-1:])

    def _encode_epi(self, epi_tokens):
        """
        epi_tokens: [B, 50, 5]
        """
        epi = self.epi_proj(epi_tokens)             # [B, L, d_epi]
        return F.layer_norm(epi, epi.shape[-1:])

    def encode(self, dna_tokens, epi_tokens):
        B, L, S = dna_tokens.shape       # L = 50
        assert L <= self.max_tokens
    
        dna_emb = self._encode_dna(dna_tokens)   # [B, L, d_base]
        epi_emb = self._encode_epi(epi_tokens)   # [B, L, d_epi]
    
        dna_emb = F.layer_norm(dna_emb, dna_emb.shape[-1:])
        epi_emb = F.layer_norm(epi_emb, epi_emb.shape[-1:])
    
        x = torch.cat([dna_emb, epi_emb], dim=-1)    # [B, L, d_base+d_epi]
        x = self.input_proj(x)                       # [B, L, d_model]
    
        cls = self.cls_token.expand(B, 1, -1)        # [B, 1, d_model]
        x = torch.cat([cls, x], dim=1)               # [B, L+1, d_model]
    
        pos = self.pos_emb[:, :L+1, :]               # [1, L+1, d_model]
        x = x + pos
    
        h = self.encoder(x)                          # [B, L+1, d_model]
    
        # cls pool
        h_cls = h[:, 0, :]                           # [B, d_model]
    
        z = self.projection_head(h_cls)              # [B, d_embed]
        z = F.normalize(z, p=2, dim=-1)
    
        return z



    def forward(self, dna_tokens, epi_tokens):
        return self.encode(dna_tokens, epi_tokens)


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, explicit_negatives=None):
        B, D = anchor.shape

        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)

        # in-batch full cross-sim
        sim_matrix = (anchor @ positive.t()) / self.temperature

        targets = torch.arange(B, device=anchor.device)

        if explicit_negatives is None:
            return F.cross_entropy(sim_matrix, targets)

        # explicit negatives: [B, K, D]
        explicit_negatives = F.normalize(explicit_negatives, p=2, dim=-1)
        B, K, D = explicit_negatives.shape

        neg_logits = torch.einsum(
            "bd,bkd->bk", anchor, explicit_negatives
        ) / self.temperature

        logits = torch.cat([sim_matrix, neg_logits], dim=1)

        return F.cross_entropy(logits, targets)


def onehot_to_tokens(onehot):
    # onehot: [..., 4, 100]
    return torch.argmax(onehot, dim=-2)  # → [..., 100]


def main():

    cache_load_dir = "/oscar/scratch/omclaugh/mango_precomputed"
    print("Loading dataset...")
    dataset = GenomicsContrastiveDataset.from_cache_dir(cache_load_dir)
    print("Dataset loaded!")

    # Sampler keeps distance bins pure
    sampler = DistanceBinBatchSampler(
        dataset, pairs_per_batch=64, bin_schedule="roundrobin"
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=lambda b: distance_binned_collate(b, "pair"),
        num_workers=4,
        pin_memory=True,
    )

    model = ContrastiveModel(
        d_base=32,
        d_epi=16,
        d_model=256,
        d_embed=512,
        n_layers=2,
        n_heads=4,
        ff_dim=1024,
    ).cuda()

    print("# parameters:", sum(p.numel() for p in model.parameters()))

    def variance_loss(z, eps=1e-4):
        # z: [B, D]
        std = torch.sqrt(z.var(dim=0) + eps)    # [D]
        return torch.mean(F.relu(1.0 - std))    # want std >= 1

    lambda_v = 0.1

    criterion = InfoNCELoss(temperature=0.07)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scaler = GradScaler()

    model.train()
    SB = 64

    batch_idx, batch = next(enumerate(loader))
    for epoch in range(10):
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(loader):

            # Convert 1-hot → token IDs
            seq_anchor = onehot_to_tokens(batch["seq_tokens_anchor"])[0:SB].cuda()
            seq_pos    = onehot_to_tokens(batch["seq_tokens_pos"])[0:SB].cuda()

            seq_negs_onehot = batch["seq_tokens_negs"][0:SB].cuda()  # [B,K,50,4,100]
            Bfull, K = seq_negs_onehot.shape[:2]

            seq_negs = onehot_to_tokens(seq_negs_onehot)  # [B,K,50,100]

            epi_anchor = batch["epi_tokens_anchor"][0:SB].cuda()
            epi_pos    = batch["epi_tokens_pos"][0:SB].cuda()
            epi_negs   = batch["epi_tokens_negs"][0:SB].cuda()  # [B,K,50,5]

            with autocast():
                embed_anchor = model(seq_anchor, epi_anchor)  # [B, d_embed]

                with torch.no_grad():
                    embed_pos    = model(seq_pos, epi_pos)

                    seq_negs_flat = seq_negs.view(Bfull * K, 50, 100)
                    epi_negs_flat = epi_negs.view(Bfull * K, 50, 5)

                    embed_negs = model(seq_negs_flat, epi_negs_flat)
                    embed_negs = embed_negs.view(Bfull, K, -1)

                contrastive_loss = criterion(embed_anchor, embed_pos, explicit_negatives=embed_negs)
                var_loss = variance_loss(embed_anchor)
                loss = contrastive_loss + lambda_v * var_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if batch_idx % 25 == 0:
                with torch.no_grad():
                    pos_sim = (embed_anchor * embed_pos).sum(-1)
                    neg_sim = (embed_anchor.unsqueeze(1) * embed_negs).sum(-1)

                    print(
                        f"[SIM] pos_mean={pos_sim.mean().item():.4f}, "
                        f"neg_mean={neg_sim.mean().item():.4f}, "
                        f"pos_min={pos_sim.min().item():.4f}, "
                        f"neg_max={neg_sim.max().item():.4f}"
                    )

                    # gradient norm
                    total_norm = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            pn = p.grad.detach().data.norm(2)
                            total_norm += pn.item() ** 2
                    print("Total grad norm:", total_norm ** 0.5)

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / len(loader)
        print(f"==> Epoch {epoch} complete | Avg Loss: {avg_loss:.4f}\n")

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            f"contrastive_checkpoint_epoch{epoch}.pt",
        )


if __name__ == "__main__":
    main()

