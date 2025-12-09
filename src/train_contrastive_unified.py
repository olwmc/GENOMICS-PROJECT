import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
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
        dropout=0.1,   # Reduced from 0.2
    ):
        super().__init__()

        self.max_tokens = max_tokens
        self.dna_embedding = nn.Embedding(5, d_base)
        self.epi_proj = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, d_epi),
            nn.ReLU(),
        )
        self.input_proj = nn.Linear(d_base + d_epi, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, max_tokens + 1, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.projection_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),  # Reduced from 0.2
            nn.ReLU(),
            nn.Linear(d_model, d_embed),
        )

    def _encode_dna(self, dna_tokens):
        """
        dna_tokens: [B, 50, 100]
        Average 100bp into a single patch embedding.
        """
        B, L, S = dna_tokens.shape
        base_emb = self.dna_embedding(dna_tokens)   # [B, L, S, d_base]
        patch_emb = base_emb.mean(dim=2)            # [B, L, d_base]
        return patch_emb

    def _encode_epi(self, epi_tokens):
        """
        epi_tokens: [B, 50, 5]
        """
        epi = self.epi_proj(epi_tokens)             # [B, L, d_epi]
        return epi

    def encode(self, dna_tokens, epi_tokens):
        B, L, S = dna_tokens.shape       # L = 50
        assert L <= self.max_tokens
    
        dna_emb = self._encode_dna(dna_tokens)   # [B, L, d_base]
        epi_emb = self._encode_epi(epi_tokens)   # [B, L, d_epi]
    
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
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, explicit_negatives=None):
        B, K, D = explicit_negatives.shape
        
        pos_sim = (anchor * positive).sum(-1, keepdim=True) / self.temperature  # [B, 1]
        neg_sim = torch.einsum("bd,bkd->bk", anchor, explicit_negatives) / self.temperature  # [B, K]
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # [B, 9]
        targets = torch.zeros(B, dtype=torch.long, device=anchor.device)
        
        return F.cross_entropy(logits, targets)

def onehot_to_tokens(onehot):
    return torch.argmax(onehot, dim=-2)  

def compute_embedding_variance(model, dataset, num_samples=100):
    """Compute variance of embeddings to check if model is learning."""
    model.eval()
    all_embeds = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            idx = torch.randint(0, len(dataset), (1,)).item()
            sample = dataset[idx]
            
            seq = onehot_to_tokens(sample["seq_tokens_anchor"].unsqueeze(0)).cuda()
            epi = sample["epi_tokens_anchor"].unsqueeze(0).cuda()
            
            if epi.dtype == torch.float16:
                epi = epi.float()
            
            embed = model(seq, epi)
            all_embeds.append(embed)
    
    all_embeds = torch.cat(all_embeds, dim=0)
    variance = all_embeds.std(dim=0).mean().item()
    
    model.train()
    return variance

def main():

    cache_load_dir = "/oscar/scratch/omclaugh/mango_precomputed_chr1"
    print("Loading dataset...")
    dataset = GenomicsContrastiveDataset.from_cache_dir(cache_load_dir)
    print(f"Dataset loaded! Size: {len(dataset)}")

    sampler = DistanceBinBatchSampler(
        dataset, pairs_per_batch=64, bin_schedule="roundrobin"
    )

    loader = DataLoader(
        dataset,
        #batch_sampler=sampler,
        batch_size=128,
        collate_fn=lambda b: distance_binned_collate(b, "pair"),
        shuffle=True,
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
        dropout=0.0,  
    ).cuda()

    print(f"# parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = InfoNCELoss(temperature=1.0)  
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=3e-4, 
        weight_decay=0.00  
    )
    
    scaler = GradScaler()

    warmup_epochs = 0.5 
    total_epochs = 20
    
    warmup_steps = int(len(loader) * warmup_epochs)
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_steps
    )
    
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=len(loader) * (total_epochs - warmup_epochs),
        eta_min=1e-5
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps]
    )

    model.train()
    global_step = 0

    for epoch in range(total_epochs):
        epoch_loss = 0.0
        
        print(f"\n{'='*60}")
        print(f"Starting Epoch {epoch}")
        print(f"{'='*60}")

        for batch_idx, batch in enumerate(loader):

            if batch_idx == 0:
                for b in range(min(5, len(batch["anchor_index"]))):
                    anchor_idx = batch["anchor_index"][b].item()
                    pos_idx = batch["pos_index"][b].item()
                    neg_indices = batch["neg_indices"][b].cpu().numpy()
            
                    print(f"Sample {b}: anchor={anchor_idx}, pos={pos_idx}, negs={neg_indices}")
            
                    if pos_idx in neg_indices:
                        print(f"  BUG STILL PRESENT: positive {pos_idx} appears in negatives!")
                    else:
                        print(f"  ✓ Good: positive {pos_idx} not in negatives")
            global_step += 1

            # Convert 1-hot → token IDs
            seq_anchor = onehot_to_tokens(batch["seq_tokens_anchor"]).cuda()
            seq_pos    = onehot_to_tokens(batch["seq_tokens_pos"]).cuda()

            seq_negs_onehot = batch["seq_tokens_negs"].cuda()  # [B,K,50,4,100]
            Bfull, K = seq_negs_onehot.shape[:2]

            seq_negs = onehot_to_tokens(seq_negs_onehot)  # [B,K,50,100]

            epi_anchor = batch["epi_tokens_anchor"].cuda()
            epi_pos    = batch["epi_tokens_pos"].cuda()
            epi_negs   = batch["epi_tokens_negs"].cuda()  # [B,K,50,5]

            with autocast():

                embed_anchor = model(seq_anchor, epi_anchor)  # [B, d_embed]
                embed_pos    = model(seq_pos, epi_pos)

                seq_negs_flat = seq_negs.view(Bfull * K, 50, 100)
                epi_negs_flat = epi_negs.view(Bfull * K, 50, 5)

                embed_negs = model(seq_negs_flat, epi_negs_flat)
                embed_negs = embed_negs.view(Bfull, K, -1)

                loss = criterion(embed_anchor, embed_pos, explicit_negatives=embed_negs)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()

            if batch_idx % 25 == 0:
                with torch.no_grad():
                    pos_sim = (embed_anchor * embed_pos).sum(-1)
                    neg_sim = (embed_anchor.unsqueeze(1) * embed_negs).sum(-1)
                    
                    similarity_gap = pos_sim.mean() - neg_sim.mean()

                    print(
                        f"[SIM] pos_mean={pos_sim.mean().item():.4f}, "
                        f"neg_mean={neg_sim.mean().item():.4f}, "
                        f"gap={similarity_gap.item():.4f}, "
                        f"pos_min={pos_sim.min().item():.4f}, "
                        f"neg_max={neg_sim.max().item():.4f}"
                    )
                    print(f"[GRAD] norm={grad_norm:.4f} | lr={scheduler.get_last_lr()[0]:.6f}")
                    
                    # Warning flags
                    if neg_sim.max() > 0.9:
                        print("⚠️  WARNING: neg_max > 0.9, negatives too similar to positives")
                    if similarity_gap < 0.1:
                        print("⚠️  WARNING: similarity gap < 0.1, model may be collapsing")

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg Loss: {epoch_loss/(batch_idx+1):.4f}"
                )

                with torch.no_grad():
                    neg_sim = (embed_anchor.unsqueeze(1) * embed_negs).sum(-1)  # [B, K]

                    max_per_sample = neg_sim.argmax(dim=1)  # [B]
                    print(f"Which negative is 1.0? Indices: {max_per_sample.unique().tolist()}")

                    for b in range(3):
                        worst_idx = max_per_sample[b].item()
                        if neg_sim[b, worst_idx] > 0.99:
                            anchor_chr = batch["chrom"][b]
                            anchor_pos = batch["anchor_index"][b].item()

                            print(f"Sample {b}: anchor={anchor_chr}:{anchor_pos}, similarity={neg_sim[b, worst_idx]:.4f}")
                        
                        if global_step % 1000 == 0:
                            embed_var = compute_embedding_variance(model, dataset)
                            print(f"[HEALTH CHECK] Embedding variance: {embed_var:.4f}")
                            if embed_var < 0.1:
                                print("⚠️  WARNING: Low embedding variance, possible collapse")

        avg_loss = epoch_loss / len(loader)
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} complete | Avg Loss: {avg_loss:.4f}")
        print(f"{'='*60}\n")

        checkpoint_path = f"contrastive_checkpoint_epoch{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "avg_loss": avg_loss,
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint: {checkpoint_path}\n")


if __name__ == "__main__":
    main()
