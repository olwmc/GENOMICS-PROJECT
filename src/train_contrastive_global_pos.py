import numpy as np
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
    Joint DNA + epigenomic encoder for 5kb loci with dedicated position token.

    Architecture:
      Tokens: [CLS | POS | feature_token_1 | ... | feature_token_50]
      - CLS: Pooling token (output representation)
      - POS: Dedicated position token carrying global genomic position
      - Features: Pure sequence/epigenomic features (no position added)

    Inputs:
      dna_tokens: [B, 50, 100]  integers in {0..4}
      epi_tokens: [B, 50, 5]
      global_position: [B] - absolute genomic position (bin index)
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
        dropout=0.1,
        max_position=100000,  # Maximum genomic position (chr1 is ~50k bins)
    ):
        super().__init__()

        self.max_tokens = max_tokens
        self.d_model = d_model
        self.max_position = max_position

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

        # 4) CLS + Position token + Local positional encoding (within the 50 tokens)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_token = nn.Parameter(torch.zeros(1, 1, d_model))  # Dedicated position token
        self.pos_emb = nn.Parameter(torch.zeros(1, max_tokens + 2, d_model))  # +2 for CLS and POS

        # 5) Transformer encoder (over 50 tokens)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        # 6) Projection head → final embedding for InfoNCE
        self.projection_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(d_model, d_embed),
        )

    def _get_positional_encoding(self, positions, d_model):
        """
        Create sinusoidal positional encodings for global genomic positions.
        
        Args:
            positions: [B] - global bin indices (e.g., 0-50000 for chr1)
            d_model: embedding dimension
        
        Returns:
            [B, d_model] - sinusoidal position encodings
        """
        B = positions.shape[0]
        pe = torch.zeros(B, d_model, device=positions.device)
        
        # Normalize positions to [0, 1] range
        positions_normalized = positions.float() / self.max_position
        
        # Create wavelengths: from high frequency (local) to low frequency (global)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=positions.device).float() * 
                            (-np.log(10000.0) / d_model))
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(positions_normalized.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(positions_normalized.unsqueeze(1) * div_term)
        
        return pe

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

    def encode(self, dna_tokens, epi_tokens, global_position):
        """
        Args:
            dna_tokens: [B, 50, 100]
            epi_tokens: [B, 50, 5]
            global_position: [B] - global genomic bin index
        """
        B, L, S = dna_tokens.shape       # L = 50
        assert L <= self.max_tokens
    
        dna_emb = self._encode_dna(dna_tokens)   # [B, L, d_base]
        epi_emb = self._encode_epi(epi_tokens)   # [B, L, d_epi]
    
        x = torch.cat([dna_emb, epi_emb], dim=-1)    # [B, L, d_base+d_epi]
        x = self.input_proj(x)                       # [B, L, d_model]
    
        # Create position token with global position encoding
        global_pe = self._get_positional_encoding(global_position, self.d_model)  # [B, d_model]
        pos_token = self.pos_token.expand(B, 1, -1) + global_pe.unsqueeze(1)  # [B, 1, d_model]
    
        cls = self.cls_token.expand(B, 1, -1)        # [B, 1, d_model]
        
        # Concatenate: [CLS, POS, feature_tokens]
        x = torch.cat([cls, pos_token, x], dim=1)    # [B, L+2, d_model]
    
        # Add local positional encoding (for all tokens including CLS and POS)
        pos = self.pos_emb[:, :L+2, :]               # [1, L+2, d_model]
        x = x + pos
    
        h = self.encoder(x)                          # [B, L+2, d_model]
    
        # cls pool
        h_cls = h[:, 0, :]                           # [B, d_model]
    
        z = self.projection_head(h_cls)              # [B, d_embed]
        z = F.normalize(z, p=2, dim=-1)
    
        return z

    def forward(self, dna_tokens, epi_tokens, global_position):
        return self.encode(dna_tokens, epi_tokens, global_position)


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, explicit_negatives=None):
        B, K, D = explicit_negatives.shape
        
        # Positive similarity
        pos_sim = (anchor * positive).sum(-1, keepdim=True) / self.temperature  # [B, 1]
        
        # Negative similarities  
        neg_sim = torch.einsum("bd,bkd->bk", anchor, explicit_negatives) / self.temperature  # [B, K]
        
        # Concatenate: [positive, neg1, neg2, ..., neg8]
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # [B, 9]
        
        # Target is always index 0
        targets = torch.zeros(B, dtype=torch.long, device=anchor.device)
        
        return F.cross_entropy(logits, targets)


def onehot_to_tokens(onehot):
    # onehot: [..., 4, 100]
    return torch.argmax(onehot, dim=-2)  # → [..., 100]


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
            pos = torch.tensor([sample["anchor_index"]], dtype=torch.long).cuda()
            
            # FIX: Convert to float32 if needed
            if epi.dtype == torch.float16:
                epi = epi.float()
            
            embed = model(seq, epi, pos)
            all_embeds.append(embed)
    
    all_embeds = torch.cat(all_embeds, dim=0)
    variance = all_embeds.std(dim=0).mean().item()
    
    model.train()
    return variance


def evaluate_on_validation(model, val_loader, criterion):
    """Evaluate model on validation set and return metrics."""
    model.eval()
    total_loss = 0.0
    all_pos_sims = []
    all_neg_sims = []
    num_batches = 0
    
    print("\n" + "="*60)
    print("VALIDATION EVALUATION")
    print("="*60)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            # Convert 1-hot → token IDs
            seq_anchor = onehot_to_tokens(batch["seq_tokens_anchor"]).cuda()
            seq_pos = onehot_to_tokens(batch["seq_tokens_pos"]).cuda()
            seq_negs_onehot = batch["seq_tokens_negs"].cuda()
            
            Bfull, K = seq_negs_onehot.shape[:2]
            seq_negs = onehot_to_tokens(seq_negs_onehot)
            
            epi_anchor = batch["epi_tokens_anchor"].cuda()
            epi_pos = batch["epi_tokens_pos"].cuda()
            epi_negs = batch["epi_tokens_negs"].cuda()
            
            pos_anchor = batch["anchor_index"].cuda()
            pos_pos = batch["pos_index"].cuda()
            pos_negs = batch["neg_indices"].cuda()
            
            # Use autocast for mixed precision (same as training)
            with autocast():
                # Forward pass
                embed_anchor = model(seq_anchor, epi_anchor, pos_anchor)
                embed_pos = model(seq_pos, epi_pos, pos_pos)
                
                seq_negs_flat = seq_negs.view(Bfull * K, 50, 100)
                epi_negs_flat = epi_negs.view(Bfull * K, 50, 5)
                pos_negs_flat = pos_negs.view(Bfull * K)
                
                embed_negs = model(seq_negs_flat, epi_negs_flat, pos_negs_flat)
                embed_negs = embed_negs.view(Bfull, K, -1)
                
                loss = criterion(embed_anchor, embed_pos, explicit_negatives=embed_negs)
            
            total_loss += loss.item()
            
            # Compute similarities (convert back to float32 for CPU storage)
            pos_sim = (embed_anchor * embed_pos).sum(-1).float()
            neg_sim = (embed_anchor.unsqueeze(1) * embed_negs).sum(-1).float()
            
            all_pos_sims.append(pos_sim.cpu())
            all_neg_sims.append(neg_sim.cpu())
            num_batches += 1
    
    # Aggregate metrics
    all_pos_sims = torch.cat(all_pos_sims)
    all_neg_sims = torch.cat(all_neg_sims, dim=0)
    
    metrics = {
        'loss': total_loss / num_batches,
        'pos_sim_mean': all_pos_sims.mean().item(),
        'neg_sim_mean': all_neg_sims.mean().item(),
        'similarity_gap': all_pos_sims.mean().item() - all_neg_sims.mean().item(),
        'pos_sim_min': all_pos_sims.min().item(),
        'neg_sim_max': all_neg_sims.max().item(),
    }
    
    # Print validation metrics
    print("\nVALIDATION METRICS:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Positive Similarity (mean): {metrics['pos_sim_mean']:.4f}")
    print(f"  Negative Similarity (mean): {metrics['neg_sim_mean']:.4f}")
    print(f"  Similarity Gap: {metrics['similarity_gap']:.4f}")
    print(f"  Positive Similarity (min): {metrics['pos_sim_min']:.4f}")
    print(f"  Negative Similarity (max): {metrics['neg_sim_max']:.4f}")
    
    if metrics['similarity_gap'] > 0.20:
        print("  ✓ Good gap! Model is discriminating well.")
    elif metrics['neg_sim_max'] > 0.95:
        print("  ⚠️  Some hard negatives remain highly similar")
    
    print("="*60 + "\n")
    
    model.train()
    return metrics


def main():

    cache_load_dir = "/oscar/scratch/omclaugh/mango_precomputed_chr1"
    print("Loading dataset...")
    full_dataset = GenomicsContrastiveDataset.from_cache_dir(cache_load_dir)
    print(f"Full dataset loaded! Size: {len(full_dataset)}")

    # 90/10 train/validation split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        collate_fn=lambda b: distance_binned_collate(b, "pair"),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        collate_fn=lambda b: distance_binned_collate(b, "pair"),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = ContrastiveModel(
        d_base=32,
        d_epi=16,
        d_model=256,
        d_embed=512,
        n_layers=3,
        n_heads=4,
        ff_dim=1024,
        dropout=0.0,
        max_position=100000,  # chr1 has ~50k bins, use 100k for safety
    ).cuda()

    print(f"# parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = InfoNCELoss(temperature=1.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-4, 
        weight_decay=0.00
    )
    
    scaler = GradScaler()

    warmup_epochs = 0.5
    total_epochs = 50
    
    warmup_steps = int(len(train_loader) * warmup_epochs)
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_steps
    )
    
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * (total_epochs - warmup_epochs),
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

        for batch_idx, batch in enumerate(train_loader):

            if batch_idx == 0 and epoch == 0:
                # Check if any negative index equals the positive index
                for b in range(min(5, len(batch["anchor_index"]))):
                    anchor_idx = batch["anchor_index"][b].item()
                    pos_idx = batch["pos_index"][b].item()
                    neg_indices = batch["neg_indices"][b].cpu().numpy()
            
                    print(f"Sample {b}: anchor={anchor_idx}, pos={pos_idx}, negs={neg_indices}")
            
                    if pos_idx in neg_indices:
                        print(f"  ⚠️ BUG STILL PRESENT: positive {pos_idx} appears in negatives!")
                    else:
                        print(f"  ✓ Good: positive {pos_idx} not in negatives")
                
                print("\n🌍 Global position encoding enabled!")
                print(f"   Architecture: [CLS | POS | 50 feature tokens]")
                print(f"   Position token carries spatial info, features stay pure\n")
            
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
            
            # NEW: Get global positions
            pos_anchor = batch["anchor_index"].cuda()      # [B]
            pos_pos = batch["pos_index"].cuda()            # [B]
            pos_negs = batch["neg_indices"].cuda()         # [B, K]

            with autocast():
                # Pass global positions to model
                embed_anchor = model(seq_anchor, epi_anchor, pos_anchor)
                embed_pos    = model(seq_pos, epi_pos, pos_pos)

                # Flatten negatives for encoding
                seq_negs_flat = seq_negs.view(Bfull * K, 50, 100)
                epi_negs_flat = epi_negs.view(Bfull * K, 50, 5)
                pos_negs_flat = pos_negs.view(Bfull * K)  # Flatten positions too

                embed_negs = model(seq_negs_flat, epi_negs_flat, pos_negs_flat)
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

            # Enhanced logging
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
                    
                    # Success indicators
                    if similarity_gap > 0.20:
                        print("✓ Good gap! Position encoding helping.")
                    elif neg_sim.max() > 0.95:
                        print("⚠️  Some hard negatives remain")

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg Loss: {epoch_loss/(batch_idx+1):.4f}"
                )

            # Check embedding variance every 1000 steps
            if global_step % 1000 == 0:
                embed_var = compute_embedding_variance(model, train_dataset)
                print(f"[HEALTH CHECK] Embedding variance: {embed_var:.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} complete | Avg Train Loss: {avg_loss:.4f}")
        print(f"{'='*60}\n")

        # Evaluate on validation set
        val_metrics = evaluate_on_validation(model, val_loader, criterion)

        # Save checkpoint
        checkpoint_path = f"contrastive_checkpoint_globalpos_epoch{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "avg_train_loss": avg_loss,
                "val_metrics": val_metrics,
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint: {checkpoint_path}\n")

    # Final validation evaluation after all training
    print("\n" + "="*80)
    print("FINAL VALIDATION EVALUATION AFTER TRAINING")
    print("="*80)
    final_val_metrics = evaluate_on_validation(model, val_loader, criterion)
    
    print("\nTraining Complete!")
    print(f"Final Validation Loss: {final_val_metrics['loss']:.4f}")
    print(f"Final Similarity Gap: {final_val_metrics['similarity_gap']:.4f}")


if __name__ == "__main__":
    main()
