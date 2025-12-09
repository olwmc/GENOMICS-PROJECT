import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from src.dataset import GenomicsContrastiveDataset, DistanceBinBatchSampler, distance_binned_collate
from src.contrastive import ContrastiveModel
from src.autoencoders.autoencoders import SequenceAutoencoder

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, explicit_negatives=None):
        """
        anchor:     [B, D]
        positive:   [B, D]
        explicit_negatives: optional [B, K, D]
        """

        B, D = anchor.shape

        anchor = F.normalize(anchor, p=2, dim=-1)      # [B, D]
        positive = F.normalize(positive, p=2, dim=-1)  # [B, D]

        sim_matrix = anchor @ positive.t()             # [B, B]
        sim_matrix = sim_matrix / self.temperature

        pos_logits = torch.diag(sim_matrix)            # [B]

        logits = sim_matrix                           # [B, B]

        if explicit_negatives is not None:
            explicit_negatives = F.normalize(explicit_negatives, p=2, dim=-1)
            neg_logits = torch.einsum(
                "bd, bkd -> bk", anchor, explicit_negatives
            )  # [B, K]
            neg_logits = neg_logits / self.temperature

            logits = torch.cat([logits, neg_logits], dim=1)  # [B, B+K]

        targets = torch.arange(B, device=anchor.device)

        loss = F.cross_entropy(logits, targets)

        return loss

def onehot_to_tokens(onehot_seq):
    """
    Convert one-hot encoded sequences to token indices.

    Args:
        onehot_seq: [..., 4, seq_len] one-hot encoded DNA

    Returns:
        [..., seq_len] token indices
    """
    return torch.argmax(onehot_seq, dim=-2)


def main():
    dna_autoencoder = SequenceAutoencoder(
         input_channels=5,      # vocab_size
         is_dna=True,
         pool_size=100
    )

    dna_autoencoder.load_state_dict(torch.load("trained_models/dna_autoencoder.pth"))

    dna_autoencoder.eval()
    for param in dna_autoencoder.parameters():
        param.requires_grad = False
    
    # Contrastive model
    model = ContrastiveModel(
        dna_autoencoder=dna_autoencoder,
        d_embed=512, #768,
        aggregation_hidden_dim=1024,#1536,
    ).cuda()

    model = torch.compile(model)
    print("# parameters:", sum([w.numel() for w in model.parameters()]))
    
    # Dataset & DataLoader
    cache_load_dir = "/oscar/scratch/omclaugh/mango_precomputed"
    print("Loading dataset...")
    dataset = GenomicsContrastiveDataset.from_cache_dir(cache_load_dir)
    print("Dataset loaded!")

    print("Loading sampler...")
    sampler = DistanceBinBatchSampler(dataset, pairs_per_batch=64, bin_schedule="roundrobin")
    print("Sampler loaded!")

    print("Loading dataloader...")
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=lambda b: distance_binned_collate(b, "pair"),
        num_workers=4,
        pin_memory=True,
    )
    print("Dataloader loaded!")
    
    # Training setup
    criterion = InfoNCELoss(temperature=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    scaler = GradScaler()
    model.train()
    SB = 24
    for epoch in range(10):
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(loader):
            # Load data and convert from one-hot to tokens
            seq_anchor = onehot_to_tokens(batch["seq_tokens_anchor"])[0:SB].cuda()  # [B, 50, 100]
            seq_pos = onehot_to_tokens(batch["seq_tokens_pos"])[0:SB].cuda()
            seq_negs_onehot = batch["seq_tokens_negs"][0:SB].cuda()  # [B, K, 50, 4, 100]
            
            # Convert negatives from one-hot
            B, K = seq_negs_onehot.shape[:2]
            seq_negs = onehot_to_tokens(seq_negs_onehot)[0:SB]  # [B, K, 50, 100]
            
            epi_anchor = batch["epi_tokens_anchor"][0:SB].cuda()
            epi_pos = batch["epi_tokens_pos"][0:SB].cuda()
            epi_negs = batch["epi_tokens_negs"][0:SB].cuda()  # [B, K, 50, 5]
            
            with autocast():
                # Forward pass
                embed_anchor = model(seq_anchor, epi_anchor)  # [B, d_embed]
                embed_pos = model(seq_pos, epi_pos)
                
                # Flatten negatives for batch processing
                seq_negs_flat = seq_negs.view(B * K, 50, 100)  # Note: no more 4 dimension
                epi_negs_flat = epi_negs.view(B * K, 50, 5)
                embed_negs = model(seq_negs_flat, epi_negs_flat).view(B, K, -1)  # [B, K, d_embed]           
                # Loss
                loss = criterion(embed_anchor, embed_pos, explicit_negatives=embed_negs)

            
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            # Optimizer step with unscaling
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()

            if batch_idx % 25 == 0:
                with torch.no_grad():
                    # anchor: [B, d], positive: [B, d], negatives: [B, K, d]
                    pos_sim = (embed_anchor * embed_pos).sum(dim=-1)            # [B]
                    neg_sim = (embed_anchor.unsqueeze(1) * embed_negs).sum(-1)  # [B, K]
                
                    print(
                        f"[SIM] pos_mean={pos_sim.mean().item():.4f}, "
                        f"neg_mean={neg_sim.mean().item():.4f}, "
                        f"pos_min={pos_sim.min().item():.4f}, "
                        f"neg_max={neg_sim.max().item():.4f}"
                    )

                    total_norm = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    print("Total grad norm:", total_norm)
            
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(loader)
        print(f"==> Epoch {epoch} complete | Avg Loss: {avg_loss:.4f}\n")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'checkpoints/contrastive_epoch_{epoch}.pt')


if __name__ == "__main__":
    main()
