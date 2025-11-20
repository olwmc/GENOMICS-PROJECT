import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import GenomicsContrastiveDataset, DistanceBinBatchSampler, distance_binned_collate
from contrastive_model import ContrastiveModel
from autoencoders.autoencoders import SequenceAutoencoder


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, anchor, positive, negatives):
        """
        Args:
            anchor: [B, d_embed] - normalized embeddings
            positive: [B, d_embed] - normalized embeddings
            negatives: [B, K, d_embed] - normalized embeddings
        """
        # Cosine similarities (inputs already normalized)
        pos_sim = (anchor * positive).sum(dim=-1) / self.temperature  # [B]
        neg_sim = torch.einsum('bd,bkd->bk', anchor, negatives) / self.temperature  # [B, K]
        
        # Concatenate: first column is positive, rest are negatives
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [B, K+1]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        return F.cross_entropy(logits, labels)


def main():
    # Load frozen DNA autoencoder
    dna_autoencoder = SequenceAutoencoder.load("checkpoints/dna_autoencoder.pt")
    dna_autoencoder.eval()
    for param in dna_autoencoder.parameters():
        param.requires_grad = False
    
    # Contrastive model
    model = ContrastiveModel(
        dna_autoencoder=dna_autoencoder,
        d_embed=768,
        aggregation_hidden_dim=1536,
        normalize_output=True
    ).cuda()
    
    # Dataset & DataLoader
    dataset = GenomicsContrastiveDataset(
        fasta_path="data/hg38.fa",
        epiphany_path="data/GM12878_X.h5",
        hic_root="data/hic/GM12878_primary",
        mode="pair",
        num_negatives=8,
        hard_negative=True,
        pos_quantile=0.8,
        neg_quantile=0.2,
        pairs_per_batch=64,
    )
    
    sampler = DistanceBinBatchSampler(dataset, pairs_per_batch=64, bin_schedule="roundrobin")
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=lambda b: distance_binned_collate(b, "pair"),
        num_workers=4,
        pin_memory=True,
    )
    
    # Training setup
    criterion = InfoNCELoss(temperature=0.07)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Train
    model.train()
    for epoch in range(10):
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(loader):
            # Load data
            seq_anchor = batch["seq_tokens_anchor"].cuda()
            seq_pos = batch["seq_tokens_pos"].cuda()
            seq_negs = batch["seq_tokens_negs"].cuda()  # [B, K, 50, 4, 100]
            
            epi_anchor = batch["epi_tokens_anchor"].cuda()
            epi_pos = batch["epi_tokens_pos"].cuda()
            epi_negs = batch["epi_tokens_negs"].cuda()  # [B, K, 50, 5]
            
            B, K = seq_negs.shape[:2]
            
            # Forward pass
            embed_anchor = model(seq_anchor, epi_anchor)  # [B, d_embed]
            embed_pos = model(seq_pos, epi_pos)
            
            # Flatten negatives for batch processing
            seq_negs_flat = seq_negs.view(B * K, 50, 4, 100)
            epi_negs_flat = epi_negs.view(B * K, 50, 5)
            embed_negs = model(seq_negs_flat, epi_negs_flat).view(B, K, -1)  # [B, K, d_embed]
            
            # Loss
            loss = criterion(embed_anchor, embed_pos, embed_negs)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
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
