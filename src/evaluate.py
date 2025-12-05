import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import json
from pathlib import Path

from src.dataset import (
    GenomicsContrastiveDataset,
    distance_binned_collate,
)
from src.contrastive_evals import evaluate_contrastive_pairs


class ContrastiveModel(nn.Module):
    """
    Joint DNA + epigenomic encoder for 5kb loci with dedicated position token.
    (Same architecture as training script)
    """

    def __init__(
        self,
        d_base=32,
        d_epi=16,
        d_model=128,
        d_embed=512,
        n_layers=2,
        n_heads=4,
        ff_dim=512,
        max_tokens=50,
        dropout=0.1,
        max_position=100000,
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

        # 4) CLS + Position token + Local positional encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, max_tokens + 2, d_model))

        # 5) Transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        # 6) Projection head
        self.projection_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(d_model, d_embed),
        )

    def _get_positional_encoding(self, positions, d_model):
        """Create sinusoidal positional encodings for global genomic positions."""
        B = positions.shape[0]
        pe = torch.zeros(B, d_model, device=positions.device)
        
        positions_normalized = positions.float() / self.max_position
        
        div_term = torch.exp(torch.arange(0, d_model, 2, device=positions.device).float() * 
                            (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(positions_normalized.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(positions_normalized.unsqueeze(1) * div_term)
        
        return pe

    def _encode_dna(self, dna_tokens):
        """dna_tokens: [B, 50, 100]"""
        B, L, S = dna_tokens.shape
        base_emb = self.dna_embedding(dna_tokens)
        patch_emb = base_emb.mean(dim=2)
        return patch_emb

    def _encode_epi(self, epi_tokens):
        """epi_tokens: [B, 50, 5]"""
        epi = self.epi_proj(epi_tokens)
        return epi

    def encode(self, dna_tokens, epi_tokens, global_position):
        """
        Args:
            dna_tokens: [B, 50, 100]
            epi_tokens: [B, 50, 5]
            global_position: [B] - global genomic bin index
        """
        B, L, S = dna_tokens.shape
        assert L <= self.max_tokens
    
        dna_emb = self._encode_dna(dna_tokens)
        epi_emb = self._encode_epi(epi_tokens)
    
        x = torch.cat([dna_emb, epi_emb], dim=-1)
        x = self.input_proj(x)
    
        # Create position token with global position encoding
        global_pe = self._get_positional_encoding(global_position, self.d_model)
        pos_token = self.pos_token.expand(B, 1, -1) + global_pe.unsqueeze(1)
    
        cls = self.cls_token.expand(B, 1, -1)
        
        # Concatenate: [CLS, POS, feature_tokens]
        x = torch.cat([cls, pos_token, x], dim=1)
    
        # Add local positional encoding
        pos = self.pos_emb[:, :L+2, :]
        x = x + pos
    
        h = self.encoder(x)
    
        # cls pool
        h_cls = h[:, 0, :]
    
        z = self.projection_head(h_cls)
        z = F.normalize(z, p=2, dim=-1)
    
        return z

    def forward(self, dna_tokens, epi_tokens, global_position):
        return self.encode(dna_tokens, epi_tokens, global_position)


def onehot_to_tokens(onehot):
    """Convert one-hot encoding to token IDs."""
    return torch.argmax(onehot, dim=-2)


def create_test_split(dataset, test_ratio=0.15, seed=42):
    """Create a reproducible test split from the dataset."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    dataset_size = len(dataset)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    
    test_size = int(dataset_size * test_ratio)
    test_indices = indices[:test_size]
    
    print(f"Total dataset size: {dataset_size}")
    print(f"Test set size: {test_size} ({test_ratio*100:.1f}%)")
    
    return test_indices


def extract_embeddings_and_labels(model, dataset, indices, device, batch_size=128):
    """
    Extract embeddings for anchor-positive pairs and create labels.
    Returns embeddings for evaluation.
    """
    model.eval()
    
    all_anchor_embeds = []
    all_pos_embeds = []
    all_labels = []  # 1 for positive pairs
    
    # Also collect some negative pairs for comparison
    all_neg_embeds = []
    
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        collate_fn=lambda b: distance_binned_collate(b, "pair"),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    print("\nExtracting embeddings from test set...")
    
    with torch.no_grad():
        for batch in tqdm(loader):
            # Convert one-hot to tokens
            seq_anchor = onehot_to_tokens(batch["seq_tokens_anchor"]).to(device)
            seq_pos = onehot_to_tokens(batch["seq_tokens_pos"]).to(device)
            
            epi_anchor = batch["epi_tokens_anchor"].to(device).float()  # Ensure float32
            epi_pos = batch["epi_tokens_pos"].to(device).float()  # Ensure float32
            
            # Get global positions
            pos_anchor = batch["anchor_index"].to(device)
            pos_pos = batch["pos_index"].to(device)
            
            # Encode
            embed_anchor = model(seq_anchor, epi_anchor, pos_anchor)
            embed_pos = model(seq_pos, epi_pos, pos_pos)
            
            all_anchor_embeds.append(embed_anchor.cpu())
            all_pos_embeds.append(embed_pos.cpu())
            all_labels.append(torch.ones(embed_anchor.shape[0]))
            
            # Also encode first negative for each anchor (for negative pairs)
            if "seq_tokens_negs" in batch:
                seq_negs = batch["seq_tokens_negs"][:, 0]  # Take first negative [B, 50, 4, 100]
                epi_negs = batch["epi_tokens_negs"][:, 0]  # [B, 50, 5]
                pos_negs = batch["neg_indices"][:, 0]  # [B]
                
                seq_negs = onehot_to_tokens(seq_negs).to(device)
                epi_negs = epi_negs.to(device).float()  # Ensure float32
                pos_negs = pos_negs.to(device)
                
                embed_negs = model(seq_negs, epi_negs, pos_negs)
                all_neg_embeds.append(embed_negs.cpu())
    
    # Concatenate all batches
    anchor_embeds = torch.cat(all_anchor_embeds, dim=0)
    pos_embeds = torch.cat(all_pos_embeds, dim=0)
    pos_labels = torch.cat(all_labels, dim=0)
    
    print(f"Extracted {anchor_embeds.shape[0]} positive pairs")
    
    # Create negative pairs: anchor vs negative
    if all_neg_embeds:
        neg_embeds = torch.cat(all_neg_embeds, dim=0)
        
        # For evaluation, we'll create anchor-negative pairs
        # Use the same anchors paired with negatives
        neg_labels = torch.zeros(neg_embeds.shape[0])
        
        # Combine positive and negative pairs
        embeddings_a = torch.cat([anchor_embeds, anchor_embeds[:len(neg_embeds)]], dim=0)
        embeddings_b = torch.cat([pos_embeds, neg_embeds], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)
        
        print(f"Added {neg_embeds.shape[0]} negative pairs")
        print(f"Total pairs for evaluation: {embeddings_a.shape[0]}")
    else:
        # If no negatives, just use positive pairs
        embeddings_a = anchor_embeds
        embeddings_b = pos_embeds
        labels = pos_labels
    
    return embeddings_a, embeddings_b, labels


def main():
    # Configuration
    CHECKPOINT_PATH = "contrastive_checkpoint_globalpos_epoch46.pt"
    CACHE_DIR = "/oscar/scratch/omclaugh/mango_precomputed_chr1"
    TEST_RATIO = 0.10
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Thresholds for binary classification evaluation
    THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print("="*60)
    print("CONTRASTIVE MODEL EVALUATION - EPOCH 46")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Test seed: {SEED}")
    print(f"Test ratio: {TEST_RATIO}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataset = GenomicsContrastiveDataset.from_cache_dir(CACHE_DIR)
    print(f"Dataset loaded! Size: {len(dataset)}")
    
    # Create test split
    test_indices = create_test_split(dataset, test_ratio=TEST_RATIO, seed=SEED)
    
    # Initialize model with same hyperparameters as training
    print("\nInitializing model...")
    model = ContrastiveModel(
        d_base=32,
        d_epi=16,
        d_model=256,
        d_embed=512,
        n_layers=3,
        n_heads=4,
        ff_dim=1024,
        dropout=0.0,
        max_position=100000,
    ).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    
    print(f"Checkpoint loaded!")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Global step: {checkpoint['global_step']}")
    if 'avg_train_loss' in checkpoint:
        print(f"  Training loss: {checkpoint['avg_train_loss']:.4f}")
    elif 'avg_loss' in checkpoint:
        print(f"  Training loss: {checkpoint['avg_loss']:.4f}")
    
    # Extract embeddings and labels from test set
    embeddings_a, embeddings_b, labels = extract_embeddings_and_labels(
        model, dataset, test_indices, DEVICE, batch_size=128
    )
    
    # Run evaluation
    print("\n" + "="*60)
    print("RUNNING EVALUATION")
    print("="*60)
    
    results = evaluate_contrastive_pairs(
        embeddings_a=embeddings_a,
        embeddings_b=embeddings_b,
        labels=labels,
        thresholds=THRESHOLDS,
    )
    
    # Print results
    print("\n" + "="*60)
    print("GLOBAL METRICS")
    print("="*60)
    for metric, value in results["global_metrics"].items():
        print(f"{metric:20s}: {value:.4f}")
    
    print("\n" + "="*60)
    print("PER-THRESHOLD METRICS")
    print("="*60)
    
    for threshold, metrics in results["per_threshold"].items():
        print(f"\nThreshold: {threshold:.2f}")
        print("-" * 40)
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  Confusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"    TN: {cm['tn']:6d}  |  FP: {cm['fp']:6d}")
        print(f"    FN: {cm['fn']:6d}  |  TP: {cm['tp']:6d}")
    
    # Save results to JSON
    output_path = "contrastive_eval_results_epoch46.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n" + "="*60)
    print(f"Results saved to: {output_path}")
    print("="*60)
    
    # Additional analysis
    print("\n" + "="*60)
    print("ADDITIONAL ANALYSIS")
    print("="*60)
    
    # Find optimal threshold based on F1 score
    best_threshold = max(
        results["per_threshold"].items(),
        key=lambda x: x[1]["f1_score"]
    )
    print(f"Best F1 score: {best_threshold[1]['f1_score']:.4f} at threshold {best_threshold[0]:.2f}")
    
    # Separate positive and negative similarities for analysis
    n_pos = (labels == 1).sum().item()
    n_neg = (labels == 0).sum().item()
    
    from contrastive_evals import compute_cosine_similarity
    all_sims = compute_cosine_similarity(embeddings_a, embeddings_b)
    pos_sims = all_sims[labels == 1]
    neg_sims = all_sims[labels == 0]
    
    print(f"\nPositive pairs ({n_pos}):")
    print(f"  Mean similarity: {pos_sims.mean():.4f}")
    print(f"  Std similarity:  {pos_sims.std():.4f}")
    print(f"  Min similarity:  {pos_sims.min():.4f}")
    print(f"  Max similarity:  {pos_sims.max():.4f}")
    
    print(f"\nNegative pairs ({n_neg}):")
    print(f"  Mean similarity: {neg_sims.mean():.4f}")
    print(f"  Std similarity:  {neg_sims.std():.4f}")
    print(f"  Min similarity:  {neg_sims.min():.4f}")
    print(f"  Max similarity:  {neg_sims.max():.4f}")
    
    print(f"\nSeparation gap: {(pos_sims.mean() - neg_sims.mean()):.4f}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
