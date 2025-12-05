import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from pathlib import Path

from src.dataset import (
    GenomicsContrastiveDataset,
    distance_binned_collate,
)

from src.contrastive_evals import evaluate_contrastive_pairs
from src.train_contrastive_global_pos import ContrastiveModel, onehot_to_tokens


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model with same architecture as training
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
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Training loss at checkpoint: {checkpoint['avg_train_loss']:.4f}")
    
    return model, checkpoint


def generate_evaluation_pairs(dataset, num_positive_pairs=1000, num_negative_pairs=1000, seed=42):
    """
    Generate pairs for evaluation.
    
    Args:
        dataset: The validation dataset
        num_positive_pairs: Number of positive pairs to generate
        num_negative_pairs: Number of negative pairs to generate
        seed: Random seed for reproducibility
    
    Returns:
        pairs: List of tuples (idx1, idx2, label) where label=1 for positive, 0 for negative
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    pairs = []
    dataset_size = len(dataset)
    
    print(f"Generating {num_positive_pairs} positive pairs and {num_negative_pairs} negative pairs...")
    
    # Generate positive pairs (samples that are close in genomic distance)
    # Since the dataset provides anchor-positive pairs, we can use those
    for _ in range(num_positive_pairs):
        idx = np.random.randint(0, dataset_size)
        sample = dataset[idx]
        # Create positive pair using anchor and positive from the dataset
        pairs.append((idx, 'positive', 1))
    
    # Generate negative pairs (samples that are far apart)
    for _ in range(num_negative_pairs):
        idx = np.random.randint(0, dataset_size)
        sample = dataset[idx]
        # Use one of the negative samples
        neg_idx = np.random.randint(0, 8)  # Assuming 8 negatives per sample
        pairs.append((idx, f'negative_{neg_idx}', 0))
    
    return pairs


def extract_embeddings_from_pairs(model, dataset, pairs, device='cuda', batch_size=128):
    """
    Extract embeddings for all pairs.
    
    Returns:
        embeddings_a: Tensor of anchor embeddings
        embeddings_b: Tensor of positive/negative embeddings
        labels: Tensor of labels (1 for positive, 0 for negative)
    """
    model.eval()
    
    all_embeddings_a = []
    all_embeddings_b = []
    all_labels = []
    
    print(f"Extracting embeddings for {len(pairs)} pairs...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(pairs), batch_size)):
            batch_pairs = pairs[i:i+batch_size]
            
            batch_embeds_a = []
            batch_embeds_b = []
            batch_labels = []
            
            for idx, pair_type, label in batch_pairs:
                sample = dataset[idx]
                
                # Get anchor
                seq_anchor = onehot_to_tokens(sample["seq_tokens_anchor"].unsqueeze(0)).to(device)
                epi_anchor = sample["epi_tokens_anchor"].unsqueeze(0).to(device)
                pos_anchor = torch.tensor([sample["anchor_index"]], dtype=torch.long).to(device)
                
                if epi_anchor.dtype == torch.float16:
                    epi_anchor = epi_anchor.float()
                
                embed_anchor = model(seq_anchor, epi_anchor, pos_anchor)
                
                # Get pair (positive or negative)
                if pair_type == 'positive':
                    seq_pair = onehot_to_tokens(sample["seq_tokens_pos"].unsqueeze(0)).to(device)
                    epi_pair = sample["epi_tokens_pos"].unsqueeze(0).to(device)
                    pos_pair = torch.tensor([sample["pos_index"]], dtype=torch.long).to(device)
                else:  # negative
                    neg_idx = int(pair_type.split('_')[1])
                    seq_pair = onehot_to_tokens(sample["seq_tokens_negs"][neg_idx].unsqueeze(0)).to(device)
                    epi_pair = sample["epi_tokens_negs"][neg_idx].unsqueeze(0).to(device)
                    pos_pair = torch.tensor([sample["neg_indices"][neg_idx]], dtype=torch.long).to(device)
                
                if epi_pair.dtype == torch.float16:
                    epi_pair = epi_pair.float()
                
                embed_pair = model(seq_pair, epi_pair, pos_pair)
                
                batch_embeds_a.append(embed_anchor)
                batch_embeds_b.append(embed_pair)
                batch_labels.append(label)
            
            all_embeddings_a.append(torch.cat(batch_embeds_a, dim=0))
            all_embeddings_b.append(torch.cat(batch_embeds_b, dim=0))
            all_labels.extend(batch_labels)
    
    embeddings_a = torch.cat(all_embeddings_a, dim=0)
    embeddings_b = torch.cat(all_embeddings_b, dim=0)
    labels = torch.tensor(all_labels, dtype=torch.long)
    
    return embeddings_a, embeddings_b, labels


def main():
    # Configuration
    CHECKPOINT_PATH = "contrastive_checkpoint_globalpos_epoch46.pt"
    BATCH_SIZE = 128
    NUM_POSITIVE_PAIRS = 1000
    NUM_NEGATIVE_PAIRS = 1000
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Thresholds for evaluation
    THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print("="*80)
    print("CONTRASTIVE MODEL EVALUATION")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Evaluation pairs: {NUM_POSITIVE_PAIRS} positive, {NUM_NEGATIVE_PAIRS} negative")
    print(f"Thresholds: {THRESHOLDS}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    # Note: Update the path to your actual data directory
    cache_load_dir = "/oscar/scratch/omclaugh/mango_precomputed_chr1"
    full_dataset = GenomicsContrastiveDataset.from_cache_dir(cache_load_dir)
    
    # Create same train/val split as training (seed=42)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    
    print(f"Full dataset size: {len(full_dataset)}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print()
    
    # Load model
    model, checkpoint = load_checkpoint(CHECKPOINT_PATH, device=DEVICE)
    
    # Generate evaluation pairs from validation set
    pairs = generate_evaluation_pairs(
        val_dataset, 
        num_positive_pairs=NUM_POSITIVE_PAIRS,
        num_negative_pairs=NUM_NEGATIVE_PAIRS,
        seed=SEED
    )
    
    # Extract embeddings
    embeddings_a, embeddings_b, labels = extract_embeddings_from_pairs(
        model, val_dataset, pairs, device=DEVICE, batch_size=BATCH_SIZE
    )
    
    print(f"\nEmbeddings shape: {embeddings_a.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Positive samples: {(labels == 1).sum().item()}")
    print(f"Negative samples: {(labels == 0).sum().item()}")
    print()
    
    # Run evaluation
    print("Running evaluation...")
    results = evaluate_contrastive_pairs(
        embeddings_a=embeddings_a,
        embeddings_b=embeddings_b,
        labels=labels,
        thresholds=THRESHOLDS
    )
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print("\nGlobal Metrics:")
    print(f"  ROC-AUC:           {results['global_metrics']['roc_auc']:.4f}")
    print(f"  PR-AUC:            {results['global_metrics']['pr_auc_score']:.4f}")
    print(f"  Mean Similarity:   {results['global_metrics']['mean_similarity']:.4f}")
    print(f"  Std Similarity:    {results['global_metrics']['std_similarity']:.4f}")
    
    print("\nPer-Threshold Metrics:")
    print("-" * 80)
    print(f"{'Threshold':<12} {'Accuracy':<12} {'F1':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 80)
    
    for threshold, metrics in sorted(results['per_threshold'].items()):
        print(
            f"{threshold:<12.2f} "
            f"{metrics['accuracy']:<12.4f} "
            f"{metrics['f1_score']:<12.4f} "
            f"{metrics['precision']:<12.4f} "
            f"{metrics['recall']:<12.4f}"
        )
    
    print("\nConfusion Matrices:")
    for threshold, metrics in sorted(results['per_threshold'].items()):
        cm = metrics['confusion_matrix']
        print(f"\nThreshold: {threshold:.2f}")
        print(f"  TN: {cm['tn']:<6} FP: {cm['fp']:<6}")
        print(f"  FN: {cm['fn']:<6} TP: {cm['tp']:<6}")
    
    # Save results to JSON
    output_path = "evaluation_results_epoch46.json"
    
    # Convert numpy types to Python types for JSON serialization
    results_serializable = {
        'checkpoint_path': CHECKPOINT_PATH,
        'epoch': checkpoint['epoch'],
        'num_positive_pairs': NUM_POSITIVE_PAIRS,
        'num_negative_pairs': NUM_NEGATIVE_PAIRS,
        'thresholds': THRESHOLDS,
        'global_metrics': results['global_metrics'],
        'per_threshold': {
            str(k): v for k, v in results['per_threshold'].items()
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
