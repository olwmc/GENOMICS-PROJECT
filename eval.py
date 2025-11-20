import torch
import numpy as np
import os
import argparse
import scipy.stats as ss
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# TODO: insert mode/result as needed
# TODO: Import specific Dataset class (e.g., Chip2HiCDataset)
# TODO: Import specific Model class (e.g., EmbeddingNet)

#############################################
# 1. Embedding → Distance → Contact Map
#############################################

def pairwise_distances(emb, metric="cosine"):
    """
    Compute pairwise distances between N bin embeddings.

    If metric is 'cosine':
    $$ d(u, v) = 1 - \frac{u \cdot v}{||u||_2 ||v||_2} $$

    Args:
        emb: (N, d) numpy array of bin embeddings.
    Returns:
        dist: (N, N) symmetric distance matrix.
    """
    # Normalize for cosine similarity
    if metric == "cosine":
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        sim = emb @ emb.T
        dist = 1 - sim
    else:
        # Euclidean fallback if needed
        # $$ d(u, v) = ||u - v||_2 $$
        from scipy.spatial.distance import pdist, squareform
        dist = squareform(pdist(emb, metric='euclidean'))
        
    return dist


def distance_to_contact(dist, mode="inv_sq"):
    """
    Convert distances to contact predictions ($P_{ij}$).
    
    Modes:
    - 'inv_sq': $$ P_{ij} \propto \frac{1}{d(x_i, x_j)^2 + \epsilon} $$
    - 'exp':    $$ P_{ij} \propto e^{-d(x_i, x_j)} $$
    """
    if mode == "inv_sq":
        return 1.0 / (dist**2 + 1e-6)
    elif mode == "exp":
        return np.exp(-dist)
    else:
        raise ValueError("Unknown distance→contact mapping")


#############################################
# 2. O/E Normalization
#############################################

def observed_expected(mat):
    """
    Compute Observed/Expected (O/E) matrix to control for genomic distance decay.
    
    $$ OE_{ij} = \frac{M_{ij}}{E(|i-j|)} $$
    
    Where $E(d)$ is the average contact frequency at distance $d$.
    """
    N = mat.shape[0]
    mat = mat.copy()
    expected = np.zeros_like(mat)

    # Calculate expected value for every diagonal offset k
    for k in range(N):
        # Extract kth diagonal
        diag = np.diag(mat, k=k)
        if len(diag) > 0:
            # $$ E_k = \frac{1}{N-k} \sum_{i} M_{i, i+k} $$
            expected_val = np.nanmean(diag)
            
            # Fill both upper and lower triangles
            expected += np.diag(np.full(len(diag), expected_val), k=k)
            if k > 0:
                expected += np.diag(np.full(len(diag), expected_val), k=-k)

    oe = mat / (expected + 1e-6)
    return oe


#############################################
# 3. Distance-stratified metrics
#############################################

def stratified_correlations(pred_oe, true_oe):
    """
    Compute Pearson ($r$) and Spearman ($\rho$) correlations per genomic distance.
    This isolates the structural prediction capability from the distance decay.
    """
    N = pred_oe.shape[0]
    pearsons = []
    spearmans = []

    # Iterate over diagonals (genomic distances)
    # Start at 1 to skip self-interaction, prevent noise at very long range
    for d in range(1, min(N, 1000)): 
        p_diag = np.diag(pred_oe, k=d)
        t_diag = np.diag(true_oe, k=d)
        
        # Mask NaNs and Infs
        mask = np.isfinite(p_diag) & np.isfinite(t_diag)
        
        if mask.sum() < 10:  # Require minimum samples for correlation
            continue
            
        pearsons.append(ss.pearsonr(p_diag[mask], t_diag[mask])[0])
        spearmans.append(ss.spearmanr(p_diag[mask], t_diag[mask])[0])

    return pearsons, spearmans


#############################################
# 4. Insulation Score (Crane et al.)
#############################################
# 

def compute_insulation_score(mat, window=20):
    """
    Compute insulation score to identify TAD boundaries (Crane et al., 2015).
    
    Score is the log2 ratio of local contact density to a smoothed background.
    Note: If bins are 5kb, window=20 implies a 100kb window.
    """
    N = mat.shape[0]
    ins = np.full(N, np.nan)

    for i in range(window, N - window):
        # Diamond window around the diagonal
        sub = mat[i - window : i, i : i + window]
        # Sum or mean of contacts in the diamond
        ins[i] = np.nanmean(sub)

    # Normalize (log2 ratio)
    # Handle zeros/nans by replacing with small epsilon
    valid_mask = np.isfinite(ins)
    ins_valid = ins[valid_mask]
    
    if len(ins_valid) == 0:
        return ins
        
    smooth = ss.uniform_filter1d(ins_valid, size=window)
    ins[valid_mask] = np.log2((ins_valid + 1e-8) / (smooth + 1e-8))

    return ins


#############################################
# 5. Loop calling (Proxy for HiCCUPS)
#############################################
# 

[Image of chromatin loop detection]


def detect_loops_heuristic(mat, threshold=2.0):
    """
    Heuristic local-max loop calling to approximate HiCCUPS for evaluation.
    Returns set of (i,j) coordinates.
    """
    N = mat.shape[0]
    loops = []
    
    # Heuristic: Look for peaks in O/E map
    # Only look at off-diagonal elements within reasonable range
    # (e.g., < 2Mb distance, > 20kb distance)
    min_dist = 4
    max_dist = 400 

    for i in range(0, N - min_dist):
        for j in range(i + min_dist, min(N, i + max_dist)):
            patch = mat[i-1 : i+2, j-1 : j+2]
            center = mat[i, j]
            
            if not np.isfinite(center):
                continue
                
            # Check if center is strictly greater than neighbors (local max)
            if center > np.nanmax(patch) - 1e-8 and center > threshold:
                loops.append((i, j))

    return set(loops)


def loop_auroc(pred_matrix, true_loops_set, N):
    """
    Compute AUROC for loop detection.
    
    y_true: Binary map (1 if loop in Ground Truth, 0 otherwise)
    y_score: Predicted contact values at those positions
    """
    # Flattening N*N is memory intensive. 
    # We sample negatives to balance, or just use flattened if N is small (<5000).
    
    y_true = np.zeros((N, N), dtype=np.int8)
    for r, c in true_loops_set:
        y_true[r, c] = 1
        
    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = pred_matrix.flatten()
    
    # Handle NaNs in prediction
    mask = np.isfinite(y_pred_flat)
    
    try:
        score = roc_auc_score(y_true_flat[mask], y_pred_flat[mask])
    except ValueError:
        score = 0.5 # Edge case: only one class present
        
    return score


#############################################
# 6. Full evaluation for a chromosome
#############################################

def evaluate_chromosome(model, dataset, chrom):
    """
    Full pipeline:
    1. Inference -> Embeddings
    2. Distance -> Contact Map ($P_{ij}$)
    3. O/E Normalization
    4. Metrics: Pearson/Spearman (Stratified), Insulation, Loop AUROC
    """
    print(f"\nEvaluating {chrom}")

    # TODO: Adapt logic to extract specific indices for the given chrom from your dataset
    # idxs = [i for i, c in enumerate(dataset.chroms) if c == chrom][0]
    # size = dataset.sizes[idxs]
    # start = sum(dataset.sizes[:idxs])
    
    # Mocking size for syntax check - replace with dataset logic above
    size = 100 
    
    emb_list = []
    gt_list = []

    # Inference Loop
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(size), desc="Inference"):
            # TODO: Get X, y from dataset
            # X, y = dataset[start + i]
            # X = torch.tensor(X).unsqueeze(0).to(device)

            # TODO: Forward pass
            # emb = model(X) 
            
            # Placeholder for demonstration
            emb = np.random.randn(1, 256) # (1 x d)
            y_vec = np.random.randn(1, size) # GT row for this bin
            
            emb_list.append(emb.squeeze())
            gt_list.append(y_vec.squeeze())

    emb_full = np.vstack(emb_list) # (N, d)
    gt_full = np.vstack(gt_list)   # (N, N)

    # 1. Distance → Contact
    dist = pairwise_distances(emb_full, metric="cosine")
    pred = distance_to_contact(dist, mode="inv_sq")

    # 2. O/E Normalization
    print("Computing O/E...")
    pred_oe = observed_expected(pred)
    gt_oe = observed_expected(gt_full)

    # 3. Correlations
    print("Computing correlations...")
    pearsons, spearmans = stratified_correlations(pred_oe, gt_oe)

    # 4. Insulation Scores
    print("Computing insulation...")
    pred_ins = compute_insulation_score(pred, window=20) # Adjust window based on bin size
    gt_ins = compute_insulation_score(gt_full, window=20)
    
    # Filter NaNs for correlation
    mask_ins = np.isfinite(pred_ins) & np.isfinite(gt_ins)
    if mask_ins.sum() > 0:
        ins_corr = ss.pearsonr(pred_ins[mask_ins], gt_ins[mask_ins])[0]
    else:
        ins_corr = 0.0

    # 5. Loops
    print("Computing loops...")
    # Use Ground Truth O/E to define "True Loops" if external bedpe not available
    true_loops = detect_loops_heuristic(gt_oe, threshold=2.0) 
    auroc = loop_auroc(pred_oe, true_loops, pred.shape[0])

    return {
        "pearson": np.nanmean(pearsons),
        "spearman": np.nanmean(spearmans),
        "insulation_corr": ins_corr,
        "loop_auroc": auroc,
        "pred_matrix": pred,
        "true_matrix": gt_full,
    }


#############################################
# 7. Main
#############################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--save_dir", default="./eval_results")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # evaluating final performance on chromosomes 20–22
    test_chroms = ["chr20", "chr21", "chr22"]

    # TODO: insert mode/result as needed (Load Dataset)
    # dataset = Chip2HiCDataset(mode="test", chroms=test_chroms)
    
    # TODO: insert mode/result as needed (Load Model)
    # model = EmbeddingNet()
    # model.load_state_dict(torch.load(args.model_path))
    
    # Mock objects for running the script without the external deps
    dataset = None 
    class MockModel(torch.nn.Module): pass
    model = MockModel()

    out = {}

    for chrom in test_chroms:
        # Pass dataset/model to eval function
        metrics = evaluate_chromosome(model, dataset, chrom)
        out[chrom] = metrics

        # Save visualizations
        # Paper mentions: "visualize 3D genome architecture"
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"{chrom} Prediction")
        plt.imshow(np.log1p(metrics["pred_matrix"]), cmap="RdBu_r", vmin=0, vmax=1)
        plt.subplot(1, 2, 2)
        plt.title(f"{chrom} Ground Truth")
        plt.imshow(np.log1p(metrics["true_matrix"]), cmap="RdBu_r", vmin=0, vmax=1)
        plt.savefig(os.path.join(args.save_dir, f"{chrom}_heatmap.png"))
        plt.close()

    # Print summary
    print("\n" + "="*30)
    print("FINAL EVALUATION RESULTS")
    print("="*30)
    for chrom, m in out.items():
        print(f"\nResults for {chrom}:")
        print(f"  Pearson (Distance-Stratified): {m['pearson']:.4f}")
        print(f"  Spearman (Distance-Stratified): {m['spearman']:.4f}")
        print(f"  Insulation Score Correlation:   {m['insulation_corr']:.4f}")
        print(f"  Loop Detection AUROC:           {m['loop_auroc']:.4f}")

if __name__ == "__main__":
    main()