import torch
import numpy as np
import os
import argparse
import scipy.stats as ss
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# TODO: import model and use for evaluations
# TODO: add integration with dataset


def pairwise_distances(emb, metric="cosine"):
    # normalize for cosine similarity
    if metric == "cosine":
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        sim = emb @ emb.T
        dist = 1 - sim
    else:
        # euclidean fallback
        from scipy.spatial.distance import pdist, squareform

        dist = squareform(pdist(emb, metric="euclidean"))

    return dist


def distance_to_contact(dist, mode="inv_sq"):
    if mode == "inv_sq":
        return 1.0 / (dist**2 + 1e-6)
    elif mode == "exp":
        return np.exp(-dist)
    else:
        raise ValueError("Unknown distance→contact mapping")


def observed_expected(mat):
    N = mat.shape[0]
    mat = mat.copy()
    expected = np.zeros_like(mat)

    # calculate expected value for every diagonal offset k
    for k in range(N):
        diag = np.diag(mat, k=k)
        if len(diag) > 0:
            expected_val = np.nanmean(diag)
            expected += np.diag(np.full(len(diag), expected_val), k=k)
            if k > 0:
                expected += np.diag(np.full(len(diag), expected_val), k=-k)

    oe = mat / (expected + 1e-6)
    return oe


def stratified_correlations(pred_oe, true_oe):
    """
    Compute Pearson ($r$) and Spearman ($\rho$) correlations per genomic distance.
    """
    N = pred_oe.shape[0]
    pearsons = []
    spearmans = []

    # iterate over diagonals (genomic distances)
    for d in range(1, min(N, 1000)):
        p_diag = np.diag(pred_oe, k=d)
        t_diag = np.diag(true_oe, k=d)

        # mask NaNs and Infs
        mask = np.isfinite(p_diag) & np.isfinite(t_diag)

        pearsons.append(ss.pearsonr(p_diag[mask], t_diag[mask])[0])
        spearmans.append(ss.spearmanr(p_diag[mask], t_diag[mask])[0])

    return pearsons, spearmans


def compute_insulation_score(mat, window=20):
    """
    Compute insulation score to identify TAD boundaries
    """
    N = mat.shape[0]
    ins = np.full(N, np.nan)

    for i in range(window, N - window):
        sub = mat[i - window : i, i : i + window]
        ins[i] = np.nanmean(sub)

    # normalize (log2 ratio)
    valid_mask = np.isfinite(ins)
    ins_valid = ins[valid_mask]

    if len(ins_valid) == 0:
        return ins

    smooth = ss.uniform_filter1d(ins_valid, size=window)
    ins[valid_mask] = np.log2((ins_valid + 1e-8) / (smooth + 1e-8))

    return ins


def detect_loops_heuristic(mat, threshold=2.0):
    """
    Heuristic local-max loop calling to approximate HiCCUPS for evaluation.
    """
    N = mat.shape[0]
    loops = []

    # Heuristic: Look for peaks in O/E map
    min_dist = 4
    max_dist = 400

    for i in range(0, N - min_dist):
        for j in range(i + min_dist, min(N, i + max_dist)):
            patch = mat[i - 1 : i + 2, j - 1 : j + 2]
            center = mat[i, j]

            if not np.isfinite(center):
                continue

            if center > np.nanmax(patch) - 1e-8 and center > threshold:
                loops.append((i, j))

    return set(loops)


def loop_auroc(pred_matrix, true_loops_set, N):
    """
    Compute AUROC for loop detection.
    """
    y_true = np.zeros((N, N), dtype=np.int8)
    for r, c in true_loops_set:
        y_true[r, c] = 1

    # flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = pred_matrix.flatten()

    # handle NaNs in prediction
    mask = np.isfinite(y_pred_flat)

    try:
        score = roc_auc_score(y_true_flat[mask], y_pred_flat[mask])
    except ValueError:
        score = 0.5  # edge case of only one class present

    return score


def evaluate_chromosome(model, dataset, chrom):
    """
    Full pipeline:
    1. Inference -> Embeddings
    2. Distance -> Contact Map ($P_{ij}$)
    3. O/E Normalization
    4. Metrics: Pearson/Spearman (Stratified), Insulation, Loop AUROC
    """
    # TODO: set size appropriately
    size = None

    emb_list = []
    gt_list = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(size), desc="Inference"):
            # TODO: Get X, y from dataset
            # X, y = None, None

            # TODO: Forward pass
            # emb = model(X)

            emb_list.append(emb.squeeze())
            gt_list.append(y_vec.squeeze())

    emb_full = np.vstack(emb_list)  # (N, d)
    gt_full = np.vstack(gt_list)  # (N, N)

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

    # 4. Insulation Scores (TODO: adjust window as needed)
    print("Computing insulation...")
    pred_ins = compute_insulation_score(pred, window=20)
    gt_ins = compute_insulation_score(gt_full, window=20)

    # filter NaNs for correlation
    mask_ins = np.isfinite(pred_ins) & np.isfinite(gt_ins)
    if mask_ins.sum() > 0:
        ins_corr = ss.pearsonr(pred_ins[mask_ins], gt_ins[mask_ins])[0]
    else:
        ins_corr = 0.0

    # 5. Loops
    print("Computing loops...")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--save_dir", default="./eval_results")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # evaluating final performance
    # TODO: change evaluation chromosomes as needed
    test_chroms = ["chr20", "chr21", "chr22"]

    # TODO: insert dataset and model
    dataset = None
    model = None

    out = {}

    for chrom in test_chroms:
        metrics = evaluate_chromosome(model, dataset, chrom)
        out[chrom] = metrics

    print("FINAL EVALUATION RESULTS")
    for chrom, m in out.items():
        print(f"\nResults for {chrom}:")
        print(f"Pearson (Distance-Stratified): {m['pearson']:.4f}")
        print(f"Spearman (Distance-Stratified): {m['spearman']:.4f}")
        print(f"Insulation Score Correlation: {m['insulation_corr']:.4f}")
        print(f"Loop Detection AUROC: {m['loop_auroc']:.4f}")


if __name__ == "__main__":
    main()
