import torch
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from dataset import GenomicsContrastiveDataset

# --- Configuration ---
CACHE_DIR = None
CHROM = "chr1"  # TODO: change as desired
WINDOW_SIZE = 500
NUM_BATCHES = 100  # TODO: change as desired


def dict_to_dense(contact_dict, start, end):
    """
    Efficiently rebuilds a dense numpy matrix from the dataset's sparse dictionary
    just for the specific window we are looking at.
    """
    size = end - start
    mat = np.zeros((size, size), dtype=np.float32)
    for i in range(start, end):
        if i in contact_dict:
            for j, val in contact_dict[i].items():
                if start <= j < end:
                    mat[i - start, j - start] = val
                    mat[j - start, i - start] = val
    return mat


def calculate_structural_score(matrix):
    # data cleaning: remove 0 / nan
    valid_mask = np.isfinite(matrix) & (matrix > 0)
    if valid_mask.sum() / matrix.size < 0.5:
        return 0  # Too sparse

    # log transform
    log_mat = np.log1p(matrix)

    # overall variance
    contrast = np.std(log_mat)

    # diagonality vs. dropoff
    diag_mean = np.mean(np.diag(log_mat, k=1))
    off_diag_mean = np.mean(np.diag(log_mat, k=100))
    decay_score = diag_mean / (off_diag_mean + 1e-6)

    return contrast * decay_score


def main():
    if CACHE_DIR:
        print(f"Loading dataset from {CACHE_DIR}...")
        ds = GenomicsContrastiveDataset.from_cache_dir(CACHE_DIR)
    else:
        ds = GenomicsContrastiveDataset.from_cache_dir(None, None, None)

    if CHROM not in ds.chrom_data:
        print(f"Error: {CHROM} not in dataset. Available: {list(ds.chrom_data.keys())}")
        return

    print(f"Scanning first {NUM_BATCHES} chunks of {CHROM}...")

    contacts_dict = ds.chrom_data[CHROM]["contacts"]
    seq_tensor = ds.seq_tokens_all[CHROM]
    epi_tensor = ds.epi_tokens_all[CHROM]

    best_score = -1
    best_data = None

    step = WINDOW_SIZE // 2  # gives overlap

    for i in tqdm(range(NUM_BATCHES)):
        start = i * step
        end = start + WINDOW_SIZE

        # Stop if we go out of bounds
        if end > len(seq_tensor):
            break

        # reconstruct matrix
        mat = dict_to_dense(contacts_dict, start, end)

        # skip empty matrices
        if np.sum(mat) == 0:
            continue

        # get score
        score = calculate_structural_score(mat)

        if score > best_score:
            best_score = score

            # extract inputs and clean them
            best_data = {
                "hic": mat,
                "seq": seq_tensor[start:end].clone().cpu().numpy(),
                "epi": epi_tensor[start:end].clone().cpu().numpy(),
                "coords": (CHROM, start, end),
            }
            print(f"Found better candidate at batch {i} (Score: {score:.4f})")

    if best_data:
        outfile = "illustrative_example.npz"
        np.savez(outfile, **best_data)
        print(f"output saved to {outfile}")
        print(f"Coordinates: {best_data['coords']}")
    else:
        print("No valid data found in the first 100 batches.")


if __name__ == "__main__":
    main()
