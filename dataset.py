"""PyTorch dataset aligning FASTA, Epiphany, and Hi-C data at 5 kb resolution."""
import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
from pyfaidx import Fasta
import torch
from torch.utils.data import Dataset

BIN_SIZE = 5000
EPIPHANY_BIN_SIZE = 100

if BIN_SIZE % EPIPHANY_BIN_SIZE != 0:
    raise ValueError("BIN_SIZE must be a multiple of EPIPHANY_BIN_SIZE")

BIN_FACTOR = BIN_SIZE // EPIPHANY_BIN_SIZE
_BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}

_NORMALIZATION_SUFFIX = {
    "KR": "KRnorm",
    "VC": "VCnorm",
    "SQRTVC": "SQRTVCnorm",
}


def _one_hot_encode(sequence: str) -> torch.FloatTensor:
    arr = np.zeros((4, len(sequence)), dtype=np.float32)
    for idx, base in enumerate(sequence.upper()):
        channel = _BASE_TO_INDEX.get(base)
        if channel is not None:
            arr[channel, idx] = 1.0
    return torch.from_numpy(arr)


def _preview_sequence(seq: torch.FloatTensor, length: int = 10) -> str:
    """Return the first `length` bases of a one-hot sequence tensor as a string."""
    arr = seq.detach().cpu().numpy()
    arr = arr[:, :length]
    bases: List[str] = []
    for column in arr.T:
        if column.sum() <= 0:
            bases.append("N")
            continue
        idx = int(column.argmax())
        bases.append("ACGT"[idx])
    return "".join(bases)


def _aggregate_epiphany(ds: h5py.Dataset, factor: int) -> np.ndarray:
    data = np.asarray(ds)

    channels, length = data.shape
    # trim 100bp blocks that can't form full 5kb bin 
    usable = (length // factor) * factor
    if usable == 0:
        return np.zeros((0, channels), dtype=np.float32)
    # reshape data from 100bp windows into 5kb bins
    chunked = data[:, :usable].reshape(channels, -1, factor) 
    # average across the windows making up a single 5kb bin
    pooled = chunked.mean(axis=2).transpose(1, 0)
    
    return pooled.astype(np.float32, copy=False) # [N_5kb, 5]


def _compute_valid_sequence_mask(fasta: Fasta, chrom: str, num_bins: int, bin_size: int) -> np.ndarray:
    if num_bins == 0:
        return np.zeros(0, dtype=bool)

    # Cut sequence data off at end of last 5kb bin
    segment = str(fasta[chrom][: num_bins * bin_size]).upper()
    mask = np.ones(num_bins, dtype=bool)
    for idx in range(num_bins):
        start = idx * bin_size
        end = start + bin_size
        if "N" in segment[start:end]:
            mask[idx] = False
    return mask


def _load_normalization_vector(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing Hi-C normalization vector: {path}")
    values = np.genfromtxt(path, dtype=np.float64)
    if values.size == 0:
        return np.zeros(0, dtype=np.float64)
    if np.isscalar(values):
        values = np.array([float(values)], dtype=np.float64)
    return values


def _load_contacts(
    path: Path,
    bin_size: int,
    max_bins: int,
    norm_values: Optional[np.ndarray] = None,
) -> Dict[int, Dict[int, float]]:
    contacts: Dict[int, Dict[int, float]] = defaultdict(dict)
    if not path.exists():
        raise FileNotFoundError(f"Missing Hi-C contact file: {path}")

    norm_length = 0 if norm_values is None else int(norm_values.shape[0])

    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            a_raw, b_raw, value_raw = parts
            try:
                a = int(float(a_raw))
                b = int(float(b_raw))
                value = float(value_raw)
            except ValueError:
                continue
            if not math.isfinite(value) or value <= 0:
                continue
            i = a // bin_size
            j = b // bin_size
            if i == j:
                continue
            if i >= max_bins or j >= max_bins:
                continue
            if norm_values is not None:
                if i >= norm_length or j >= norm_length:
                    continue
                # find norm value for anchor and neighbor 
                norm_i = float(norm_values[i])
                norm_j = float(norm_values[j])
                # ignore NaNs and -ve's
                if not math.isfinite(norm_i) or not math.isfinite(norm_j):
                    continue
                if norm_i <= 0.0 or norm_j <= 0.0:
                    continue
                # normalise raw contact count 
                value = value / (norm_i * norm_j)
            current = contacts[i].get(j)
            if current is None or value > current:
                contacts[i][j] = value
                contacts[j][i] = value

    return {int(anchor): {int(neigh): float(val) for neigh, val in nbrs.items()} for anchor, nbrs in contacts.items()}


def _build_positive_edges(
    contacts: Dict[int, Dict[int, float]],
    valid_mask: np.ndarray,
    threshold: float,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    positives: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    limit = valid_mask.shape[0]
    for anchor, nbrs in contacts.items():
        if anchor >= limit or not valid_mask[anchor]:
            continue
        selected: List[Tuple[int, float]] = []
        for neighbor, value in nbrs.items():
            if neighbor >= limit or not valid_mask[neighbor]:
                continue
            if value >= threshold:
                selected.append((neighbor, value))
        if not selected:
            continue
        selected.sort(key=lambda item: item[0])
        neighbors = np.array([item[0] for item in selected], dtype=np.int64)
        strengths = np.array([item[1] for item in selected], dtype=np.float32)
        positives[int(anchor)] = (neighbors, strengths)
    return positives


class GenomicsTripletDataset(Dataset):
    """Triplet dataset aligning sequence, epigenomic, and Hi-C data at 5 kb resolution."""

    def __init__(
        self,
        fasta_path: str,
        epiphany_path: str,
        hic_root: str,
        chroms: Optional[Iterable[str]] = None,
        contact_threshold: float = 5.0,
        hic_mapq: str = "MAPQGE30", # picked over MAPQG0 because MAPQ>30 indicates higher confidence the read aligns 
        hic_field: str = "RAWobserved",
        hic_norm: str = "none",
        bin_size: int = BIN_SIZE,
        epiphany_bin_size: int = EPIPHANY_BIN_SIZE,
        negative_attempts: int = 32,
        random_seed: Optional[int] = None,
    ) -> None:
        if bin_size % epiphany_bin_size != 0:
            raise ValueError("bin_size must be divisible by epiphany_bin_size")

        self.fasta_path = Path(fasta_path)
        self.epiphany_path = Path(epiphany_path)
        self.hic_root = Path(hic_root)
        self.hic_mapq = hic_mapq
        self.hic_field = hic_field
        norm_mode = (hic_norm or "none").upper()
        if norm_mode != "NONE" and norm_mode not in _NORMALIZATION_SUFFIX:
            raise ValueError(f"Unsupported Hi-C normalization mode: {hic_norm}")
        self.hic_norm = norm_mode
        self.bin_size = int(bin_size)
        self.epiphany_bin_size = int(epiphany_bin_size)
        self.contact_threshold = float(contact_threshold)
        self.negative_attempts = max(1, int(negative_attempts))
        self.bin_factor = self.bin_size // self.epiphany_bin_size
        self.rng = np.random.default_rng(random_seed)

        self._fa: Optional[Fasta] = None
        self._fa = Fasta(str(self.fasta_path), as_raw=True)

        self.chrom_data: Dict[str, Dict[str, object]] = {}
        self.anchor_lookup: List[Tuple[str, int]] = []

        resolution_dir = self.hic_root / f"{self.bin_size // 1000}kb_resolution_intrachromosomal"

        with h5py.File(self.epiphany_path, "r") as h5:
            # load chr's from the sequence and epigenomic datasets
            epi_chroms = {key for key in h5.keys() if key.startswith("chr")}
            fasta_chroms = set(self._fa.keys())
            if chroms is None:
                target_chroms = sorted(epi_chroms & fasta_chroms)
            else:
                target_chroms = [c for c in chroms if c in epi_chroms and c in fasta_chroms]

            if not target_chroms:
                raise ValueError("No chromosomes available across all datasets")

            for chrom in target_chroms:
                contact_file = (
                    resolution_dir
                    / chrom
                    / self.hic_mapq
                    / f"{chrom}_{self.bin_size // 1000}kb.{self.hic_field}"
                )
                if not contact_file.exists():
                    continue

                # Aggregate 100 bp windows into single value for a 5kb bin
                features = _aggregate_epiphany(h5[chrom], self.bin_factor) # [N_5kb, 5]
                seq_bins = len(self._fa[chrom]) // self.bin_size
                norm_values_full: Optional[np.ndarray] = None
                if self.hic_norm != "NONE":
                    norm_suffix = _NORMALIZATION_SUFFIX[self.hic_norm]
                    norm_file = (
                        resolution_dir
                        / chrom
                        / self.hic_mapq
                        / f"{chrom}_{self.bin_size // 1000}kb.{norm_suffix}"
                    )
                    if not norm_file.exists():
                        continue
                    # get norm value for each bin
                    norm_values_full = _load_normalization_vector(norm_file) # [N_5kb] 
                max_bins = min(seq_bins, features.shape[0])
                if norm_values_full is not None:
                    if norm_values_full.size == 0:
                        continue
                    max_bins = min(max_bins, int(norm_values_full.shape[0]))
                if max_bins == 0:
                    continue
                # take features up until we run out of corresponding sequence data
                features = features[:max_bins]
                # Return binary mask, masking 5kb sequence bins containing N's
                valid_mask = _compute_valid_sequence_mask(self._fa, chrom, max_bins, self.bin_size)
                norm_values = None
                if norm_values_full is not None:
                    # clip norm values so we don't have more values than we have bins
                    norm_values = norm_values_full[:max_bins]
                    finite_mask = np.isfinite(norm_values) & (norm_values > 0.0)
                    valid_mask &= finite_mask
                # contacts: Anchor index --> neighbor index --> contact score (Raw)
                # each index points to a 5kb bin
                contacts = _load_contacts(contact_file, self.bin_size, max_bins, norm_values)
                # filter contacts map down to those that correspond to valid sequences (via valid_mask) and those that are above the "positive" threshold 
                positives = _build_positive_edges(contacts, valid_mask, self.contact_threshold)
                if not positives:
                    continue

                anchors = sorted(positives.keys())
                valid_indices = np.flatnonzero(valid_mask).astype(np.int64)
                if len(valid_indices) == 0:
                    continue

                chrom_info = {
                    "features": features, # [N_5kb, 5] 5 epigenomic tracks 
                    "valid_mask": valid_mask, # [N_5kb] binary mask 
                    "valid_indices": valid_indices, # list of valid 5kb bin indices 
                    "positives": positives, # map: anchor_i --> neigh_i --> score (for positive pairs only)
                    "contacts": contacts, # map: anchor_i --> neigh_i --> score 
                    "norm_values": norm_values,
                }
                self.chrom_data[chrom] = chrom_info
                # populate anchor_lookup: List[(chr_n, anchor index), (chr_n, anchor_index), ...]
                for anchor in anchors:
                    self.anchor_lookup.append((chrom, int(anchor)))

        if not self.anchor_lookup:
            raise RuntimeError("No valid anchors with positive Hi-C contacts were found")

    def __len__(self) -> int:
        return len(self.anchor_lookup)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        # for given inx get index of anchor (5kb sequence) and the chr it belongs to 
        chrom, anchor = self.anchor_lookup[idx]
        # get features, contacts, etc
        info = self.chrom_data[chrom]
        positives = info["positives"]  
        neighbors, strengths = positives[anchor]  
        pos_idx = int(self.rng.integers(len(neighbors)))
        pos_bin = int(neighbors[pos_idx])
        pos_strength = float(strengths[pos_idx])

        # randomly find a valid bin whose contact score with the anchor is below the threshold 
        neg_bin = self._sample_negative(info, anchor, neighbors)
        contacts = info["contacts"]  
        neg_strength = float(contacts.get(anchor, {}).get(neg_bin, 0.0))

        anchor_seq = self._fetch_sequence(chrom, anchor)
        pos_seq = self._fetch_sequence(chrom, pos_bin)
        neg_seq = self._fetch_sequence(chrom, neg_bin)

        features = info["features"]  
        anchor_feat = torch.tensor(features[anchor], dtype=torch.float32)
        pos_feat = torch.tensor(features[pos_bin], dtype=torch.float32)
        neg_feat = torch.tensor(features[neg_bin], dtype=torch.float32)

        return {
            "seq": anchor_seq,
            "feat": anchor_feat,
            "pos_seq": pos_seq,
            "pos_feat": pos_feat,
            "neg_seq": neg_seq,
            "neg_feat": neg_feat,
            "chrom": chrom,
            "anchor_index": anchor,
            "pos_index": pos_bin,
            "neg_index": neg_bin,
            "pos_contact_strength": pos_strength,
            "neg_contact_strength": neg_strength,
        }

    def _get_fasta(self) -> Fasta:
        if self._fa is None:
            self._fa = Fasta(str(self.fasta_path), as_raw=True)
        return self._fa

    def _fetch_sequence(self, chrom: str, bin_index: int) -> torch.FloatTensor:
        fasta = self._get_fasta()
        start = bin_index * self.bin_size
        end = start + self.bin_size
        seq = str(fasta[chrom][start:end])
        return _one_hot_encode(seq)

    def _sample_negative(self, info: Dict[str, object], anchor: int, positive_neighbors: np.ndarray) -> int:
        valid_indices = info["valid_indices"]  
        contacts = info["contacts"]  
        positives_set = set(int(x) for x in positive_neighbors.tolist())
        # sample random indices negative_attempts # of times and see if its a valid negative partner for the given anchor bin
        for _ in range(self.negative_attempts):
            candidate = int(self.rng.choice(valid_indices))
            if candidate == anchor or candidate in positives_set:
                continue
            if contacts.get(anchor, {}).get(candidate, 0.0) >= self.contact_threshold:
                continue
            return candidate
        # Deterministic safety net: if our random guesses above fail we just work through all possible valid_indices 
        for candidate in valid_indices:
            candidate = int(candidate)
            if candidate == anchor or candidate in positives_set:
                continue
            if contacts.get(anchor, {}).get(candidate, 0.0) >= self.contact_threshold:
                continue
            return candidate
        raise RuntimeError(f"Unable to sample a negative example for anchor {anchor}")

    def __getstate__(self) -> Dict[str, object]:
        state = self.__dict__.copy()
        state["_fa"] = None
        return state

    def __setstate__(self, state: Dict[str, object]) -> None:
        self.__dict__.update(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a single triplet sample")
    parser.add_argument('--fasta', default='data/hg38/hg38.fa', help='Path to hg38 FASTA file')
    parser.add_argument('--epiphany', default='data/epiphany/GM12878_X.h5', help='Path to Epiphany H5 file')
    parser.add_argument('--hic-root', default='data/hic/GM12878_primary', help='Root directory containing Hi-C contact files')
    parser.add_argument('--chroms', nargs='+', help='Optional list of chromosomes to load, e.g. chr1 chr2')
    parser.add_argument('--contact-threshold', type=float, default=5.0, help='Hi-C contact threshold for positives')
    parser.add_argument('--hic-mapq', default='MAPQGE30', help='Hi-C MAPQ directory to use')
    parser.add_argument('--hic-field', default='RAWobserved', help='Hi-C field suffix to read (matching file name)')
    parser.add_argument('--hic-norm', choices=['none', 'KR', 'VC', 'SQRTVC'], default='none', help='Hi-C normalization vector to apply to raw contacts')
    parser.add_argument('--index', type=int, default=0, help='Dataset index to inspect')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for sampling positives/negatives')
    args = parser.parse_args()

    chroms = args.chroms if args.chroms else None
    dataset = GenomicsTripletDataset(
        fasta_path=args.fasta,
        epiphany_path=args.epiphany,
        hic_root=args.hic_root,
        chroms=chroms,
        contact_threshold=args.contact_threshold,
        hic_mapq=args.hic_mapq,
        hic_field=args.hic_field,
        hic_norm=args.hic_norm,
        random_seed=args.seed,
    )

    if len(dataset) == 0:
        raise RuntimeError('Dataset contains no samples with current configuration')

    index = args.index % len(dataset)
    sample = dataset[index]

    print(f'Dataset length: {len(dataset)}')
    print(f'Inspecting index {index}')
    print(f"Chromosome: {sample['chrom']} | anchor: {sample['anchor_index']} | positive: {sample['pos_index']} | negative: {sample['neg_index']}")
    print(f"Hi-C contact strength (anchor→positive): {sample['pos_contact_strength']:.4f}")
    print(f"Hi-C contact strength (anchor→negative): {sample['neg_contact_strength']:.4f}")

    for key, label in [('seq', 'Anchor'), ('pos_seq', 'Positive'), ('neg_seq', 'Negative')]:
        seq_str = _preview_sequence(sample[key], length=10)
        print(f"{label} sequence first 10 bp: {seq_str}")

    print('Anchor features:', [float(x) for x in sample['feat'].tolist()])
    print('Positive features:', [float(x) for x in sample['pos_feat'].tolist()])
    print('Negative features:', [float(x) for x in sample['neg_feat'].tolist()])


if __name__ == '__main__':
    main()
