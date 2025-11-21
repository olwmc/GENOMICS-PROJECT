"""PyTorch dataset aligning FASTA, Epiphany, and Hi-C data at 5 kb resolution."""

import argparse
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from pyfaidx import Fasta
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

BIN_SIZE = 5000
EPIPHANY_BIN_SIZE = 100
SEQ_DTYPE = torch.float16  # change to float32 if you prefer exact previous dtype
EPI_DTYPE = torch.float16  # change to float32 to match old behavior exactly
PRECOMPUTE_SEQ = False     # precompute sequence tokens (set True for faster access, False for smaller cache/footprint)

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


def _compute_valid_sequence_mask(fasta: Fasta, chrom: str, num_bins: int, bin_size: int) -> np.ndarray:
    if num_bins == 0:
        return np.zeros(0, dtype=bool)

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
            # genomic coord for first locus, second locus, observed contact score 
            a_raw, b_raw, value_raw = parts
            try:
                a = int(float(a_raw))
                b = int(float(b_raw))
                value = float(value_raw)
            except ValueError:
                continue
            if not math.isfinite(value) or value <= 0:
                continue
            # convert genomic coords to integers (one bin every 5000 bp)
            i = a // bin_size
            j = b // bin_size
            if i == j:
                continue
            if i >= max_bins or j >= max_bins:
                continue
            # if we have a normalisation method specified
            if norm_values is not None:
                if i >= norm_length or j >= norm_length:
                    continue
                norm_i = float(norm_values[i])
                norm_j = float(norm_values[j])
                if not math.isfinite(norm_i) or not math.isfinite(norm_j):
                    continue
                if norm_i <= 0.0 or norm_j <= 0.0:
                    continue
                # normalise the contact values
                value = value / (norm_i * norm_j)
            current = contacts[i].get(j)
            if current is None or value > current:
                contacts[i][j] = value
                contacts[j][i] = value

    return {int(anchor): {int(neigh): float(val) for neigh, val in nbrs.items()} for anchor, nbrs in contacts.items()}


def _smooth_expectation(curve: Dict[int, float], window: int = 2) -> Dict[int, float]:
    if not curve:
        return {}
    sorted_keys = sorted(curve.keys())
    smoothed: Dict[int, float] = {}
    values = [curve[k] for k in sorted_keys]
    for idx, key in enumerate(sorted_keys):
        start = max(0, idx - window)
        end = min(len(sorted_keys), idx + window + 1)
        smoothed[key] = float(np.mean(values[start:end]))
    return smoothed


class GenomicsContrastiveDataset(Dataset):
    """Contrastive dataset aligning sequence, epigenomic, and Hi-C data at 5 kb resolution."""

    def __init__(
        self,
        fasta_path: str,
        epiphany_path: str,
        hic_root: str,
        chroms: Optional[Iterable[str]] = None,
        contact_threshold: Optional[float] = None,
        hic_mapq: str = "MAPQGE30",
        hic_field: str = "RAWobserved",
        hic_norm: str = "none",
        mode: str = "pair",
        bin_edges: Optional[Sequence[int]] = None,
        min_distance_bp: int = 25_000,
        max_distance_bp: Optional[int] = None,
        oe_metric: str = "oe",
        pos_quantile: float = 0.8,
        neg_quantile: float = 0.2,
        num_negatives: int = 8,
        bin_schedule: str = "roundrobin",
        pairs_per_batch: int = 64,
        patch_size_bp: int = 100,
        emit_pos_ids: bool = False,
        epiphany_bin_size: int = EPIPHANY_BIN_SIZE,
        negative_attempts: int = 32,
        random_seed: Optional[int] = None,
    ) -> None:
        if contact_threshold is not None:
            warnings.warn("Deprecated; ignored in favor of quantiles.", RuntimeWarning, stacklevel=2)

        mode = mode.lower()
        if mode not in {"pair", "triplet"}:
            raise ValueError("mode must be 'pair' or 'triplet'")
        self.mode = mode

        self.emit_pos_ids = bool(emit_pos_ids)

        if patch_size_bp != epiphany_bin_size:
            raise ValueError("--patch-size-bp must equal the Epiphany bin size (100 bp) to keep 50x100bp patches")
        if BIN_SIZE % patch_size_bp != 0:
            raise ValueError("--patch-size-bp must divide 5000 bp locus")

        self.patch_size_bp = int(patch_size_bp)
        self.tokens_per_locus = BIN_SIZE // self.patch_size_bp
        assert self.tokens_per_locus == BIN_SIZE // self.patch_size_bp, "Token count mismatch for 5 kb locus"

        oe_metric = oe_metric.lower()
        if oe_metric not in {"oe", "logresid"}:
            raise ValueError("--oe-metric must be 'oe' or 'logresid'")
        self.oe_metric = oe_metric

        self.num_negatives = int(num_negatives) if mode == "pair" else 1
        if self.num_negatives <= 0:
            raise ValueError("--num-negatives must be positive")

        self.pos_quantile = float(pos_quantile)
        self.neg_quantile = float(neg_quantile)
        if not 0.0 < self.pos_quantile < 1.0 or not 0.0 < self.neg_quantile < 1.0:
            raise ValueError("Quantiles must be in (0,1)")

        if self.neg_quantile > self.pos_quantile:
            warnings.warn(
                "neg-quantile > pos-quantile; pools may not overlap as expected.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.min_distance_bp = int(min_distance_bp)
        self.max_distance_bp = None if max_distance_bp is None else int(max_distance_bp)

        bin_edges_list = list(bin_edges) if bin_edges is not None else [25_000, 100_000, 400_000, 1_000_000, 10_000_000]
        if not bin_edges_list:
            raise ValueError("--bin-edges must contain at least one value")
        if sorted(bin_edges_list) != list(bin_edges_list):
            raise ValueError("--bin-edges must be sorted ascending")
        self.bin_edges = [int(x) for x in bin_edges_list]

        norm_mode = (hic_norm or "none").upper()
        if norm_mode != "NONE" and norm_mode not in _NORMALIZATION_SUFFIX:
            raise ValueError(f"Unsupported Hi-C normalization mode: {hic_norm}")
        self.hic_norm = norm_mode

        self.fasta_path = Path(fasta_path)
        self.epiphany_path = Path(epiphany_path)
        self.hic_root = Path(hic_root)
        self.hic_mapq = hic_mapq
        self.hic_field = hic_field

        self.bin_size = BIN_SIZE
        self.epiphany_bin_size = int(epiphany_bin_size)
        self.epi_windows_per_bin = self.bin_size // self.epiphany_bin_size
        self.windows_per_token = self.patch_size_bp // self.epiphany_bin_size

        self.bin_schedule = bin_schedule
        self.pairs_per_batch = int(pairs_per_batch)

        self.negative_attempts = max(1, int(negative_attempts))
        self.anchor_retry_limit = max(4, self.negative_attempts // 2)
        self.rng = np.random.default_rng(random_seed)

        self._fa: Optional[Fasta] = None
        self._fa = Fasta(str(self.fasta_path), as_raw=True)

        self.chrom_data = {}
        self.pos_pool = {}
        self.neg_pool = {}
        self.bin_thresholds = {}
        self.bin_ranges = {}
        self.sample_entries = []
        self.bin_to_indices = defaultdict(list)
        self.bin_anchor_lookup = defaultdict(list)
        self.bin_neighbors = {}

        # precomputed tensors per chromosome (filled in _prepare_chromosome)
        # Each entry shape by chrom:
        #   seq_tokens_all: (num_bins, tokens_per_locus, 4, patch_size_bp)
        #   epi_tokens_all: (num_bins, tokens_per_locus, epi_channels)
        self.seq_tokens_all = {}
        self.epi_tokens_all = {}

        resolution_dir = self.hic_root / f"{self.bin_size // 1000}kb_resolution_intrachromosomal"

        with h5py.File(self.epiphany_path, "r") as h5:
            # extract chromosomes from the epiphany and fasta datasets as specified by chroms arg (all chroms (1-22) if None specified)
            epi_chroms = {key for key in h5.keys() if key.startswith("chr")}
            fasta_chroms = set(self._fa.keys())
            if chroms is None:
                target_chroms = sorted(epi_chroms & fasta_chroms)
            else:
                target_chroms = [c for c in chroms if c in epi_chroms and c in fasta_chroms]

            if not target_chroms:
                raise ValueError("No chromosomes available across all datasets")

            total_chroms = len(target_chroms)
            print(f"[dataset] Preparing {total_chroms} chromosome(s) ...")
            for chrom in tqdm(target_chroms, desc="chroms", unit="chrom"):
                contact_file = (
                    resolution_dir
                    / chrom
                    / self.hic_mapq
                    / f"{chrom}_{self.bin_size // 1000}kb.{self.hic_field}"
                )
                if not contact_file.exists():
                    continue

                # for the given chromosome get entire epiphany data (5, 2489564)
                epiphany_raw = np.asarray(h5[chrom], dtype=np.float32)

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
                    norm_values_full = _load_normalization_vector(norm_file)

                max_bins = min(seq_bins, epiphany_raw.shape[1] // self.epi_windows_per_bin)
                # make sure we have normalisation values for each 5kb bin
                if norm_values_full is not None:
                    if norm_values_full.size == 0:
                        continue
                    max_bins = min(max_bins, int(norm_values_full.shape[0]))
                if max_bins == 0:
                    continue
                
                # trim raw 100 bp epiphany values 
                epiphany_raw = epiphany_raw[:, : max_bins * self.epi_windows_per_bin]

                # mask out N's in the FASTA data
                valid_mask = _compute_valid_sequence_mask(self._fa, chrom, max_bins, self.bin_size)
                if valid_mask.size == 0:
                    continue

                norm_values = None
                if norm_values_full is not None:
                    norm_values = norm_values_full[:max_bins]
                    finite_mask = np.isfinite(norm_values) & (norm_values > 0.0)
                    valid_mask &= finite_mask

                # load and normalise Hi-C data for given chr 
                # anchor: neighbor: value (nested dict)
                contacts = _load_contacts(contact_file, self.bin_size, max_bins, norm_values)
                chrom_info = self._prepare_chromosome(chrom, contacts, valid_mask, epiphany_raw)
                if chrom_info is None:
                    continue
                self.chrom_data[chrom] = chrom_info

        self._finalize_dataset()


        if not self.sample_entries:
            raise RuntimeError(
                "No anchors expose both positive and negative pools; consider relaxing --pos-quantile/--neg-quantile or distance filters."
            )
        

        for bin_id, (lower, upper) in self._compute_bin_ranges().items():
            self.bin_ranges[bin_id] = (lower, upper)

    def _prepare_chromosome(
        self,
        chrom: str,
        contacts: Dict[int, Dict[int, float]],
        valid_mask: np.ndarray,
        epiphany_raw: np.ndarray,
    ) -> Optional[Dict[str, np.ndarray]]:
        valid_mask = valid_mask.astype(bool, copy=False)
        valid_indices = np.flatnonzero(valid_mask).astype(np.int64)
        if valid_indices.size == 0:
            return None

        distance_contacts: Dict[int, List[float]] = defaultdict(list)
        distance_logs: Dict[int, List[float]] = defaultdict(list)
        pair_records: List[Tuple[int, int, float, int]] = []


        for anchor, nbrs in contacts.items():
            # validity check: if anchor is out of scope or invalid sequence (infinite values or N's) skip  
            if anchor >= valid_mask.size or not valid_mask[anchor]:
                continue
            for neighbor, value in nbrs.items():
                # so we don't double count 
                if neighbor <= anchor:
                    continue
                # same validity check on neighbor 
                if neighbor >= valid_mask.size or not valid_mask[neighbor]:
                    continue
                distance_bins = abs(neighbor - anchor)
                distance_bp = distance_bins * self.bin_size
                # skip contact scores between bins closer than 25 kb so close range contacts don't dominate the data 
                if distance_bp < self.min_distance_bp:
                    continue
                if self.max_distance_bp is not None and distance_bp > self.max_distance_bp:
                    continue
                # maps difference in genomic coord index to contact value --> allowing for stratified distance bins down the line 
                distance_contacts[distance_bins].append(value)
                distance_logs[distance_bins].append(math.log1p(value))
                pair_records.append((anchor, neighbor, value, distance_bp))

        if not pair_records:
            return None

        # distance_contacts: range of interaction --> contact values 
        # get expected distance for the each given distance (between two bins)
        expectation = {dist: float(np.mean(values)) for dist, values in distance_contacts.items() if values}
        expectation = _smooth_expectation(expectation)
        mean_logs = {dist: float(np.mean(values)) for dist, values in distance_logs.items() if values}

        per_bin_metrics: Dict[Tuple[str, int], List[float]] = defaultdict(list)
        filtered_records: List[Tuple[int, int, float, int]] = []
        # compute filtered records: distance normalised contact scores
        for anchor, neighbor, contact_value, distance_bp in pair_records:
            dist_bins = abs(neighbor - anchor)
            metric: Optional[float] = None
            # compute observed over expected to normalise for distance 
            if self.oe_metric == "oe":
                expected = expectation.get(dist_bins)
                if expected is None or expected <= 0.0 or not math.isfinite(expected):
                    continue
                metric = contact_value / expected
            else:
                mean_log = mean_logs.get(dist_bins)
                if mean_log is None or not math.isfinite(mean_log):
                    continue
                metric = math.log1p(contact_value) - mean_log
            if metric is None or not math.isfinite(metric):
                continue
            # get id of one of the stratified bins for given genomic separation
            # i.e how far away are the anchor and neighbor? 25kb, 100kb, 250kb,...
            bin_id = self._distance_to_bin(distance_bp)
            if bin_id is None:
                continue
            # for given chr and genomic separation: [distance normalised contact scores]
            per_bin_metrics[(chrom, bin_id)].append(metric)
            filtered_records.append((anchor, neighbor, metric, distance_bp))

        if not filtered_records:
            return None

        for (chrom_key, bin_id), metrics in per_bin_metrics.items():
            if not metrics:
                continue
            arr = np.array(metrics, dtype=np.float32)
            # based on distance normalised values, determine which contacts make up positive and negative samples 
            self.bin_thresholds[(chrom_key, bin_id)] = (
                float(np.quantile(arr, self.pos_quantile)),
                float(np.quantile(arr, self.neg_quantile)),
            )

        # go through distance normalised contact scores and add to positive and negative pools
        for anchor, neighbor, metric, distance_bp in filtered_records:
            bin_id = self._distance_to_bin(distance_bp)
            if bin_id is None:
                continue
            thresholds = self.bin_thresholds.get((chrom, bin_id))
            if thresholds is None:
                continue
            q_pos, q_neg = thresholds
            if metric >= q_pos:
                self._add_to_pool(self.pos_pool, chrom, anchor, bin_id, neighbor, metric, distance_bp)
                self._add_to_pool(self.pos_pool, chrom, neighbor, bin_id, anchor, metric, distance_bp)
            if metric <= q_neg:
                self._add_to_pool(self.neg_pool, chrom, anchor, bin_id, neighbor, metric, distance_bp)
                self._add_to_pool(self.neg_pool, chrom, neighbor, bin_id, anchor, metric, distance_bp)

        chrom_keys_present = any(key[0] == chrom for key in self.pos_pool.keys())
        if not chrom_keys_present:
            return None

        # Flatten per-bin 100 bp epigenomic signals (50 windows * C tracks) for similarity computations
        features_full = epiphany_raw.reshape(epiphany_raw.shape[0], -1, self.epi_windows_per_bin)  # (C, num_bins, 50)
        features_full = features_full.transpose(1, 2, 0).reshape(-1, self.epi_windows_per_bin * epiphany_raw.shape[0])  # (num_bins, 50*C)
        features_norm = features_full.copy()
        norms = np.linalg.norm(features_norm, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        features_norm /= norms

        # --- PRECOMPUTE TOKENS PER BIN (sequence + epigenomic) ---
        num_bins = int(valid_mask.shape[0])

        # Precompute sequence tokens (optional to save RAM)
        if PRECOMPUTE_SEQ:
            seq_tokens_list = []
            fasta = self._get_fasta()
            for b in range(num_bins):
                if not valid_mask[b]:
                    # fill with zeros to keep indexing consistent
                    seq_tokens_list.append(torch.zeros(self.tokens_per_locus, 4, self.patch_size_bp, dtype=SEQ_DTYPE))
                    continue
                start = b * self.bin_size
                end = start + self.bin_size
                seq = str(fasta[chrom][start:end])
                if len(seq) < self.bin_size:
                    seq = seq + "N" * (self.bin_size - len(seq))
                elif len(seq) > self.bin_size:
                    seq = seq[: self.bin_size]
                onehot = _one_hot_encode(seq).to(dtype=SEQ_DTYPE)  # (4, 5000)
                tok = onehot.view(4, self.tokens_per_locus, self.patch_size_bp).permute(1, 0, 2).contiguous()
                seq_tokens_list.append(tok)
            seq_tokens_all = torch.stack(seq_tokens_list, dim=0)  # (num_bins, T, 4, P)
            self.seq_tokens_all[chrom] = seq_tokens_all
        else:
            # we’ll fetch sequence on demand (old path)
            self.seq_tokens_all[chrom] = torch.empty(0)  # sentinel

        # Precompute epigenomic tokens
        epi = epiphany_raw  # (C, num_bins * epi_windows_per_bin)
        T = self.tokens_per_locus # 50 100bp tokens per locus/bin
        C = epi.shape[0] # num epigenomic tracks (5)
        epi_tokens_list = []
        for b in range(num_bins):
            # establish window for given bin 
            start = b * self.epi_windows_per_bin
            end = start + self.epi_windows_per_bin
            win = epi[:, start:end].reshape(C, T, self.windows_per_token)  # (C, 50, 1) with 100bp patches
            pooled = win.mean(axis=2, dtype=np.float32).transpose(1, 0)  # (T, C); mean is no-op when windows_per_token==1
            epi_tokens_list.append(torch.from_numpy(pooled.astype(np.float32, copy=False)).to(dtype=EPI_DTYPE))
        epi_tokens_all = torch.stack(epi_tokens_list, dim=0)  # (num_bins, T, C)
        self.epi_tokens_all[chrom] = epi_tokens_all

        return {
            "features": features_full.astype(np.float32, copy=False),
            "features_norm": features_norm.astype(np.float32, copy=False),
            "epiphany_raw": epiphany_raw.astype(np.float32, copy=False),
            "valid_mask": valid_mask,
            "valid_indices": valid_indices,
        }

    def _add_to_pool(
        self,
        pool: Dict[Tuple[str, int, int], Dict[str, np.ndarray]],
        chrom: str,
        anchor: int,
        bin_id: int,
        partner: int,
        metric: float,
        distance_bp: int,
    ) -> None:
        if anchor == partner:
            return
        key = (chrom, anchor, bin_id)
        entry = pool.get(key)
        if entry is None:
            entry = {
                "partners": [],
                "metrics": [],
                "distances": [],
            }
            pool[key] = entry
        entry["partners"].append(int(partner))
        entry["metrics"].append(float(metric))
        entry["distances"].append(int(distance_bp))

    def _compute_bin_ranges(self) -> Dict[int, Tuple[int, Optional[int]]]:
        ranges: Dict[int, Tuple[int, Optional[int]]] = {}
        for idx, edge in enumerate(self.bin_edges):
            if idx == len(self.bin_edges) - 1:
                ranges[idx] = (self.bin_edges[idx], None)
            else:
                ranges[idx] = (self.bin_edges[idx], self.bin_edges[idx + 1])
        return ranges

    def _distance_to_bin(self, distance_bp: int) -> Optional[int]:
        if distance_bp < self.bin_edges[0]:
            return None
        for idx in range(len(self.bin_edges) - 1):
            lower = self.bin_edges[idx]
            upper = self.bin_edges[idx + 1]
            if lower <= distance_bp < upper:
                return idx
        return len(self.bin_edges) - 1

    @staticmethod
    def _finalize_pool_entry(entry: Dict[str, object]) -> Dict[str, np.ndarray]:
        partners = entry.get("partners", [])
        metrics = entry.get("metrics", [])
        distances = entry.get("distances", [])
        if not isinstance(partners, np.ndarray):
            partners = np.array(partners, dtype=np.int64)
        if not isinstance(metrics, np.ndarray):
            metrics = np.array(metrics, dtype=np.float32)
        if not isinstance(distances, np.ndarray):
            distances = np.array(distances, dtype=np.int64)
        entry["partners"] = partners
        entry["metrics"] = metrics
        entry["distances"] = distances
        return entry

    def _finalize_dataset(self) -> None:
        self.bin_anchor_lookup = defaultdict(list)
        self.sample_entries: List[int] = []
        self.bin_to_indices = defaultdict(list)
        self.bin_neighbors = {}

        empty_counts: Dict[Tuple[str, int], int] = defaultdict(int)
        bin_to_anchors: Dict[int, List[Tuple[str, int]]] = defaultdict(list)

        # populate bin_to_anchors which maps distance bin --> chr, anchor 
        # pos_pool: [chr, anchor, bin_id] --> neighbors, OE scores, distances (within given bin_id)
        for key, pos_entry in list(self.pos_pool.items()):
            # for a given positive pair 
            chrom, anchor, bin_id = key
            pos_entry = self._finalize_pool_entry(pos_entry)
            self.pos_pool[key] = pos_entry

            neg_entry = self.neg_pool.get(key)
            if neg_entry is None:
                neg_entry = {"partners": [], "metrics": [], "distances": []}
            neg_entry = self._finalize_pool_entry(neg_entry)
            self.neg_pool[key] = neg_entry

            if pos_entry["partners"].size == 0:
                continue
            if neg_entry["partners"].size == 0:
                empty_counts[(chrom, bin_id)] += 1
                continue

            bin_to_anchors[bin_id].append((chrom, anchor))

        for (chrom, bin_id), count in empty_counts.items():
            warnings.warn(
                f"{count} anchors in ({chrom}, bin {bin_id}) have empty negative pools.",
                RuntimeWarning,
                stacklevel=2,
            )

        if not bin_to_anchors:
            return

        for bin_id, anchors in bin_to_anchors.items():
            unique = list(dict.fromkeys(anchors))
            self.bin_anchor_lookup[bin_id] = unique

        max_len = max(len(anchors) for anchors in self.bin_anchor_lookup.values())
        bin_ids_sorted = sorted(self.bin_anchor_lookup.keys())
        schedule: List[int] = []
        for offset in range(max_len):
            for bin_id in bin_ids_sorted:
                anchors = self.bin_anchor_lookup[bin_id]
                if offset < len(anchors):
                    schedule.append(bin_id)
        # round robin schedule for distance bins
        self.sample_entries = schedule

        for idx, bin_id in enumerate(self.sample_entries):
            self.bin_to_indices[bin_id].append(idx)

        for bin_id in self.bin_to_indices:
            others = sorted(self.bin_to_indices.keys(), key=lambda x: (abs(x - bin_id), x))
            self.bin_neighbors[bin_id] = [b for b in others if b != bin_id]

        # Ensure all negative pools are finalized, including any keys without corresponding positives
        for key, neg_entry in list(self.neg_pool.items()):
            self.neg_pool[key] = self._finalize_pool_entry(neg_entry)
        
    def __len__(self) -> int:
        return len(self.sample_entries)

    def _choose_bin_id(self, idx: int) -> int:
        if not self.sample_entries:
            raise RuntimeError("Dataset is empty.")
        return self.sample_entries[idx % len(self.sample_entries)]

    def __getitem__(self, idx: int) -> Dict[str, object]:
        bin_id = self._choose_bin_id(idx)
        sample = self._draw_sample_from_bin(bin_id)
        if sample is None:
            for fallback_bin in self.bin_neighbors.get(bin_id, []):
                sample = self._draw_sample_from_bin(fallback_bin)
                if sample is not None:
                    return sample
            sample = self._draw_sample_any()
            if sample is None:
                raise RuntimeError("Unable to construct a sample from any bin.")
        return sample

    def _draw_sample_any(self) -> Optional[Dict[str, object]]:
        for bin_id in sorted(self.bin_to_indices.keys()):
            sample = self._draw_sample_from_bin(bin_id)
            if sample is not None:
                return sample
        return None

    def _draw_sample_from_bin(self, bin_id: int) -> Optional[Dict[str, object]]:
        anchors = self.bin_anchor_lookup.get(bin_id)
        if not anchors:
            return None
        for _ in range(self.anchor_retry_limit):
            chrom, anchor = anchors[self.rng.integers(len(anchors))]
            sample = self._build_sample(chrom, anchor, bin_id)
            if sample is not None:
                return sample
        return None

    def _build_sample(self, chrom: str, anchor: int, bin_id: int) -> Optional[Dict[str, object]]:
        pos_key = (chrom, anchor, bin_id)
        neg_key = pos_key
        pos_entry = self.pos_pool.get(pos_key)
        neg_entry = self.neg_pool.get(neg_key)
        if pos_entry is None or neg_entry is None:
            return None

        if pos_entry["partners"].size == 0 or neg_entry["partners"].size == 0:
            return None

        pos_idx = int(self.rng.integers(pos_entry["partners"].size))
        pos_partner = int(pos_entry["partners"][pos_idx])
        pos_metric = float(pos_entry["metrics"][pos_idx])
        distance_bp = int(pos_entry["distances"][pos_idx])

        neg_needed = self.num_negatives if self.mode == "pair" else 1
        replace = neg_entry["partners"].size < neg_needed

        neg_choices = self.rng.choice(
            neg_entry["partners"].size,
            size=neg_needed,
            replace=replace,
            p=None,
        )
        neg_indices = neg_entry["partners"][neg_choices]
        neg_metrics = neg_entry["metrics"][neg_choices]

        seq_tokens_anchor = self._sequence_tokens(chrom, anchor)
        seq_tokens_pos = self._sequence_tokens(chrom, pos_partner)
        epi_tokens_anchor = self._epigenomic_tokens(chrom, anchor)
        epi_tokens_pos = self._epigenomic_tokens(chrom, pos_partner)

        if self.mode == "pair":
            seq_tokens_negs = torch.stack(
                [self._sequence_tokens(chrom, int(idx)) for idx in neg_indices.tolist()],
                dim=0,
            )
            epi_tokens_negs = torch.stack(
                [self._epigenomic_tokens(chrom, int(idx)) for idx in neg_indices.tolist()],
                dim=0,
            )
        else:
            neg_idx_scalar = int(neg_indices[0])
            seq_tokens_negs = self._sequence_tokens(chrom, neg_idx_scalar)
            epi_tokens_negs = self._epigenomic_tokens(chrom, neg_idx_scalar)

        if self.mode == "pair":
            neg_metrics_tensor = torch.from_numpy(neg_metrics.astype(np.float32, copy=False))
            neg_indices_tensor = torch.from_numpy(neg_indices.astype(np.int64, copy=False))
        else:
            neg_metrics_tensor = torch.tensor(float(neg_metrics[0]), dtype=torch.float32)
            neg_indices_tensor = torch.tensor(int(neg_indices[0]), dtype=torch.long)

        result: Dict[str, object] = {
            "chrom": chrom,
            "anchor_index": int(anchor),
            "pos_index": int(pos_partner),
            "distance_bp": int(distance_bp),
            "bin_id": int(bin_id),
            "pos_metric": float(pos_metric),
            "seq_tokens_anchor": seq_tokens_anchor,
            "seq_tokens_pos": seq_tokens_pos,
            "epi_tokens_anchor": epi_tokens_anchor,
            "epi_tokens_pos": epi_tokens_pos,
        }

        if self.mode == "pair":
            result["neg_indices"] = neg_indices_tensor
            result["neg_metrics"] = neg_metrics_tensor
            result["seq_tokens_negs"] = seq_tokens_negs
            result["epi_tokens_negs"] = epi_tokens_negs
        else:
            result["neg_index"] = neg_indices_tensor
            result["neg_metric"] = neg_metrics_tensor
            result["seq_tokens_neg"] = seq_tokens_negs
            result["epi_tokens_neg"] = epi_tokens_negs

        if self.emit_pos_ids:
            pos_ids = torch.arange(self.tokens_per_locus, dtype=torch.long)
            result["pos_ids"] = pos_ids

        if self.mode == "triplet":
            # keep compatibility naming
            result["neg_metrics"] = neg_metrics_tensor
            result["neg_indices"] = neg_indices_tensor

        return result

    def _get_fasta(self) -> Fasta:
        if self._fa is None:
            self._fa = Fasta(str(self.fasta_path), as_raw=True)
        return self._fa

    def _fetch_sequence(self, chrom: str, bin_index: int) -> torch.FloatTensor:
        fasta = self._get_fasta()
        start = bin_index * self.bin_size
        end = start + self.bin_size
        seq = str(fasta[chrom][start:end])
        if len(seq) < self.bin_size:
            seq = seq + "N" * (self.bin_size - len(seq))
        elif len(seq) > self.bin_size:
            seq = seq[: self.bin_size]
        return _one_hot_encode(seq)

    def _sequence_tokens(self, chrom: str, bin_index: int) -> torch.FloatTensor:
        pre = self.seq_tokens_all.get(chrom)
        if pre is not None and pre.numel() > 0:
            return pre[bin_index].contiguous()
        # fallback (PRECOMPUTE_SEQ=False): old path
        seq = self._fetch_sequence(chrom, bin_index)
        seq = seq[:, : self.bin_size]
        return seq.view(4, self.tokens_per_locus, self.patch_size_bp).permute(1, 0, 2).contiguous().to(dtype=SEQ_DTYPE)

    def _epigenomic_tokens(self, chrom: str, bin_index: int) -> torch.FloatTensor:
        return self.epi_tokens_all[chrom][bin_index].contiguous()

    def close(self) -> None:
        if self._fa is not None:
            self._fa.close()
            self._fa = None

    def _slim_for_cache(self) -> None:
        """Drop heavy fields that are not needed when loading from cache."""
        for chrom, chrom_info in self.chrom_data.items():
            chrom_info.pop("epiphany_raw", None)
            chrom_info.pop("features", None)
            chrom_info.pop("features_norm", None)

    def save_cache_dir(self, cache_dir: Path) -> None:
        """Save lightweight per-chromosome caches to a directory."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "version": 1,
            "bin_edges": self.bin_edges,
            "bin_size": self.bin_size,
            "epiphany_bin_size": self.epiphany_bin_size,
            "patch_size_bp": self.patch_size_bp,
            "min_distance_bp": self.min_distance_bp,
            "max_distance_bp": self.max_distance_bp,
            "pos_quantile": self.pos_quantile,
            "neg_quantile": self.neg_quantile,
            "num_negatives": self.num_negatives,
            "mode": self.mode,
            "bin_schedule": self.bin_schedule,
            "pairs_per_batch": self.pairs_per_batch,
            "emit_pos_ids": self.emit_pos_ids,
            "fasta_path": str(self.fasta_path),
            "epiphany_path": str(self.epiphany_path),
            "hic_root": str(self.hic_root),
            "hic_mapq": self.hic_mapq,
            "hic_field": self.hic_field,
            "hic_norm": self.hic_norm,
            "random_seed": None,
            "chroms": sorted(self.epi_tokens_all.keys()),
        }
        torch.save(meta, cache_dir / "meta.pt")

        for chrom in meta["chroms"]:
            pos_subset = {k: v for k, v in self.pos_pool.items() if k[0] == chrom}
            neg_subset = {k: v for k, v in self.neg_pool.items() if k[0] == chrom}
            thresholds_subset = {k[1]: v for k, v in self.bin_thresholds.items() if k[0] == chrom}
            chrom_payload = {
                "epi_tokens": self.epi_tokens_all[chrom].cpu(),
                "pos_pool": pos_subset,
                "neg_pool": neg_subset,
                "bin_thresholds": thresholds_subset,
                "valid_mask": self.chrom_data[chrom]["valid_mask"],
                "valid_indices": self.chrom_data[chrom]["valid_indices"],
            }
            torch.save(chrom_payload, cache_dir / f"{chrom}.pt")

    def __getstate__(self) -> Dict[str, object]:
        state = self.__dict__.copy()
        fasta = state.pop("_fa", None)
        if fasta is not None:
            fasta.close()
        state["_fa"] = None
        return state

    def __setstate__(self, state: Dict[str, object]) -> None:
        self.__dict__.update(state)
        if self._fa is None:
            self._fa = Fasta(str(self.fasta_path), as_raw=True)

    @classmethod
    def from_cache_dir(cls, cache_dir: str) -> "GenomicsContrastiveDataset":
        cache_path = Path(cache_dir)
        meta_path = cache_path / "meta.pt"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta.pt in cache dir: {cache_dir}")
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)

        self = cls.__new__(cls)
        # basic config
        self.bin_edges = [int(x) for x in meta["bin_edges"]]
        self.bin_size = int(meta["bin_size"])
        self.epiphany_bin_size = int(meta["epiphany_bin_size"])
        self.patch_size_bp = int(meta["patch_size_bp"])
        self.tokens_per_locus = BIN_SIZE // self.patch_size_bp
        self.epi_windows_per_bin = self.bin_size // self.epiphany_bin_size
        self.windows_per_token = self.patch_size_bp // self.epiphany_bin_size
        self.mode = meta["mode"]
        self.bin_schedule = meta["bin_schedule"]
        self.pairs_per_batch = int(meta["pairs_per_batch"])
        self.emit_pos_ids = bool(meta["emit_pos_ids"])
        self.min_distance_bp = int(meta["min_distance_bp"])
        self.max_distance_bp = None if meta["max_distance_bp"] is None else int(meta["max_distance_bp"])
        self.pos_quantile = float(meta["pos_quantile"])
        self.neg_quantile = float(meta["neg_quantile"])
        self.num_negatives = int(meta["num_negatives"])
        self.negative_attempts = 32
        self.anchor_retry_limit = max(4, self.negative_attempts // 2)
        self.rng = np.random.default_rng(meta.get("random_seed", None))

        # paths
        self.fasta_path = Path(meta["fasta_path"])
        self.epiphany_path = Path(meta["epiphany_path"])
        self.hic_root = Path(meta["hic_root"])
        self.hic_mapq = meta["hic_mapq"]
        self.hic_field = meta["hic_field"]
        self.hic_norm = meta["hic_norm"]

        # containers
        self.chrom_data = {}
        self.pos_pool = {}
        self.neg_pool = {}
        self.bin_thresholds = {}
        self.sample_entries = []
        self.bin_ranges = {}
        self.bin_to_indices = defaultdict(list)
        self.bin_anchor_lookup = defaultdict(list)
        self.bin_neighbors = {}
        self.seq_tokens_all = {}
        self.epi_tokens_all = {}
        self._fa = None

        chroms = meta["chroms"]
        for chrom in chroms:
            payload = torch.load(cache_path / f"{chrom}.pt", map_location="cpu", weights_only=False)
            self.epi_tokens_all[chrom] = payload["epi_tokens"]
            self.chrom_data[chrom] = {
                "valid_mask": payload["valid_mask"],
                "valid_indices": payload["valid_indices"],
            }
            for key, val in payload["pos_pool"].items():
                self.pos_pool[key] = val
            for key, val in payload["neg_pool"].items():
                self.neg_pool[key] = val
            for bin_id, thresh in payload["bin_thresholds"].items():
                self.bin_thresholds[(chrom, int(bin_id))] = thresh

        # finalize schedules from cached pools
        self._finalize_dataset()
        for bin_id, (lower, upper) in self._compute_bin_ranges().items():
            self.bin_ranges[bin_id] = (lower, upper)

        return self

    def get_bin_range(self, bin_id: int) -> Tuple[int, Optional[int]]:
        return self.bin_ranges.get(bin_id, (self.bin_edges[0], None))

    def get_indices_for_bin(self, bin_id: int) -> Sequence[int]:
        return self.bin_to_indices.get(bin_id, [])


def distance_binned_collate(batch: List[Dict[str, object]], mode: str) -> Dict[str, object]:
    if not batch:
        raise ValueError("Empty batch")

    mode = mode.lower()
    if mode not in {"pair", "triplet"}:
        raise ValueError("mode must be 'pair' or 'triplet'")

    result: Dict[str, object] = {}
    result["chrom"] = [item["chrom"] for item in batch]
    result["anchor_index"] = torch.tensor([item["anchor_index"] for item in batch], dtype=torch.long)
    result["pos_index"] = torch.tensor([item["pos_index"] for item in batch], dtype=torch.long)
    result["distance_bp"] = torch.tensor([item["distance_bp"] for item in batch], dtype=torch.long)
    result["bin_id"] = torch.tensor([item["bin_id"] for item in batch], dtype=torch.long)
    result["pos_metric"] = torch.tensor([item["pos_metric"] for item in batch], dtype=torch.float32)

    result["seq_tokens_anchor"] = torch.stack([item["seq_tokens_anchor"] for item in batch], dim=0)
    result["seq_tokens_pos"] = torch.stack([item["seq_tokens_pos"] for item in batch], dim=0)
    result["epi_tokens_anchor"] = torch.stack([item["epi_tokens_anchor"] for item in batch], dim=0)
    result["epi_tokens_pos"] = torch.stack([item["epi_tokens_pos"] for item in batch], dim=0)

    if mode == "pair":
        neg_indices = torch.stack([item["neg_indices"] for item in batch], dim=0)
        neg_metrics = torch.stack([item["neg_metrics"] for item in batch], dim=0)
        seq_tokens_negs = torch.stack([item["seq_tokens_negs"] for item in batch], dim=0)
        epi_tokens_negs = torch.stack([item["epi_tokens_negs"] for item in batch], dim=0)
        result["neg_indices"] = neg_indices
        result["neg_metrics"] = neg_metrics
        result["seq_tokens_negs"] = seq_tokens_negs
        result["epi_tokens_negs"] = epi_tokens_negs
    else:
        neg_index = torch.stack([item["neg_indices"] for item in batch], dim=0)
        neg_metric = torch.stack([item["neg_metrics"] for item in batch], dim=0)
        result["neg_index"] = neg_index
        result["neg_indices"] = neg_index
        result["neg_metric"] = neg_metric
        result["neg_metrics"] = neg_metric
        result["seq_tokens_neg"] = torch.stack([item["seq_tokens_neg"] for item in batch], dim=0)
        result["epi_tokens_neg"] = torch.stack([item["epi_tokens_neg"] for item in batch], dim=0)

    if batch[0].get("pos_ids") is not None:
        pos_ids = torch.stack([item["pos_ids"] for item in batch], dim=0)
        result["pos_ids"] = pos_ids

    return result


class DistanceBinBatchSampler(Sampler[List[int]]):
    """Bin-pure batch sampler cycling bins with simple schedules."""

    def __init__(
        self,
        dataset: GenomicsContrastiveDataset,
        pairs_per_batch: int,
        bin_schedule: str = "roundrobin",
        seed: Optional[int] = None,
    ) -> None:
        if dataset.mode != "pair":
            raise ValueError("DistanceBinBatchSampler can only be used with pair mode datasets")
        self.dataset = dataset
        self.pairs_per_batch = int(pairs_per_batch)
        if self.pairs_per_batch <= 0:
            raise ValueError("pairs_per_batch must be positive")
        self.bin_schedule = bin_schedule.lower()
        if self.bin_schedule not in {"roundrobin", "longrange_upweight"}:
            raise ValueError("bin_schedule must be 'roundrobin' or 'longrange_upweight'")
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.bin_ids = sorted(dataset.bin_to_indices.keys())
        if not self.bin_ids:
            raise ValueError("Dataset exposes no bins to sample from")
        self._roundrobin_idx = 0

        self.weights = np.ones(len(self.bin_ids), dtype=np.float64)
        if self.bin_schedule == "longrange_upweight":
            tail = min(2, len(self.bin_ids))
            self.weights[-tail:] *= 2.0
            self.weights /= self.weights.sum()

    def __len__(self) -> int:
        if len(self.dataset) == 0:
            return 0
        return math.ceil(len(self.dataset) / self.pairs_per_batch)

    def __iter__(self):
        self.rng = np.random.default_rng(self.seed)
        self._roundrobin_idx = 0
        for _ in range(len(self)):
            batch_indices: List[int] = []
            for _ in range(self.pairs_per_batch):
                bin_id = self._select_bin()
                candidates = self.dataset.get_indices_for_bin(bin_id)
                if not candidates:
                    continue
                choice = int(self.rng.choice(candidates))
                batch_indices.append(choice)
            if not batch_indices:
                continue
            yield batch_indices

    def _select_bin(self) -> int:
        if self.bin_schedule == "roundrobin":
            bin_id = self.bin_ids[self._roundrobin_idx % len(self.bin_ids)]
            self._roundrobin_idx += 1
            return bin_id
        idx = int(self.rng.choice(len(self.bin_ids), p=self.weights))
        return self.bin_ids[idx]


def _format_range(lower: int, upper: Optional[int]) -> str:
    if upper is None:
        return f"[{lower:,}, inf)"
    return f"[{lower:,}, {upper:,})"


def _print_batch_stats(batch: Dict[str, object], dataset: GenomicsContrastiveDataset, mode: str) -> None:
    mode = mode.lower()
    bin_ids = batch["bin_id"]
    unique_bins = torch.unique(bin_ids).tolist()
    print(f"unique bin_ids: {unique_bins}")
    for bin_id in unique_bins:
        lower, upper = dataset.get_bin_range(int(bin_id))
        print(f"bin {bin_id}: distance range {_format_range(lower, upper)}")

    pos_metric = batch["pos_metric"]
    print(
        f"pos_metric stats -> mean: {float(pos_metric.mean()):.4f} | median: {float(pos_metric.median()):.4f} | std: {float(pos_metric.std(unbiased=False)):.4f}"
    )

    if mode == "pair":
        neg_metrics = batch["neg_metrics"]
        print(
            f"neg_metrics stats -> min: {float(neg_metrics.min()):.4f} | median: {float(neg_metrics.median()):.4f} | max: {float(neg_metrics.max()):.4f}"
        )
    else:
        neg_metric = batch["neg_metric"]
        print(
            f"neg_metric stats -> min: {float(neg_metric.min()):.4f} | median: {float(neg_metric.median()):.4f} | max: {float(neg_metric.max()):.4f}"
        )

    tensor_keys = [
        "seq_tokens_anchor",
        "seq_tokens_pos",
        "epi_tokens_anchor",
        "epi_tokens_pos",
    ]
    if mode == "pair":
        tensor_keys.extend(["seq_tokens_negs", "epi_tokens_negs"])
    else:
        tensor_keys.extend(["seq_tokens_neg", "epi_tokens_neg"])

    for key in tensor_keys:
        value = batch[key]
        print(f"{key} shape: {tuple(value.shape)}")

    if batch.get("pos_ids") is not None:
        print(f"pos_ids shape: {tuple(batch['pos_ids'].shape)}")


def _print_sample_shapes(sample: Dict[str, object]) -> None:
    print("Single sample shapes:")
    for key, value in sample.items():
        if torch.is_tensor(value):
            print(f"{key}: {tuple(value.shape)}")
        elif isinstance(value, np.ndarray):
            print(f"{key}: {value.shape} (numpy)")
        else:
            print(f"{key}: type={type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Genomics contrastive dataset demo")
    parser.add_argument("--fasta", default='/users/jrober48/data/jroberts/2952-G/data/hg38/hg38.fa', help="Path to FASTA file")
    parser.add_argument("--epiphany", default='/users/jrober48/data/jroberts/2952-G/data/epiphany/GM12878_X.h5', help="Path to Epiphany HDF5 file")
    parser.add_argument("--hic-root", default='/users/jrober48/data/jroberts/2952-G/data/hic/GM12878_primary', help="Root directory for Hi-C contacts")
    parser.add_argument("--chroms", nargs="*", default=None, help="Subset of chromosomes to load")

    parser.add_argument("--mode", choices=["pair", "triplet"], default="pair")
    parser.add_argument("--hic-mapq", default="MAPQGE30")
    parser.add_argument("--hic-field", default="RAWobserved")
    parser.add_argument("--hic-norm", choices=["none", "KR", "VC", "SQRTVC"], default="none")
    parser.add_argument("--contact-threshold", type=float, default=None)
    parser.add_argument("--bin-edges", type=str, default=None, help="Comma separated distance bin edges")
    parser.add_argument("--min-distance-bp", type=int, default=25_000)
    parser.add_argument("--max-distance-bp", type=int, default=None)
    parser.add_argument("--oe-metric", choices=["oe", "logresid"], default="oe")
    parser.add_argument("--pos-quantile", type=float, default=0.7)
    parser.add_argument("--neg-quantile", type=float, default=0.3)
    parser.add_argument("--num-negatives", type=int, default=8)
    parser.add_argument("--bin-schedule", choices=["roundrobin", "longrange_upweight"], default="roundrobin")
    parser.add_argument("--pairs-per-batch", type=int, default=64)
    parser.add_argument("--patch-size-bp", type=int, default=100)
    parser.add_argument("--emit-pos-ids", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--index", type=int, default=0, help="Dataset index to inspect (triplet mode)")
    parser.add_argument("--cache-load-dir", type=str, default=None, help="Optional directory to load per-chrom cached data")
    parser.add_argument("--cache-save-dir", type=str, default=None, help="Optional directory to save per-chrom cached data")

    args = parser.parse_args()

    bin_edges = None
    if args.bin_edges:
        bin_edges = [int(edge.strip()) for edge in args.bin_edges.split(",") if edge.strip()]

    dataset: GenomicsContrastiveDataset
    if args.cache_load_dir:
        dataset = GenomicsContrastiveDataset.from_cache_dir(args.cache_load_dir)
    else:
        dataset = GenomicsContrastiveDataset(
            fasta_path=args.fasta,
            epiphany_path=args.epiphany,
            hic_root=args.hic_root,
            chroms=args.chroms,
            contact_threshold=args.contact_threshold,
            hic_mapq=args.hic_mapq,
            hic_field=args.hic_field,
            hic_norm=args.hic_norm,
            mode=args.mode,
            bin_edges=bin_edges,
            min_distance_bp=args.min_distance_bp,
            max_distance_bp=args.max_distance_bp,
            oe_metric=args.oe_metric,
            pos_quantile=args.pos_quantile,
            neg_quantile=args.neg_quantile,
            num_negatives=args.num_negatives,
            bin_schedule=args.bin_schedule,
            pairs_per_batch=args.pairs_per_batch,
            patch_size_bp=args.patch_size_bp,
            emit_pos_ids=args.emit_pos_ids,
            random_seed=args.seed,
        )
        if args.cache_save_dir:
            cache_dir = Path(args.cache_save_dir)
            dataset._slim_for_cache()
            dataset.save_cache_dir(cache_dir)

    if len(dataset) == 0:
        raise RuntimeError("Dataset contains no samples with current configuration")

    sample = dataset[args.index % len(dataset)]
    _print_sample_shapes(sample)

    # # anchors with pos>0 per (chrom, bin)
    # total_counts = defaultdict(int)
    # # anchors with pos>0 but neg==0 per (chrom, bin)
    # empty_counts = defaultdict(int)
    # # anchors with pos>0 and neg>0 (usable)
    # usable_counts = defaultdict(int)

    # for (chrom, anchor, bin_id), pos_entry in dataset.pos_pool.items():
    #     if pos_entry["partners"].size == 0:
    #         continue  # only anchors that truly have positives
    #     total_counts[(chrom, bin_id)] += 1
    #     neg_entry = dataset.neg_pool.get((chrom, anchor, bin_id), {"partners": np.array([], dtype=np.int64)})
    #     if neg_entry["partners"].size == 0:
    #         empty_counts[(chrom, bin_id)] += 1
    #     else:
    #         usable_counts[(chrom, bin_id)] += 1

    # for (chrom, bin_id) in sorted(total_counts.keys()):
    #     total = total_counts[(chrom, bin_id)]
    #     empty = empty_counts.get((chrom, bin_id), 0)
    #     usable = usable_counts.get((chrom, bin_id), 0)
    #     pct_empty = 100 * empty / total if total else 0.0
    #     low, high = dataset.get_bin_range(bin_id)
    #     print(f"{chrom} bin {bin_id} ({low:,}–{high or 'inf'} bp): "
    #         f"empty_neg={empty}/{total} ({pct_empty:.1f}%), usable={usable}")

    # if args.mode == "pair":
    #     sampler = DistanceBinBatchSampler(
    #         dataset=dataset,
    #         pairs_per_batch=args.pairs_per_batch,
    #         bin_schedule=args.bin_schedule,
    #         seed=args.seed,
    #     )
    #     loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=lambda b: distance_binned_collate(b, "pair"))
    #     batch = next(iter(loader))
    #     _print_batch_stats(batch, dataset, mode="pair")
    # else:
    #     loader = DataLoader(
    #         dataset,
    #         batch_size=args.pairs_per_batch,
    #         shuffle=False,
    #         collate_fn=lambda b: distance_binned_collate(b, "triplet"),
    #     )
    #     batch = next(iter(loader))
    #     _print_batch_stats(batch, dataset, mode="triplet")
    #     index = args.index % len(dataset)
    #     sample = dataset[index]
    #     print(
    #         f"Triplet sample idx {index}: chrom={sample['chrom']} anchor={sample['anchor_index']} pos={sample['pos_index']} neg={sample['neg_index']}"
    #     )


if __name__ == "__main__":
    main()
