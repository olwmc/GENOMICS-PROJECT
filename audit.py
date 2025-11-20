#!/usr/bin/env python3
"""
audit_contrastive.py

Standalone auditor for GenomicsContrastiveDataset/loader outputs.
- Imports dataset primitives from dataset.py in the same folder.
- Inspects a few batches and prints an interpretable report.
- Optionally saves a JSON report and a CSV of "suspicious" pairs.

Usage (examples):
  python audit_contrastive.py \
    --fasta /path/hg38.fa \
    --epiphany /path/GM12878_X.h5 \
    --hic-root /path/GM12878_primary \
    --chroms chr1 chr2 chr3 \
    --mode pair \
    --pairs-per-batch 64 \
    --max-batches 10 \
    --report-json audit.json \
    --suspects-csv suspects.csv
"""

import argparse
import json
import math
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Import from your dataset.py (must be in same directory or on PYTHONPATH)
from dataset import (
    GenomicsContrastiveDataset,
    DistanceBinBatchSampler,
    distance_binned_collate,
)


@torch.no_grad()
def audit_contrastive_loader(
    loader: torch.utils.data.DataLoader,
    dataset: GenomicsContrastiveDataset,
    max_batches: int = 8,
    check_feature_cosine: bool = True,   # uses dataset.chrom_data[chrom]["features_norm"]
    check_epi_cosine: bool = True,       # cosine on mean epigenomic token vectors per locus
    check_seq_gc: bool = True,           # GC% similarity on sequences
    top_k_suspects: int = 10,            # how many suspicious pairs to print
    assert_no_neg_eq_anchor_or_pos: bool = False,  # optional strictness checks
    assert_no_duplicate_negs: bool = False,
) -> Dict[str, object]:
    """
    Audit a few batches to verify that positives look 'closer' and negatives look 'farther'
    under multiple signals (Hi-C metric, feature-space cosine, epi cosine, GC%).

    Prints an interpretable report and returns a metrics dict you can assert on.
    Works for both pair and triplet modes (prefers pair).
    """
    # ------- helpers -------
    def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < eps or nb < eps:
            return np.nan
        return float(np.dot(a, b) / (na * nb))

    def get_neg_distances(chrom: str, anchor: int, bin_id: int, neg_indices: np.ndarray) -> np.ndarray:
        """Lookup genomic distances (bp) for the chosen negatives from the dataset.neg_pool."""
        key = (chrom, anchor, bin_id)
        entry = dataset.neg_pool.get(key)
        if entry is None or entry["partners"].size == 0:
            return np.full(neg_indices.shape, np.nan, dtype=np.float64)
        partners = entry["partners"]
        dists = entry["distances"]
        if not isinstance(partners, np.ndarray):
            partners = np.asarray(partners, dtype=np.int64)
        if not isinstance(dists, np.ndarray):
            dists = np.asarray(dists, dtype=np.int64)
        out = np.empty_like(neg_indices, dtype=np.float64)
        for i, pid in enumerate(neg_indices.tolist()):
            hits = np.flatnonzero(partners == pid)
            out[i] = float(dists[hits[0]]) if hits.size > 0 else np.nan
        return out

    def epi_mean_vector(epi_tokens: torch.Tensor) -> np.ndarray:
        """
        Reduce epi tokens for a locus to a 1D feature vector:
          - thin mode: (T, C) -> mean over T
          - rich mode: (T, C, 100) -> mean over T and 100
        """
        if epi_tokens.dim() == 2:
            v = epi_tokens.mean(dim=0).detach().cpu().float().numpy()
        elif epi_tokens.dim() == 3:
            v = epi_tokens.mean(dim=(0, 2)).detach().cpu().float().numpy()
        else:
            v = epi_tokens.view(-1).detach().cpu().float().numpy()
        return v

    def seq_gc_fraction(seq_tokens: torch.Tensor) -> float:
        """
        Approximate GC% from one-hot tokens:
          seq_tokens: (T, 4, P) where channels order == A,C,G,T
        GC% = (sum over C + sum over G) / total bases
        """
        if seq_tokens.dim() != 3 or seq_tokens.shape[1] != 4:
            return float("nan")
        counts = seq_tokens.sum(dim=(0, 2))  # (4,)
        counts = counts.detach().cpu().float().numpy()
        total = counts.sum()
        if total <= 0:
            return float("nan")
        gc = float(counts[1] + counts[2])  # C + G
        return gc / total

    # ------- accumulators -------
    # Determine mode robustly
    first_item = dataset[0]
    mode = "pair" if ("neg_indices" in first_item or "seq_tokens_negs" in first_item) else "triplet"

    n_batches = 0
    n_items = 0

    # distance checks
    bin_id_hist = Counter()
    pos_distances: List[float] = []
    neg_distances: List[float] = []  # flattened over all negatives
    wrong_bin_count = 0
    wrong_bin_examples: List[Tuple[int, int, Tuple[int, Optional[int]]]] = []

    # metric checks
    pos_metrics: List[float] = []
    neg_metrics: List[float] = []
    below_pos_quantile = 0
    above_neg_quantile = 0
    quantile_total = 0

    # cosine checks
    feat_cos_pos: List[float] = []
    feat_cos_neg: List[float] = []
    epi_cos_pos: List[float] = []
    epi_cos_neg: List[float] = []

    # GC% checks
    gc_diff_pos: List[float] = []
    gc_diff_neg: List[float] = []

    # pair-level suspicious examples
    suspects: List[Tuple[float, Dict[str, object]]] = []

    # ------- iterate batches -------
    for batch in loader:
        n_batches += 1
        if n_batches > max_batches:
            break

        chroms = batch["chrom"]                            # list[str], len=B
        anchors = batch["anchor_index"].tolist()           # [B]
        poses = batch["pos_index"].tolist()                # [B]
        bin_ids = batch["bin_id"].tolist()                 # [B]
        dists_pos = batch["distance_bp"].tolist()          # [B]
        pos_m = batch["pos_metric"].detach().cpu().numpy() # [B]

        # Strictness (optional): ensure no neg equals anchor or pos; ensure no duplicate negatives per row
        if assert_no_neg_eq_anchor_or_pos or assert_no_duplicate_negs:
            if mode == "pair":
                neg_idx_mat = batch["neg_indices"].detach().cpu().numpy()     # [B, K]
            else:
                # [B] -> wrap to [B,1] for uniform loop
                neg_idx_mat = batch["neg_indices"].detach().cpu().numpy().reshape(-1, 1)

            for i in range(len(chroms)):
                ni = neg_idx_mat[i].tolist()
                if assert_no_neg_eq_anchor_or_pos:
                    if int(anchors[i]) in ni:
                        raise AssertionError(f"Negative equals anchor: row {i}, anchor={anchors[i]} in negs={ni}")
                    if int(poses[i]) in ni:
                        raise AssertionError(f"Negative equals positive: row {i}, pos={poses[i]} in negs={ni}")
                if assert_no_duplicate_negs:
                    if len(ni) != len(set(ni)):
                        raise AssertionError(f"Duplicate negatives in row {i}: {ni}")

        bin_id_hist.update(bin_ids)
        pos_distances.extend(dists_pos)
        pos_metrics.extend(pos_m.tolist())

        # sanity: check distance lies in advertised bin range
        for b, d in zip(bin_ids, dists_pos):
            low, up = dataset.get_bin_range(int(b))
            ok = (d >= low) and (up is None or d < up)
            if not ok:
                wrong_bin_count += 1
                wrong_bin_examples.append((int(b), int(d), (int(low), None if up is None else int(up))))

        # neg side
        if mode == "pair":
            neg_idx_mat = batch["neg_indices"].detach().cpu().numpy()     # [B, K]
            neg_m_mat = batch["neg_metrics"].detach().cpu().numpy()       # [B, K]
            K = neg_idx_mat.shape[1]
        else:
            neg_idx_mat = batch["neg_indices"].detach().cpu().numpy()[:, None]  # [B,1]
            neg_m_mat = batch["neg_metrics"].detach().cpu().numpy()[:, None]
            K = 1

        neg_metrics.extend(neg_m_mat.flatten().tolist())

        # compare to bin quantiles + compute distances for chosen negatives
        for i in range(len(chroms)):
            chrom = chroms[i]
            a = int(anchors[i])
            p = int(poses[i])
            b = int(bin_ids[i])

            # look up quantiles for this (chrom, bin)
            q = dataset.bin_thresholds.get((chrom, b))
            if q is not None:
                q_pos, q_neg = q
                quantile_total += 1
                if pos_m[i] < q_pos:
                    below_pos_quantile += 1
                # for negatives, any that violate q_neg?
                if np.any(neg_m_mat[i] > q_neg):
                    above_neg_quantile += 1

            # neg distances from pool
            neg_ids = neg_idx_mat[i]
            nd = get_neg_distances(chrom, a, b, neg_ids)
            neg_distances.extend(nd.tolist())

            # ------- similarities -------
            # feature-space cosine (Epiphany mean-normalized features)
            if check_feature_cosine:
                feats = dataset.chrom_data[chrom]["features_norm"]
                fa = feats[a]
                fp = feats[p]
                feat_pos = cosine(fa, fp)
                feat_cos_pos.append(feat_pos)
                for pid in neg_ids.tolist():
                    feat_cos_neg.append(cosine(fa, feats[int(pid)]))

            # epi cosine (from tokens in batch)
            if check_epi_cosine:
                epi_a = epi_mean_vector(batch["epi_tokens_anchor"][i])
                epi_p = epi_mean_vector(batch["epi_tokens_pos"][i])
                epi_cos_pos.append(cosine(epi_a, epi_p))
                if mode == "pair":
                    for j in range(len(neg_ids)):
                        epi_n = epi_mean_vector(batch["epi_tokens_negs"][i, j])
                        epi_cos_neg.append(cosine(epi_a, epi_n))
                else:
                    epi_n = epi_mean_vector(batch["epi_tokens_neg"][i])
                    epi_cos_neg.append(cosine(epi_a, epi_n))

            # GC% differences
            if check_seq_gc:
                gc_a = seq_gc_fraction(batch["seq_tokens_anchor"][i])
                gc_p = seq_gc_fraction(batch["seq_tokens_pos"][i])
                if not np.isnan(gc_a) and not np.isnan(gc_p):
                    gc_diff_pos.append(abs(gc_a - gc_p))
                if mode == "pair":
                    for j in range(len(neg_ids)):
                        gc_n = seq_gc_fraction(batch["seq_tokens_negs"][i, j])
                        if not np.isnan(gc_a) and not np.isnan(gc_n):
                            gc_diff_neg.append(abs(gc_a - gc_n))
                else:
                    gc_n = seq_gc_fraction(batch["seq_tokens_neg"][i])
                    if not np.isnan(gc_a) and not np.isnan(gc_n):
                        gc_diff_neg.append(abs(gc_a - gc_n))

            # flag suspicious examples (any signal where pos looks worse than negs)
            crit = []
            pos_vs_neg_med = float(pos_m[i]) - float(np.nanmedian(neg_m_mat[i]))
            crit.append(("pos_minus_negMetricMedian", pos_vs_neg_med))
            if check_feature_cosine:
                feats = dataset.chrom_data[chrom]["features_norm"]
                fa = feats[a]; fp = feats[p]
                pos_feat = cosine(fa, fp)
                neg_feat = [cosine(fa, feats[int(pid)]) for pid in neg_ids.tolist()]
                crit.append(("featCos_pos_minus_meanNeg", pos_feat - float(np.nanmean(neg_feat))))
            if check_epi_cosine:
                epi_a = epi_mean_vector(batch["epi_tokens_anchor"][i])
                epi_p = epi_mean_vector(batch["epi_tokens_pos"][i])
                pos_epi = cosine(epi_a, epi_p)
                if mode == "pair":
                    neg_epi = [cosine(epi_a, epi_mean_vector(batch["epi_tokens_negs"][i, j])) for j in range(len(neg_ids))]
                else:
                    neg_epi = [cosine(epi_a, epi_mean_vector(batch["epi_tokens_neg"][i]))]
                crit.append(("epiCos_pos_minus_meanNeg", pos_epi - float(np.nanmean(neg_epi))))

            score = 0.0
            details = {
                "chrom": chrom, "bin_id": int(b),
                "anchor": int(a), "pos": int(p),
                "distance_bp_pos": int(dists_pos[i]),
                "pos_metric": float(pos_m[i]),
                "neg_metrics_sample": [float(x) for x in neg_m_mat[i][:min(3, len(neg_m_mat[i]))].tolist()],
            }
            for name, val in crit:
                details[name] = float(val) if not np.isnan(val) else float("nan")
                if not (val is None or math.isnan(val)) and val <= 0:
                    score += abs(val)
            if score > 0:
                suspects.append((score, details))

        n_items += len(chroms)

    # ------- summarize -------
    def _safe_stats(arr):
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return {"count": 0}
        return {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    report: Dict[str, object] = {
        "mode": mode,
        "batches_inspected": n_batches,
        "items_inspected": n_items,
        "bin_id_hist": dict(bin_id_hist),
        "distance_pos_stats_bp": _safe_stats(pos_distances),
        "distance_neg_stats_bp": _safe_stats(neg_distances),
        "pos_metric_stats": _safe_stats(pos_metrics),
        "neg_metric_stats": _safe_stats(neg_metrics),
        "feat_cos_pos_stats": _safe_stats(feat_cos_pos) if check_feature_cosine else None,
        "feat_cos_neg_stats": _safe_stats(feat_cos_neg) if check_feature_cosine else None,
        "epi_cos_pos_stats": _safe_stats(epi_cos_pos) if check_epi_cosine else None,
        "epi_cos_neg_stats": _safe_stats(epi_cos_neg) if check_epi_cosine else None,
        "gc_absdiff_pos_stats": _safe_stats(gc_diff_pos) if check_seq_gc else None,
        "gc_absdiff_neg_stats": _safe_stats(gc_diff_neg) if check_seq_gc else None,
        "wrong_bin_assignments": int(wrong_bin_count),
        "wrong_bin_examples": wrong_bin_examples[:10],
        "quantile_checks": {
            "checked_pairs": int(quantile_total),
            "pos_below_posQuantile": int(below_pos_quantile),
            "neg_above_negQuantile": int(above_neg_quantile),
        },
        # include suspects in report for optional CSV dump
        "suspects": [det for _, det in sorted(suspects, key=lambda x: -x[0])],
    }

    # convenience deltas
    if check_feature_cosine and report["feat_cos_pos_stats"] and report["feat_cos_neg_stats"]:
        report["feat_cos_gap_mean"] = report["feat_cos_pos_stats"]["mean"] - report["feat_cos_neg_stats"]["mean"]
    if check_epi_cosine and report["epi_cos_pos_stats"] and report["epi_cos_neg_stats"]:
        report["epi_cos_gap_mean"] = report["epi_cos_pos_stats"]["mean"] - report["epi_cos_neg_stats"]["mean"]
    if check_seq_gc and report["gc_absdiff_pos_stats"] and report["gc_absdiff_neg_stats"]:
        report["gc_absdiff_gap_mean"] = report["gc_absdiff_neg_stats"]["mean"] - report["gc_absdiff_pos_stats"]["mean"]

    # ------- print readable summary -------
    print("\n=== Contrastive Batch Audit ===")
    print(f"Mode: {report['mode']} | Batches: {report['batches_inspected']} | Items: {report['items_inspected']}")
    print(f"Bins covered: {sorted(report['bin_id_hist'].keys())}  (counts: {report['bin_id_hist']})")

    # Distance sanity
    dp, dn = report["distance_pos_stats_bp"], report["distance_neg_stats_bp"]
    def _fmt(stats, key):
        return f"{stats.get(key, '-'):.1f}" if key in stats else "-"
    print(f"\n[Distance bp]  pos: n={dp.get('count',0)}  mean={_fmt(dp,'mean')}  "
          f"median={_fmt(dp,'median')}  |  neg: n={dn.get('count',0)}  mean={_fmt(dn,'mean')}  "
          f"median={_fmt(dn,'median')}")

    # Quantile rule adherence
    qc = report["quantile_checks"]
    if qc["checked_pairs"] > 0:
        print(f"[Quantiles] pairs checked={qc['checked_pairs']} | "
              f"pos below pos-quantile={qc['pos_below_posQuantile']} | "
              f"neg above neg-quantile={qc['neg_above_negQuantile']}")

    # Metric separation
    pm, nm = report["pos_metric_stats"], report["neg_metric_stats"]
    if pm.get("count", 0) and nm.get("count", 0):
        print(f"[Hi-C metric] pos mean={pm['mean']:.3f}, median={pm['median']:.3f}  "
              f"| neg mean={nm['mean']:.3f}, median={nm['median']:.3f}")

    # Cosine separations
    if check_feature_cosine and report["feat_cos_pos_stats"] and report["feat_cos_neg_stats"]:
        fpos, fneg = report["feat_cos_pos_stats"], report["feat_cos_neg_stats"]
        gap = report.get("feat_cos_gap_mean", float('nan'))
        print(f"[Feature cosine] pos mean={fpos['mean']:.3f} vs neg mean={fneg['mean']:.3f}  (gap={gap:.3f})")
    if check_epi_cosine and report["epi_cos_pos_stats"] and report["epi_cos_neg_stats"]:
        epos, eneg = report["epi_cos_pos_stats"], report["epi_cos_neg_stats"]
        gap = report.get("epi_cos_gap_mean", float('nan'))
        print(f"[Epi cosine]     pos mean={epos['mean']:.3f} vs neg mean={eneg['mean']:.3f}  (gap={gap:.3f})")
    if check_seq_gc and report["gc_absdiff_pos_stats"] and report["gc_absdiff_neg_stats"]:
        gpos, gneg = report["gc_absdiff_pos_stats"], report["gc_absdiff_neg_stats"]
        gap = report.get("gc_absdiff_gap_mean", float('nan'))
        print(f"[GC% abs diff]   pos mean={gpos['mean']:.4f} vs neg mean={gneg['mean']:.4f}  (neg-pos gap={gap:.4f})")

    # Bin-range errors
    if report["wrong_bin_assignments"] > 0:
        print(f"[WARNING] {report['wrong_bin_assignments']} samples had distance outside their bin range.")
        for b, d, (lo, up) in report["wrong_bin_examples"]:
            print(f"  - bin={b} dist={d} not in [{lo}, {up or 'inf'})")

    # Top suspicious pairs
    suspects_sorted = [det for _, det in sorted(suspects, key=lambda x: -x[0])]
    if suspects_sorted:
        print(f"\n[Suspect pairs] top {min(top_k_suspects, len(suspects_sorted))}/{len(suspects_sorted)} where pos looks weak:")
        for det in suspects_sorted[:top_k_suspects]:
            print(f"  chrom={det['chrom']} bin={det['bin_id']} anchor={det['anchor']} "
                  f"pos={det['pos']} dist={det['distance_bp_pos']:,} "
                  f"| pos_metric={det['pos_metric']:.3f} "
                  f"| Δmetric={det.get('pos_minus_negMetricMedian', float('nan')):.3f} "
                  f"| ΔfeatCos={det.get('featCos_pos_minus_meanNeg', float('nan')):.3f} "
                  f"| ΔepiCos={det.get('epiCos_pos_minus_meanNeg', float('nan')):.3f}")
    else:
        print("\n[Suspect pairs] none flagged by current criteria.")

    print("=== End Audit ===\n")
    return report


def save_report_json(report: Dict[str, object], path: str) -> None:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def save_suspects_csv(report: Dict[str, object], path: str) -> None:
    import csv
    suspects = report.get("suspects", [])
    if not suspects:
        # still create an empty CSV with header
        header = ["chrom","bin_id","anchor","pos","distance_bp_pos","pos_metric",
                  "pos_minus_negMetricMedian","featCos_pos_minus_meanNeg","epiCos_pos_minus_meanNeg"]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
        return

    keys = ["chrom","bin_id","anchor","pos","distance_bp_pos","pos_metric",
            "pos_minus_negMetricMedian","featCos_pos_minus_meanNeg","epiCos_pos_minus_meanNeg"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for det in suspects:
            row = {k: det.get(k) for k in keys}
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit GenomicsContrastiveDataset dataloader pairs")
    # Dataset args (mirror your dataset's CLI where it matters)
    p.add_argument("--fasta", required=True, help="Path to FASTA file")
    p.add_argument("--epiphany", required=True, help="Path to Epiphany HDF5 file")
    p.add_argument("--hic-root", required=True, help="Root directory for Hi-C contacts")
    p.add_argument("--chroms", nargs="*", default=None, help="Subset of chromosomes to load")

    p.add_argument("--mode", choices=["pair", "triplet"], default="pair")
    p.add_argument("--hic-mapq", default="MAPQGE30")
    p.add_argument("--hic-field", default="RAWobserved")
    p.add_argument("--hic-norm", choices=["none", "KR", "VC", "SQRTVC"], default="none")
    p.add_argument("--bin-edges", type=str, default=None, help="Comma separated distance bin edges")
    p.add_argument("--min-distance-bp", type=int, default=25_000)
    p.add_argument("--max-distance-bp", type=int, default=None)
    p.add_argument("--oe-metric", choices=["oe", "logresid"], default="oe")
    p.add_argument("--pos-quantile", type=float, default=0.65)
    p.add_argument("--neg-quantile", type=float, default=0.35)
    p.add_argument("--num-negatives", type=int, default=8)
    p.add_argument("--hard-negative", action="store_true", default=False)
    p.add_argument("--bin-schedule", choices=["roundrobin", "longrange_upweight"], default="roundrobin")
    p.add_argument("--pairs-per-batch", type=int, default=64)
    p.add_argument("--patch-size-bp", type=int, default=100)
    p.add_argument("--emit-pos-ids", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)

    # Auditor args
    p.add_argument("--max-batches", type=int, default=8)
    p.add_argument("--no-feature-cosine", action="store_true", help="Disable feature cosine check")
    p.add_argument("--no-epi-cosine", action="store_true", help="Disable epigenomic cosine check")
    p.add_argument("--no-seq-gc", action="store_true", help="Disable GC%% similarity check")
    p.add_argument("--top-k-suspects", type=int, default=10)
    p.add_argument("--assert-no-neg-eq-anchor-or-pos", action="store_true", help="Raise if any neg equals anchor or pos")
    p.add_argument("--assert-no-duplicate-negs", action="store_true", help="Raise if any duplicates among negatives per item")

    # Outputs
    p.add_argument("--report-json", type=str, default=None, help="Optional path to save the JSON report")
    p.add_argument("--suspects-csv", type=str, default=None, help="Optional path to save a CSV of suspicious pairs")

    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    bin_edges = None
    if args.bin_edges:
        bin_edges = [int(edge.strip()) for edge in args.bin_edges.split(",") if edge.strip()]

    dataset = GenomicsContrastiveDataset(
        fasta_path=args.fasta,
        epiphany_path=args.epiphany,
        hic_root=args.hic_root,
        chroms=args.chroms,
        contact_threshold=None,  # deprecated
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
        hard_negative=args.hard_negative,
        bin_schedule=args.bin_schedule,
        pairs_per_batch=args.pairs_per_batch,
        patch_size_bp=args.patch_size_bp,
        emit_pos_ids=args.emit_pos_ids,
        random_seed=args.seed,
    )

    if len(dataset) == 0:
        raise RuntimeError("Dataset contains no samples with current configuration")

    # Build loader
    if args.mode == "pair":
        sampler = DistanceBinBatchSampler(
            dataset=dataset,
            pairs_per_batch=args.pairs_per_batch,
            bin_schedule=args.bin_schedule,
            seed=args.seed,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=lambda b: distance_binned_collate(b, "pair"),
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=args.pairs_per_batch,
            shuffle=False,
            collate_fn=lambda b: distance_binned_collate(b, "triplet"),
        )

    report = audit_contrastive_loader(
        loader=loader,
        dataset=dataset,
        max_batches=args.max_batches,
        check_feature_cosine=not args.no_feature_cosine,
        check_epi_cosine=not args.no_epi_cosine,
        check_seq_gc=not args.no_seq_gc,
        top_k_suspects=args.top_k_suspects,
        assert_no_neg_eq_anchor_or_pos=args.assert_no_neg_eq_anchor_or_pos,
        assert_no_duplicate_negs=args.assert_no_duplicate_negs,
    )

    if args.report_json:
        save_report_json(report, args.report_json)
        print(f"[Saved] JSON report -> {args.report_json}")
    if args.suspects_csv:
        save_suspects_csv(report, args.suspects_csv)
        print(f"[Saved] Suspects CSV -> {args.suspects_csv}")


if __name__ == "__main__":
    main()
