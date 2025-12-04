import torch
import torch.nn.functional as F
from torchmetrics.functional import (
    accuracy,
    f1_score,
    precision,
    recall,
    confusion_matrix,
    auroc,
    average_precision,
)
from typing import List, Dict, Any, Tuple

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_cosine_similarity(
    embeds1: torch.Tensor, embeds2: torch.Tensor
) -> torch.Tensor:

    return F.cosine_similarity(embeds1, embeds2, dim=1)


def evaluate_contrastive_pairs(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    labels: torch.Tensor,
    thresholds: List[float],
) -> Dict[str, Any]:

    embeddings_a = embeddings_a.to(DEVICE)
    embeddings_b = embeddings_b.to(DEVICE)
    labels = labels.to(DEVICE).long()

    # cosine Similarities
    scores = compute_cosine_similarity(embeddings_a, embeddings_b)

    # overall metrics
    try:
        roc_auc_val = auroc(scores, labels, task="binary").item()
    except Exception:
        roc_auc_val = 0.0

    try:
        pr_auc_val = average_precision(scores, labels, task="binary").item()
    except Exception:
        pr_auc_val = 0.0

    results = {
        "global_metrics": {
            "roc_auc": roc_auc_val,
            "pr_auc_score": pr_auc_val,
            "mean_similarity": scores.mean().item(),
            "std_similarity": scores.std().item(),
        },
        "per_threshold": {},
    }

    # metrics based on thresholds
    for t in thresholds:
        # convert to binary predictions based on threshold
        predictions = (scores >= t).long()

        # compute confusion matrix components
        cm = confusion_matrix(predictions, labels, task="binary", num_classes=2)
        tn, fp, fn, tp = cm.flatten().tolist()

        metric_pack = {
            "accuracy": accuracy(predictions, labels, task="binary").item(),
            "f1_score": f1_score(predictions, labels, task="binary").item(),
            "precision": precision(predictions, labels, task="binary").item(),
            "recall": recall(predictions, labels, task="binary").item(),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        }

        results["per_threshold"][t] = metric_pack

    return results
