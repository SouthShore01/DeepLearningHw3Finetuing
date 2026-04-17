from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class VerificationResult:
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc_value: float
    eer: float
    best_threshold: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def default_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )



def make_lfw_people(root: str, split: str, transform: Callable, min_faces_per_person: int = 20):
    return datasets.LFWPeople(
        root=root,
        split=split,
        image_set="deepfunneled",
        transform=transform,
        download=True,
        min_faces_per_person=min_faces_per_person,
    )



def make_lfw_pairs(root: str, split: str, transform: Callable):
    return datasets.LFWPairs(
        root=root,
        split=split,
        image_set="deepfunneled",
        transform=transform,
        download=True,
    )



def similarity_score(feat1: torch.Tensor, feat2: torch.Tensor, metric: str = "cosine") -> torch.Tensor:
    metric = metric.lower()
    if metric == "cosine":
        return F.cosine_similarity(feat1, feat2)
    if metric == "euclidean":
        return -torch.norm(feat1 - feat2, p=2, dim=1)
    if metric == "l1":
        return -torch.norm(feat1 - feat2, p=1, dim=1)
    raise ValueError(f"Unsupported metric: {metric}")



def evaluate_scores(scores: np.ndarray, labels: np.ndarray) -> VerificationResult:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc_value = auc(fpr, tpr)

    fnr = 1.0 - tpr
    idx_eer = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fnr[idx_eer] + fpr[idx_eer]) / 2.0)

    best_idx = int(np.nanargmax(tpr - fpr))
    best_threshold = float(thresholds[best_idx])

    return VerificationResult(
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        auc_value=float(auc_value),
        eer=eer,
        best_threshold=best_threshold,
    )



def save_metrics(result: VerificationResult, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "auc": result.auc_value,
        "eer": result.eer,
        "best_threshold": result.best_threshold,
    }
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")



def plot_roc(result: VerificationResult, out_file: Path, title: str) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.plot(result.fpr, result.tpr, label=f"ROC (AUC={result.auc_value:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=160)
    plt.close()
