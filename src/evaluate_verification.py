from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from utils import (
    default_transform,
    evaluate_scores,
    make_lfw_pairs,
    plot_roc,
    save_metrics,
    set_seed,
    similarity_score,
)



def build_backbone(name: str) -> nn.Module:
    name = name.lower()
    if name == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        backbone = nn.Sequential(
            model.features,
            model.avgpool,
            nn.Flatten(),
            *list(model.classifier.children())[:-1],
        )
        feat_dim = model.classifier[-1].in_features
    elif name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        backbone = nn.Sequential(
            model.features,
            model.avgpool,
            nn.Flatten(),
            *list(model.classifier.children())[:-1],
        )
        feat_dim = model.classifier[-1].in_features
    else:
        raise ValueError(f"Unsupported model: {name}")

    backbone.feat_dim = feat_dim  # type: ignore[attr-defined]
    return backbone



def load_checkpoint_if_needed(model_name: str, checkpoint: str, device: torch.device) -> nn.Module:
    if checkpoint.lower() == "none":
        return build_backbone(model_name).to(device)

    cls_model: nn.Module
    if model_name == "alexnet":
        cls_model = models.alexnet(weights=None)
    elif model_name == "vgg16":
        cls_model = models.vgg16(weights=None)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    ckpt = torch.load(checkpoint, map_location=device)
    num_classes = int(ckpt["num_classes"])
    in_dim = cls_model.classifier[-1].in_features
    cls_model.classifier[-1] = nn.Linear(in_dim, num_classes)
    cls_model.load_state_dict(ckpt["state_dict"])

    backbone = nn.Sequential(
        cls_model.features,
        cls_model.avgpool,
        nn.Flatten(),
        *list(cls_model.classifier.children())[:-1],
    )
    return backbone.to(device)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--model", type=str, choices=["alexnet", "vgg16"], required=True)
    parser.add_argument("--checkpoint", type=str, default="none")
    parser.add_argument("--metric", type=str, choices=["cosine", "euclidean", "l1"], default="cosine")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="outputs/eval")
    parser.add_argument("--max_pairs", type=int, default=0)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = default_transform(224)
    dataset = make_lfw_pairs(args.data_root, split="10fold", transform=transform)
    if args.max_pairs > 0:
        dataset = torch.utils.data.Subset(dataset, list(range(min(args.max_pairs, len(dataset)))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = load_checkpoint_if_needed(args.model, args.checkpoint, device)
    model.eval()

    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for img1, img2, target in tqdm(loader, desc="eval"):
            img1 = img1.to(device)
            img2 = img2.to(device)

            feat1 = model(img1)
            feat2 = model(img2)
            score = similarity_score(feat1, feat2, metric=args.metric)

            all_scores.append(score.cpu().numpy())
            all_labels.append(target.numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels).astype(np.int32)

    result = evaluate_scores(scores, labels)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "scores_labels.npz", scores=scores, labels=labels)
    save_metrics(result, out_dir / "metrics.json")
    plot_roc(result, out_dir / "roc.png", title=f"{args.model.upper()}-{args.metric}")

    print(f"AUC: {result.auc_value:.6f}")
    print(f"EER: {result.eer:.6f}")
    print(f"Best threshold: {result.best_threshold:.6f}")
    print(f"Saved outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
