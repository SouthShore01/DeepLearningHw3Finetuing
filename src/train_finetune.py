from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from utils import default_transform, make_lfw_people, set_seed



def str2bool(x: str) -> bool:
    return x.lower() in {"1", "true", "yes", "y"}



def build_model(name: str, num_classes: int, freeze_backbone: bool) -> nn.Module:
    name = name.lower()
    if name == "alexnet":
        weights = models.AlexNet_Weights.IMAGENET1K_V1
        model = models.alexnet(weights=weights)
        in_dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_dim, num_classes)
    elif name == "vgg16":
        weights = models.VGG16_Weights.IMAGENET1K_V1
        model = models.vgg16(weights=weights)
        in_dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_dim, num_classes)
    else:
        raise ValueError(f"Unsupported model: {name}")

    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    return model



def run_one_epoch(model: nn.Module, loader: DataLoader, optimizer: Adam, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total += x.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)



def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in tqdm(loader, desc="val", leave=False):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += float(loss.item()) * x.size(0)
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total += x.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--model", type=str, choices=["alexnet", "vgg16"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_backbone", type=str, default="false")
    parser.add_argument("--min_faces_per_person", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="outputs/train")
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = default_transform(224)

    train_set = make_lfw_people(
        root=args.data_root,
        split="train",
        transform=transform,
        min_faces_per_person=args.min_faces_per_person,
    )
    val_set = make_lfw_people(
        root=args.data_root,
        split="test",
        transform=transform,
        min_faces_per_person=args.min_faces_per_person,
    )

    if args.max_train_samples > 0:
        train_set = torch.utils.data.Subset(
            train_set, list(range(min(args.max_train_samples, len(train_set))))
        )
    if args.max_val_samples > 0:
        val_set = torch.utils.data.Subset(
            val_set, list(range(min(args.max_val_samples, len(val_set))))
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(
        name=args.model,
        num_classes=(
            len(train_set.dataset.classes)
            if isinstance(train_set, torch.utils.data.Subset)
            else len(train_set.classes)
        ),
        freeze_backbone=str2bool(args.freeze_backbone),
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_one_epoch(model, train_loader, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, device)

        print(
            f"[Epoch {epoch:02d}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
        )

        if va_acc > best_acc:
            best_acc = va_acc
            ckpt = {
                "model_name": args.model,
                "state_dict": model.state_dict(),
                "num_classes": (
                    len(train_set.dataset.classes)
                    if isinstance(train_set, torch.utils.data.Subset)
                    else len(train_set.classes)
                ),
                "best_val_acc": best_acc,
            }
            torch.save(ckpt, out_dir / "best.pt")


if __name__ == "__main__":
    main()
