from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import auc, roc_curve
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LFW_IMAGE_DIR_BY_SET = {
    "original": "lfw",
    "funneled": "lfw_funneled",
    "deepfunneled": "lfw-deepfunneled",
}
LFW_PEOPLE_SPLIT_FILE = {
    "10fold": "people.txt",
    "train": "peopleDevTrain.txt",
    "test": "peopleDevTest.txt",
}
LFW_PAIRS_SPLIT_FILE = {
    "10fold": "pairs.txt",
    "train": "pairsDevTrain.txt",
    "test": "pairsDevTest.txt",
}
LFW_PEOPLE_SPLIT_FILE_CSV = {
    "10fold": "people.csv",
    "train": "peopleDevTrain.csv",
    "test": "peopleDevTest.csv",
}
LFW_PAIRS_SPLIT_FILE_CSV = {
    "10fold": "pairs.csv",
    "train": "pairsDevTrain.csv",
    "test": "pairsDevTest.csv",
}


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


def _resolve_lfw_base_dir(root: str) -> Path:
    root_path = Path(root)
    if root_path.name == "lfw-py":
        return root_path
    if (root_path / "lfw-deepfunneled").exists():
        return root_path
    return root_path / "lfw-py"


def _resolve_lfw_image_dir(base_dir: Path, image_set: str) -> Path:
    configured_dir_name = LFW_IMAGE_DIR_BY_SET[image_set]
    direct_path = base_dir / configured_dir_name
    nested_path = direct_path / configured_dir_name
    if nested_path.exists() and nested_path.is_dir():
        return nested_path
    return direct_path


def _resolve_first_existing_file(base_dir: Path, candidate_names: list[str]) -> Path:
    for candidate_name in candidate_names:
        candidate_path = base_dir / candidate_name
        if candidate_path.exists():
            return candidate_path
    return base_dir / candidate_names[0]


def _format_missing_files_error(base_dir: Path, missing_items: list[Path]) -> str:
    missing_preview = "\n".join(f"  - {item}" for item in missing_items[:8])
    if len(missing_items) > 8:
        missing_preview += "\n  - ..."
    return (
        "LFW dataset files are missing.\n"
        f"Expected base directory: {base_dir}\n"
        f"Missing items:\n{missing_preview}\n"
        "Please manually download and extract LFW under the base directory."
    )


def _read_non_empty_lines(file_path: Path) -> list[str]:
    if not file_path.exists():
        raise RuntimeError(f"Required file not found: {file_path}")
    return [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_csv_rows(file_path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with file_path.open("r", encoding="utf-8", newline="") as file_handle:
        csv_reader = csv.reader(file_handle)
        for row in csv_reader:
            normalized_row = [column.strip() for column in row if column.strip() != ""]
            if normalized_row:
                rows.append(normalized_row)
    return rows


def _is_readable_image(image_path: Path) -> bool:
    if not image_path.exists():
        return False
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except Exception:
        return False


class ManualLFWPeople(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        image_set: str,
        transform: Callable | None,
        min_faces_per_person: int = 0,
    ) -> None:
        self.base_dir = _resolve_lfw_base_dir(root)
        self.split = split.lower()
        self.image_set = image_set.lower()
        self.transform = transform
        self.min_faces_per_person = max(min_faces_per_person, 0)

        if self.split not in LFW_PEOPLE_SPLIT_FILE:
            raise ValueError(f"Unsupported split for LFWPeople: {split}")
        if self.image_set not in LFW_IMAGE_DIR_BY_SET:
            raise ValueError(f"Unsupported image_set for LFWPeople: {image_set}")

        self.image_dir = _resolve_lfw_image_dir(self.base_dir, self.image_set)
        self.labels_file = _resolve_first_existing_file(
            self.base_dir,
            [LFW_PEOPLE_SPLIT_FILE[self.split], LFW_PEOPLE_SPLIT_FILE_CSV[self.split]],
        )
        self.samples, self.targets, self.classes = self._build_samples()

    def _build_samples(self) -> tuple[list[Path], list[int], list[str]]:
        missing_items: list[Path] = []
        if not self.base_dir.exists():
            missing_items.append(self.base_dir)
        if not self.image_dir.exists():
            missing_items.append(self.image_dir)
        if not self.labels_file.exists():
            missing_items.append(self.labels_file)
        if missing_items:
            raise RuntimeError(_format_missing_files_error(self.base_dir, missing_items))

        if self.labels_file.suffix.lower() == ".csv":
            lines = []
            csv_rows = _read_csv_rows(self.labels_file)
            for csv_row in csv_rows[1:]:
                if len(csv_row) >= 2:
                    lines.append(f"{csv_row[0]}\t{csv_row[1]}")
            lines = [str(len(lines))] + lines
        else:
            lines = _read_non_empty_lines(self.labels_file)
        identity_to_image_count: dict[str, int] = {}

        if self.split == "10fold":
            number_of_folds = int(lines[0])
            cursor = 1
            for _ in range(number_of_folds):
                number_of_identities = int(lines[cursor])
                cursor += 1
                for line in lines[cursor : cursor + number_of_identities]:
                    identity_name, image_count_text = line.split("\t")
                    image_count = int(image_count_text)
                    previous_count = identity_to_image_count.get(identity_name, 0)
                    identity_to_image_count[identity_name] = max(previous_count, image_count)
                cursor += number_of_identities
        else:
            number_of_identities = int(lines[0])
            for line in lines[1 : 1 + number_of_identities]:
                identity_name, image_count_text = line.split("\t")
                identity_to_image_count[identity_name] = int(image_count_text)

        class_names = sorted(
            identity_name
            for identity_name, image_count in identity_to_image_count.items()
            if image_count >= self.min_faces_per_person
        )
        class_to_index = {identity_name: index for index, identity_name in enumerate(class_names)}

        image_paths: list[Path] = []
        targets: list[int] = []
        for identity_name in class_names:
            image_count = identity_to_image_count[identity_name]
            for image_number in range(1, image_count + 1):
                image_path = self.image_dir / identity_name / f"{identity_name}_{image_number:04d}.jpg"
                if _is_readable_image(image_path):
                    image_paths.append(image_path)
                    targets.append(class_to_index[identity_name])

        if not image_paths:
            raise RuntimeError(
                f"No valid LFW people images found in {self.image_dir}. "
                "Please verify that the dataset was extracted correctly."
            )

        return image_paths, targets, class_names

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path = self.samples[index]
        target = self.targets[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class ManualLFWPairs(Dataset):
    def __init__(self, root: str, split: str, image_set: str, transform: Callable | None) -> None:
        self.base_dir = _resolve_lfw_base_dir(root)
        self.split = split.lower()
        self.image_set = image_set.lower()
        self.transform = transform

        if self.split not in LFW_PAIRS_SPLIT_FILE:
            raise ValueError(f"Unsupported split for LFWPairs: {split}")
        if self.image_set not in LFW_IMAGE_DIR_BY_SET:
            raise ValueError(f"Unsupported image_set for LFWPairs: {image_set}")

        self.image_dir = _resolve_lfw_image_dir(self.base_dir, self.image_set)
        self.labels_file = _resolve_first_existing_file(
            self.base_dir,
            [LFW_PAIRS_SPLIT_FILE[self.split], LFW_PAIRS_SPLIT_FILE_CSV[self.split]],
        )
        self.samples = self._build_samples()

    def _pair_image_path(self, identity_name: str, image_number: int) -> Path:
        return self.image_dir / identity_name / f"{identity_name}_{image_number:04d}.jpg"

    def _build_samples(self) -> list[tuple[Path, Path, int]]:
        missing_items: list[Path] = []
        if not self.base_dir.exists():
            missing_items.append(self.base_dir)
        if not self.image_dir.exists():
            missing_items.append(self.image_dir)
        if not self.labels_file.exists():
            missing_items.append(self.labels_file)
        if missing_items:
            raise RuntimeError(_format_missing_files_error(self.base_dir, missing_items))

        if self.labels_file.suffix.lower() == ".csv":
            return self._build_samples_from_csv()

        lines = _read_non_empty_lines(self.labels_file)
        samples: list[tuple[Path, Path, int]] = []

        if self.split == "10fold":
            first_line_parts = lines[0].split("\t")
            number_of_folds = int(first_line_parts[0])
            number_of_pairs_per_fold = int(first_line_parts[1])
            cursor = 1
            for _ in range(number_of_folds):
                matched_pairs = lines[cursor : cursor + number_of_pairs_per_fold]
                unmatched_pairs = lines[
                    cursor + number_of_pairs_per_fold : cursor + 2 * number_of_pairs_per_fold
                ]
                cursor += 2 * number_of_pairs_per_fold

                for pair_line in matched_pairs:
                    identity_name, first_image_text, second_image_text = pair_line.split("\t")
                    first_image_path = self._pair_image_path(identity_name, int(first_image_text))
                    second_image_path = self._pair_image_path(identity_name, int(second_image_text))
                    if _is_readable_image(first_image_path) and _is_readable_image(second_image_path):
                        samples.append((first_image_path, second_image_path, 1))
                for pair_line in unmatched_pairs:
                    first_identity_name, first_image_text, second_identity_name, second_image_text = pair_line.split(
                        "\t"
                    )
                    first_image_path = self._pair_image_path(first_identity_name, int(first_image_text))
                    second_image_path = self._pair_image_path(second_identity_name, int(second_image_text))
                    if _is_readable_image(first_image_path) and _is_readable_image(second_image_path):
                        samples.append((first_image_path, second_image_path, 0))
        else:
            number_of_pairs = int(lines[0])
            matched_pairs = lines[1 : 1 + number_of_pairs]
            unmatched_pairs = lines[1 + number_of_pairs : 1 + 2 * number_of_pairs]
            for pair_line in matched_pairs:
                identity_name, first_image_text, second_image_text = pair_line.split("\t")
                first_image_path = self._pair_image_path(identity_name, int(first_image_text))
                second_image_path = self._pair_image_path(identity_name, int(second_image_text))
                if _is_readable_image(first_image_path) and _is_readable_image(second_image_path):
                    samples.append((first_image_path, second_image_path, 1))
            for pair_line in unmatched_pairs:
                first_identity_name, first_image_text, second_identity_name, second_image_text = pair_line.split("\t")
                first_image_path = self._pair_image_path(first_identity_name, int(first_image_text))
                second_image_path = self._pair_image_path(second_identity_name, int(second_image_text))
                if _is_readable_image(first_image_path) and _is_readable_image(second_image_path):
                    samples.append((first_image_path, second_image_path, 0))

        if not samples:
            raise RuntimeError(
                f"No valid LFW pairs found from {self.labels_file}. "
                "Please verify that pair files and images exist under lfw-py."
            )

        return samples

    def _build_samples_from_csv(self) -> list[tuple[Path, Path, int]]:
        csv_rows = _read_csv_rows(self.labels_file)
        if len(csv_rows) <= 1:
            raise RuntimeError(f"CSV file is empty: {self.labels_file}")

        samples: list[tuple[Path, Path, int]] = []
        for csv_row in csv_rows[1:]:
            if len(csv_row) == 3:
                identity_name = csv_row[0]
                first_image_path = self._pair_image_path(identity_name, int(csv_row[1]))
                second_image_path = self._pair_image_path(identity_name, int(csv_row[2]))
                if _is_readable_image(first_image_path) and _is_readable_image(second_image_path):
                    samples.append((first_image_path, second_image_path, 1))
            elif len(csv_row) >= 4:
                first_identity_name = csv_row[0]
                second_identity_name = csv_row[2]
                first_image_path = self._pair_image_path(first_identity_name, int(csv_row[1]))
                second_image_path = self._pair_image_path(second_identity_name, int(csv_row[3]))
                if _is_readable_image(first_image_path) and _is_readable_image(second_image_path):
                    samples.append((first_image_path, second_image_path, 0))

        if not samples and self.split in {"10fold", "test", "train"}:
            matched_fallback = self.base_dir / f"matchpairsDev{self.split.capitalize()}.csv"
            mismatched_fallback = self.base_dir / f"mismatchpairsDev{self.split.capitalize()}.csv"
            if matched_fallback.exists() and mismatched_fallback.exists():
                samples.extend(self._read_matched_pairs_csv(matched_fallback))
                samples.extend(self._read_mismatched_pairs_csv(mismatched_fallback))

        if not samples and self.split == "10fold":
            matched_test = self.base_dir / "matchpairsDevTest.csv"
            mismatched_test = self.base_dir / "mismatchpairsDevTest.csv"
            if matched_test.exists() and mismatched_test.exists():
                samples.extend(self._read_matched_pairs_csv(matched_test))
                samples.extend(self._read_mismatched_pairs_csv(mismatched_test))

        if not samples:
            raise RuntimeError(
                f"No valid CSV pairs found from {self.labels_file}. "
                "Please verify CSV columns and image paths."
            )
        return samples

    def _read_matched_pairs_csv(self, csv_path: Path) -> list[tuple[Path, Path, int]]:
        csv_rows = _read_csv_rows(csv_path)
        samples: list[tuple[Path, Path, int]] = []
        for csv_row in csv_rows[1:]:
            if len(csv_row) < 3:
                continue
            identity_name = csv_row[0]
            first_image_path = self._pair_image_path(identity_name, int(csv_row[1]))
            second_image_path = self._pair_image_path(identity_name, int(csv_row[2]))
            if _is_readable_image(first_image_path) and _is_readable_image(second_image_path):
                samples.append((first_image_path, second_image_path, 1))
        return samples

    def _read_mismatched_pairs_csv(self, csv_path: Path) -> list[tuple[Path, Path, int]]:
        csv_rows = _read_csv_rows(csv_path)
        samples: list[tuple[Path, Path, int]] = []
        for csv_row in csv_rows[1:]:
            if len(csv_row) < 4:
                continue
            first_identity_name = csv_row[0]
            second_identity_name = csv_row[2]
            first_image_path = self._pair_image_path(first_identity_name, int(csv_row[1]))
            second_image_path = self._pair_image_path(second_identity_name, int(csv_row[3]))
            if _is_readable_image(first_image_path) and _is_readable_image(second_image_path):
                samples.append((first_image_path, second_image_path, 0))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        first_image_path, second_image_path, target = self.samples[index]
        first_image = Image.open(first_image_path).convert("RGB")
        second_image = Image.open(second_image_path).convert("RGB")
        if self.transform is not None:
            first_image = self.transform(first_image)
            second_image = self.transform(second_image)
        return first_image, second_image, target


def make_lfw_people(root: str, split: str, transform: Callable, min_faces_per_person: int = 20):
    return ManualLFWPeople(
        root=root,
        split=split,
        image_set="deepfunneled",
        transform=transform,
        min_faces_per_person=min_faces_per_person,
    )


def make_lfw_pairs(root: str, split: str, transform: Callable):
    return ManualLFWPairs(
        root=root,
        split=split,
        image_set="deepfunneled",
        transform=transform,
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
