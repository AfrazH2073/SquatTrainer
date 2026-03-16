"""Dataset utilities for squat posture classification."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .bootstrap import ensure_local_dependencies

ensure_local_dependencies()

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_NAMES = ["bad_back", "bad_heel", "good"]
CLASS_NAME_MAP = {
    "badback": "bad_back",
    "badheel": "bad_heel",
    "good": "good",
}
DISPLAY_NAMES = {
    "bad_back": "Bad Back",
    "bad_heel": "Bad Heel",
    "good": "Good",
}


def canonicalize_label(name: str) -> str:
    cleaned = re.sub(r"[^a-z]", "", name.lower())
    if cleaned not in CLASS_NAME_MAP:
        raise ValueError(f"Unsupported class folder name: {name}")
    return CLASS_NAME_MAP[cleaned]


@dataclass(frozen=True)
class SampleRecord:
    image_path: Path
    label_name: str
    label_index: int


class SquatImageDataset(Dataset):
    """Torch dataset that loads squat posture images from sample records."""

    def __init__(self, records: list[SampleRecord], transform: Callable | None = None) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, record.label_index


def scan_split(root_dir: str | Path, split: str) -> list[SampleRecord]:
    split_dir = Path(root_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    records: list[SampleRecord] = []
    label_to_index = {label_name: idx for idx, label_name in enumerate(CLASS_NAMES)}

    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        label_name = canonicalize_label(class_dir.name)
        label_index = label_to_index[label_name]
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(
                    SampleRecord(
                        image_path=image_path,
                        label_name=label_name,
                        label_index=label_index,
                    )
                )

    if not records:
        raise ValueError(f"No image files found in {split_dir}")
    return records


def stratified_train_val_split(
    records: list[SampleRecord],
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    labels = [record.label_index for record in records]
    train_records, val_records = train_test_split(
        records,
        test_size=val_ratio,
        random_state=random_state,
        stratify=labels,
    )
    return list(train_records), list(val_records)


def summarize_records(records: Iterable[SampleRecord]) -> dict[str, int]:
    counts = Counter(record.label_name for record in records)
    return {DISPLAY_NAMES[label_name]: counts.get(label_name, 0) for label_name in CLASS_NAMES}


def class_names_from_records(records: list[SampleRecord]) -> list[str]:
    present = {record.label_name for record in records}
    return [label_name for label_name in CLASS_NAMES if label_name in present]
