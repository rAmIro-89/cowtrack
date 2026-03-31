from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import torch
from torch.utils.data import Dataset


@dataclass
class Sample:
    path: Path
    label_idx: int


class CowReIDDataset(Dataset):
    """Dataset structure: root/individual_id/image.jpg."""

    def __init__(self, root_dir: str, transform: Callable | None = None) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}
        self.samples: List[Sample] = []
        self._load_samples()

    def _load_samples(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        labels = sorted([p.name for p in self.root_dir.iterdir() if p.is_dir()])
        self.label_to_idx = {label: i for i, label in enumerate(labels)}
        self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}

        for label in labels:
            label_dir = self.root_dir / label
            for image_path in sorted(label_dir.glob("*")):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                self.samples.append(Sample(path=image_path, label_idx=self.label_to_idx[label]))

        if not self.samples:
            raise ValueError(f"No image samples found in {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[index]
        image = cv2.imread(str(sample.path))
        if image is None:
            raise ValueError(f"Failed to read image: {sample.path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, sample.label_idx
