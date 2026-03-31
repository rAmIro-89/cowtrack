from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.preprocessing.transforms import build_train_transform
from src.reid.dataset import CowReIDDataset
from src.reid.model import EmbeddingNet


@dataclass
class TrainConfig:
    train_dir: str
    output_model_path: str
    metrics_json_path: str | None = None
    metrics_csv_path: str | None = None
    checkpoint_dir: str | None = None
    embedding_dim: int = 256
    batch_size: int = 16
    epochs: int = 10
    lr: float = 3e-4
    margin: float = 0.3


def _build_triplets(labels: torch.Tensor) -> List[tuple[int, int, int]]:
    triplets: List[tuple[int, int, int]] = []
    label_to_indices: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels.tolist()):
        label_to_indices.setdefault(label, []).append(idx)

    for anchor_idx, anchor_label in enumerate(labels.tolist()):
        positives = [i for i in label_to_indices[anchor_label] if i != anchor_idx]
        negatives = [i for lbl, indices in label_to_indices.items() if lbl != anchor_label for i in indices]
        if not positives or not negatives:
            continue
        triplets.append((anchor_idx, positives[0], negatives[0]))
    return triplets


def train_reid(config: TrainConfig) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = CowReIDDataset(root_dir=config.train_dir, transform=build_train_transform())
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    model = EmbeddingNet(embedding_dim=config.embedding_dim, pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.TripletMarginLoss(margin=config.margin, p=2)
    ckpt_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    epoch_metrics: List[dict[str, float | int]] = []
    for epoch in range(config.epochs):
        running_loss = 0.0
        steps_with_triplets = 0
        for images, labels in tqdm(loader, desc=f"epoch {epoch + 1}/{config.epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            embeddings = model(images)
            triplets = _build_triplets(labels)
            if not triplets:
                continue

            a_idx = torch.tensor([t[0] for t in triplets], device=device)
            p_idx = torch.tensor([t[1] for t in triplets], device=device)
            n_idx = torch.tensor([t[2] for t in triplets], device=device)

            anchor = embeddings[a_idx]
            positive = embeddings[p_idx]
            negative = embeddings[n_idx]

            loss = criterion(anchor, positive, negative)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            steps_with_triplets += 1

        avg_loss = running_loss / max(len(loader), 1)
        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "steps_with_triplets": steps_with_triplets,
            }
        )
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}")
        if ckpt_dir is not None:
            ckpt_path = ckpt_dir / f"epoch_{epoch + 1:03d}.pt"
            torch.save(model.state_dict(), ckpt_path)

    output_path = Path(config.output_model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved ReID model to: {output_path}")

    summary = {
        "device": device,
        "train_dir": config.train_dir,
        "checkpoint_dir": str(ckpt_dir) if ckpt_dir is not None else None,
        "num_samples": len(dataset),
        "num_identities": len(dataset.label_to_idx),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "embedding_dim": config.embedding_dim,
        "epoch_metrics": epoch_metrics,
        "final_loss": epoch_metrics[-1]["avg_loss"] if epoch_metrics else None,
        "model_path": str(output_path),
    }

    if config.metrics_json_path:
        metrics_json = Path(config.metrics_json_path)
        metrics_json.parent.mkdir(parents=True, exist_ok=True)
        metrics_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved training metrics JSON to: {metrics_json}")

    if config.metrics_csv_path:
        metrics_csv = Path(config.metrics_csv_path)
        metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        with metrics_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "avg_loss", "steps_with_triplets"])
            writer.writeheader()
            writer.writerows(epoch_metrics)
        print(f"Saved training metrics CSV to: {metrics_csv}")

    return output_path
