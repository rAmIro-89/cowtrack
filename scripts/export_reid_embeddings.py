from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.transforms import build_eval_transform
from src.reid.model import EmbeddingNet


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Export embeddings for a split directory by ID")
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--out_meta_json", required=True)
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingNet(embedding_dim=args.embedding_dim, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    transform = build_eval_transform(image_size=args.image_size)

    vectors: list[np.ndarray] = []
    labels: list[str] = []
    paths: list[str] = []

    for id_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        for image_path in sorted([p for p in id_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]):
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = transform(image=rgb)["image"].unsqueeze(0).to(device)
            emb = model(tensor).cpu().numpy().astype(np.float32)
            vectors.append(emb)
            labels.append(id_dir.name)
            paths.append(str(image_path))

    if not vectors:
        raise ValueError(f"No embeddings exported from split_dir={split_dir}")

    mat = np.vstack(vectors)
    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, embeddings=mat, labels=np.array(labels), paths=np.array(paths))

    out_meta = Path(args.out_meta_json)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "split_dir": str(split_dir),
        "weights": args.weights,
        "device": device,
        "count": int(mat.shape[0]),
        "embedding_dim": int(mat.shape[1]),
        "unique_ids": len(set(labels)),
        "out_npz": str(out_npz),
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
