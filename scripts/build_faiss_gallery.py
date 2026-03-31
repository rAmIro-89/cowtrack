from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import faiss
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
    parser = argparse.ArgumentParser(description="Build FAISS gallery from train split")
    parser.add_argument("--gallery_dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--index_out", required=True)
    parser.add_argument("--meta_out", required=True)
    parser.add_argument("--mode", default="prototype", choices=["prototype", "all"])
    args = parser.parse_args()

    gallery_dir = Path(args.gallery_dir)
    if not gallery_dir.exists():
        raise FileNotFoundError(f"gallery_dir not found: {gallery_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingNet(embedding_dim=args.embedding_dim, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    transform = build_eval_transform(image_size=args.image_size)

    vectors: list[np.ndarray] = []
    labels: list[str] = []
    image_paths: list[str] = []

    for id_dir in sorted([p for p in gallery_dir.iterdir() if p.is_dir()]):
        embs: list[np.ndarray] = []
        used_paths: list[str] = []
        for image_path in sorted([p for p in id_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]):
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = transform(image=rgb)["image"].unsqueeze(0).to(device)
            emb = model(tensor).cpu().numpy().astype(np.float32)
            emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
            embs.append(emb)
            used_paths.append(str(image_path))

        if not embs:
            continue

        if args.mode == "prototype":
            stack = np.vstack(embs)
            proto = np.mean(stack, axis=0, keepdims=True)
            proto = proto / np.clip(np.linalg.norm(proto, axis=1, keepdims=True), 1e-9, None)
            vectors.append(proto)
            labels.append(id_dir.name)
            image_paths.append(used_paths[0])
        else:
            for emb, p in zip(embs, used_paths):
                vectors.append(emb)
                labels.append(id_dir.name)
                image_paths.append(p)

    if not vectors:
        raise ValueError("No gallery embeddings were created")

    mat = np.vstack(vectors).astype(np.float32)
    index = faiss.IndexFlatIP(args.embedding_dim)
    index.add(mat)

    index_out = Path(args.index_out)
    meta_out = Path(args.meta_out)
    index_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_out))

    meta = {
        "gallery_dir": str(gallery_dir),
        "weights": args.weights,
        "mode": args.mode,
        "device": device,
        "embedding_dim": args.embedding_dim,
        "vectors_count": int(mat.shape[0]),
        "unique_ids": len(set(labels)),
        "labels": labels,
        "image_paths": image_paths,
        "index_path": str(index_out),
    }
    meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k not in {"labels", "image_paths"}}, indent=2))


if __name__ == "__main__":
    main()
