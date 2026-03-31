from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from src.preprocessing.transforms import build_eval_transform
from src.reid.faiss_index import FaissGallery
from src.reid.model import EmbeddingNet


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS gallery from labeled identity folders")
    parser.add_argument("--dataset_dir", required=True, help="Folder with subfolders per cow ID")
    parser.add_argument("--weights", required=True, help="Path to trained ReID model")
    parser.add_argument("--out_npz", required=True, help="Output path for identity embeddings npz")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingNet(embedding_dim=256, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    transform = build_eval_transform(image_size=224)

    identity_embeddings: dict[str, list[np.ndarray]] = defaultdict(list)
    root = Path(args.dataset_dir)

    with torch.no_grad():
        for identity_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            for image_path in identity_dir.glob("*"):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                img = cv2.imread(str(image_path))
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tensor = transform(image=rgb)["image"].unsqueeze(0).to(device)
                emb = model(tensor).cpu().numpy().astype(np.float32)
                identity_embeddings[identity_dir.name].append(emb)

    gallery = FaissGallery(embedding_dim=256)
    gallery.aggregate_gallery_from_embeddings(identity_embeddings)

    packed = {k: np.vstack(v) for k, v in identity_embeddings.items() if v}
    np.savez(args.out_npz, **packed)
    print(f"Saved gallery embeddings to {args.out_npz}")


if __name__ == "__main__":
    main()
