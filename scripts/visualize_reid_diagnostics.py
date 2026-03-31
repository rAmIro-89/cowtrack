from __future__ import annotations

import argparse
import csv
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


def _collect_images(subset_dir: Path, max_images: int | None = None) -> list[Path]:
    images = sorted([p for p in subset_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if max_images is not None:
        images = images[:max_images]
    return images


@torch.no_grad()
def _embed(model: EmbeddingNet, image_paths: list[Path], image_size: int, device: str) -> tuple[np.ndarray, list[Path]]:
    transform = build_eval_transform(image_size=image_size)
    tensors = []
    used_paths: list[Path] = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensors.append(transform(image=rgb)["image"])
        used_paths.append(path)

    if not tensors:
        return np.zeros((0, 256), dtype=np.float32), []

    batch = torch.stack(tensors, dim=0).to(device)
    embs = model(batch).cpu().numpy().astype(np.float32)
    embs = embs / np.clip(np.linalg.norm(embs, axis=1, keepdims=True), 1e-9, None)
    return embs, used_paths


def _save_similarity_heatmap(sim: np.ndarray, out_png: Path) -> None:
    if sim.size == 0:
        return
    sim01 = ((sim + 1.0) / 2.0).clip(0.0, 1.0)
    gray = (sim01 * 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    scale = 8 if sim.shape[0] <= 64 else 4
    heat = cv2.resize(heat, (heat.shape[1] * scale, heat.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), heat)


def _write_neighbors(sim: np.ndarray, used_paths: list[Path], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_image", "query_id", "nn1_image", "nn1_id", "nn1_similarity"])
        if sim.size == 0:
            return
        for i in range(sim.shape[0]):
            row = sim[i].copy()
            row[i] = -np.inf
            j = int(np.argmax(row))
            q = used_paths[i]
            nn = used_paths[j]
            writer.writerow([str(q), q.parent.name, str(nn), nn.parent.name, float(sim[i, j])])


def _save_pca_projection(embs: np.ndarray, labels: list[str], out_png: Path) -> None:
    if embs.shape[0] < 2:
        return
    x = embs - np.mean(embs, axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    pca2 = x @ vt[:2].T

    w, h = 900, 700
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    xmin, ymin = pca2.min(axis=0)
    xmax, ymax = pca2.max(axis=0)
    xr = max(xmax - xmin, 1e-6)
    yr = max(ymax - ymin, 1e-6)

    unique = sorted(set(labels))
    palette = {}
    for i, lbl in enumerate(unique):
        hue = int((i * 179) / max(len(unique), 1))
        color = cv2.cvtColor(np.uint8([[[hue, 200, 220]]]), cv2.COLOR_HSV2BGR)[0, 0]
        palette[lbl] = (int(color[0]), int(color[1]), int(color[2]))

    for i, pt in enumerate(pca2):
        px = int(40 + (pt[0] - xmin) / xr * (w - 80))
        py = int(40 + (pt[1] - ymin) / yr * (h - 80))
        py = h - py
        cv2.circle(canvas, (px, py), 4, palette[labels[i]], -1)

    cv2.putText(canvas, "PCA projection of subset embeddings", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ReID diagnostics visual artifacts")
    parser.add_argument("--subset_dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_images", type=int, default=80)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    subset_dir = Path(args.subset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _collect_images(subset_dir, max_images=args.max_images)
    if not image_paths:
        raise ValueError(f"No images found in subset_dir={subset_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingNet(embedding_dim=args.embedding_dim, pretrained=False).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    embs, used_paths = _embed(model, image_paths, image_size=args.image_size, device=device)
    if embs.shape[0] == 0:
        raise ValueError("No embeddings produced from subset images")

    sim = embs @ embs.T
    sim_png = out_dir / "subset_similarity_matrix.png"
    nn_csv = out_dir / "subset_nearest_neighbors.csv"
    nn_examples_csv = out_dir / "subset_nn_examples.csv"
    pca_png = out_dir / "subset_pca_projection.png"
    summary_json = out_dir / "diagnostics_summary.json"

    _save_similarity_heatmap(sim, sim_png)
    _write_neighbors(sim, used_paths, nn_csv)
    labels = [p.parent.name for p in used_paths]
    _save_pca_projection(embs, labels, pca_png)

    # Store a compact table of correct/incorrect NN examples.
    with nn_examples_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_image", "query_id", "nn1_image", "nn1_id", "nn1_similarity", "is_correct"])
        for i in range(sim.shape[0]):
            row = sim[i].copy()
            row[i] = -np.inf
            j = int(np.argmax(row))
            q = used_paths[i]
            nn = used_paths[j]
            is_ok = q.parent.name == nn.parent.name
            writer.writerow([str(q), q.parent.name, str(nn), nn.parent.name, float(sim[i, j]), int(is_ok)])

    unique_ids = sorted({p.parent.name for p in used_paths})
    result = {
        "subset_dir": str(subset_dir),
        "num_images_used": len(used_paths),
        "num_unique_ids": len(unique_ids),
        "unique_ids": unique_ids,
        "embedding_shape": [int(embs.shape[0]), int(embs.shape[1])],
        "similarity_matrix_png": str(sim_png),
        "nearest_neighbors_csv": str(nn_csv),
        "nearest_neighbor_examples_csv": str(nn_examples_csv),
        "pca_projection_png": str(pca_png),
        "device": device,
    }
    summary_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
