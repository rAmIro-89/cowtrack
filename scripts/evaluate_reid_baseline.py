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


def _collect_images(root: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for id_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        imgs = sorted([p for p in id_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
        if imgs:
            out[id_dir.name] = imgs
    return out


@torch.no_grad()
def _embed_paths(model: EmbeddingNet, image_paths: list[Path], image_size: int, device: str) -> np.ndarray:
    transform = build_eval_transform(image_size=image_size)
    tensors = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensors.append(transform(image=rgb)["image"])
    if not tensors:
        return np.zeros((0, 256), dtype=np.float32)

    batch = torch.stack(tensors, dim=0).to(device)
    embs = model(batch).cpu().numpy().astype(np.float32)
    return embs


def _topk_hits(sim_row: np.ndarray, true_id: str, gallery_ids: list[str], k: int) -> bool:
    k = min(k, len(gallery_ids))
    top_idx = np.argsort(sim_row)[::-1][:k]
    return any(gallery_ids[i] == true_id for i in top_idx)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ReID baseline with train-gallery vs val/query top-k")
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--out_json", default="reports/dataset_inspection/opencows2020_reid_val_eval.json")
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("train_dir and val_dir must exist")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingNet(embedding_dim=args.embedding_dim, pretrained=False).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    train_map = _collect_images(train_dir)
    val_map = _collect_images(val_dir)

    gallery_ids: list[str] = []
    gallery_embs: list[np.ndarray] = []
    for identity, paths in train_map.items():
        embs = _embed_paths(model, paths, image_size=args.image_size, device=device)
        if embs.shape[0] == 0:
            continue
        proto = embs.mean(axis=0, keepdims=True)
        proto = proto / np.clip(np.linalg.norm(proto, axis=1, keepdims=True), 1e-9, None)
        gallery_ids.append(identity)
        gallery_embs.append(proto)

    if not gallery_embs:
        raise ValueError("No gallery embeddings were created from train_dir")

    gallery = np.vstack(gallery_embs)

    total = 0
    correct_top1 = 0
    correct_topk = 0
    all_query_embs: list[np.ndarray] = []
    all_query_labels: list[str] = []
    for identity, paths in val_map.items():
        q_embs = _embed_paths(model, paths, image_size=args.image_size, device=device)
        if q_embs.shape[0] == 0:
            continue
        q_embs = q_embs / np.clip(np.linalg.norm(q_embs, axis=1, keepdims=True), 1e-9, None)
        sims = q_embs @ gallery.T
        pred_idx = np.argmax(sims, axis=1)
        preds = [gallery_ids[i] for i in pred_idx.tolist()]
        correct_top1 += sum(1 for p in preds if p == identity)
        correct_topk += sum(1 for i in range(sims.shape[0]) if _topk_hits(sims[i], identity, gallery_ids, args.top_k))
        all_query_embs.append(q_embs)
        all_query_labels.extend([identity] * q_embs.shape[0])
        total += len(preds)

    top1 = (correct_top1 / total) if total else 0.0
    topk = (correct_topk / total) if total else 0.0

    intra_dists: list[float] = []
    inter_dists: list[float] = []
    if all_query_embs:
        emb_all = np.vstack(all_query_embs)
        lbl_all = np.array(all_query_labels)
        sim_all = emb_all @ emb_all.T
        dist_all = 1.0 - sim_all
        n = dist_all.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if lbl_all[i] == lbl_all[j]:
                    intra_dists.append(float(dist_all[i, j]))
                else:
                    inter_dists.append(float(dist_all[i, j]))

    result = {
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "weights": str(args.weights),
        "device": device,
        "gallery_id_count": len(gallery_ids),
        "gallery_item_count": int(gallery.shape[0]),
        "val_id_count": len(val_map),
        "query_count": total,
        "val_images_evaluated": total,
        "top1_accuracy": top1,
        f"top{args.top_k}_accuracy": topk,
        "top_k": args.top_k,
        "intra_id_distance_mean": float(np.mean(intra_dists)) if intra_dists else None,
        "inter_id_distance_mean": float(np.mean(inter_dists)) if inter_dists else None,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
