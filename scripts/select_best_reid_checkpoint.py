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
    return model(batch).cpu().numpy().astype(np.float32)


def _topk_hits(sim_row: np.ndarray, true_id: str, gallery_ids: list[str], k: int) -> bool:
    k = min(k, len(gallery_ids))
    idx = np.argsort(sim_row)[::-1][:k]
    return any(gallery_ids[i] == true_id for i in idx)


@torch.no_grad()
def evaluate_checkpoint(
    ckpt: Path,
    train_map: dict[str, list[Path]],
    val_map: dict[str, list[Path]],
    embedding_dim: int,
    image_size: int,
    top_k: int,
    device: str,
) -> dict[str, float | int | str]:
    model = EmbeddingNet(embedding_dim=embedding_dim, pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    gallery_ids: list[str] = []
    gallery_embs: list[np.ndarray] = []
    for identity, paths in train_map.items():
        embs = _embed_paths(model, paths, image_size=image_size, device=device)
        if embs.shape[0] == 0:
            continue
        proto = np.mean(embs, axis=0, keepdims=True)
        proto = proto / np.clip(np.linalg.norm(proto, axis=1, keepdims=True), 1e-9, None)
        gallery_ids.append(identity)
        gallery_embs.append(proto)

    if not gallery_embs:
        raise ValueError(f"No gallery embeddings for checkpoint {ckpt}")

    gallery = np.vstack(gallery_embs)
    total = 0
    ok1 = 0
    okk = 0
    for identity, paths in val_map.items():
        q_embs = _embed_paths(model, paths, image_size=image_size, device=device)
        if q_embs.shape[0] == 0:
            continue
        q_embs = q_embs / np.clip(np.linalg.norm(q_embs, axis=1, keepdims=True), 1e-9, None)
        sims = q_embs @ gallery.T
        pred = np.argmax(sims, axis=1)
        preds = [gallery_ids[i] for i in pred.tolist()]
        ok1 += sum(1 for p in preds if p == identity)
        okk += sum(1 for i in range(sims.shape[0]) if _topk_hits(sims[i], identity, gallery_ids, top_k))
        total += len(preds)

    return {
        "checkpoint": str(ckpt),
        "top1_accuracy": (ok1 / total) if total else 0.0,
        f"top{top_k}_accuracy": (okk / total) if total else 0.0,
        "query_count": total,
        "gallery_id_count": len(gallery_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints and select best by Top-1")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    checkpoints = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No epoch checkpoints found in {ckpt_dir}")

    train_map = _collect_images(train_dir)
    val_map = _collect_images(val_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows: list[dict[str, float | int | str]] = []
    for ckpt in checkpoints:
        rows.append(
            evaluate_checkpoint(
                ckpt=ckpt,
                train_map=train_map,
                val_map=val_map,
                embedding_dim=args.embedding_dim,
                image_size=args.image_size,
                top_k=args.top_k,
                device=device,
            )
        )

    rows = sorted(rows, key=lambda r: (float(r["top1_accuracy"]), float(r[f"top{args.top_k}_accuracy"])), reverse=True)
    best = rows[0]

    by_path = {r["checkpoint"]: r for r in rows}
    last_ckpt = str(checkpoints[-1])
    last = by_path[last_ckpt]

    output = {
        "device": device,
        "top_k": args.top_k,
        "candidates_evaluated": len(rows),
        "best": best,
        "last_checkpoint": last,
        "all_ranked": rows,
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")

    md = [
        "# Best checkpoint selection",
        "",
        f"- Device: {device}",
        f"- Candidates evaluated: {len(rows)}",
        f"- Best checkpoint: {best['checkpoint']}",
        f"- Best Top-1: {best['top1_accuracy']:.6f}",
        f"- Best Top-{args.top_k}: {best[f'top{args.top_k}_accuracy']:.6f}",
        f"- Last checkpoint: {last['checkpoint']}",
        f"- Last Top-1: {last['top1_accuracy']:.6f}",
        f"- Last Top-{args.top_k}: {last[f'top{args.top_k}_accuracy']:.6f}",
        "",
        "## Ranked checkpoints",
    ]
    for i, r in enumerate(rows, start=1):
        md.append(
            f"{i}. {r['checkpoint']} | Top-1={float(r['top1_accuracy']):.6f} | Top-{args.top_k}={float(r[f'top{args.top_k}_accuracy']):.6f}"
        )
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
