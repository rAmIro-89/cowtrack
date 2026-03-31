from __future__ import annotations

import argparse
import csv
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
    parser = argparse.ArgumentParser(description="Query test split against FAISS gallery")
    parser.add_argument("--query_dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--meta_path", required=True)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--unknown_threshold", type=float, default=None)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    query_dir = Path(args.query_dir)
    if not query_dir.exists():
        raise FileNotFoundError(f"query_dir not found: {query_dir}")

    index = faiss.read_index(str(Path(args.index_path)))
    meta = json.loads(Path(args.meta_path).read_text(encoding="utf-8"))
    labels: list[str] = meta["labels"]
    k = min(args.top_k, index.ntotal)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingNet(embedding_dim=args.embedding_dim, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    transform = build_eval_transform(image_size=args.image_size)

    rows: list[dict[str, object]] = []
    top1_ok = 0
    topk_ok = 0
    top1_ok_thresholded = 0
    topk_ok_thresholded = 0
    total = 0
    query_ids: set[str] = set()
    rejected = 0
    false_rejects = 0
    false_accepts = 0
    accepted = 0

    for id_dir in sorted([p for p in query_dir.iterdir() if p.is_dir()]):
        for image_path in sorted([p for p in id_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]):
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = transform(image=rgb)["image"].unsqueeze(0).to(device)
            emb = model(tensor).cpu().numpy().astype(np.float32)
            emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)

            sims, idxs = index.search(emb, k)
            idx_list = [int(i) for i in idxs[0].tolist() if i >= 0]
            pred_ids = [labels[i] for i in idx_list]
            true_id = id_dir.name
            query_ids.add(true_id)
            pred_top1 = pred_ids[0] if pred_ids else "unknown"
            top1_similarity = float(sims[0][0]) if len(sims[0]) else 0.0
            ok1 = int(pred_top1 == true_id)
            okk = int(true_id in pred_ids)
            top1_ok += ok1
            topk_ok += okk
            total += 1

            is_rejected = False
            if args.unknown_threshold is not None:
                is_rejected = top1_similarity < args.unknown_threshold
                if is_rejected:
                    rejected += 1
                    # In this protocol all queries are known IDs, so rejection is false reject.
                    false_rejects += 1
                else:
                    accepted += 1
                    if ok1:
                        top1_ok_thresholded += 1
                    if okk:
                        topk_ok_thresholded += 1
                    if not ok1:
                        false_accepts += 1

            rows.append(
                {
                    "query_path": str(image_path),
                    "true_id": true_id,
                    "pred_top1": pred_top1,
                    "pred_topk": "|".join(pred_ids),
                    "top1_correct": ok1,
                    "topk_correct": okk,
                    "top1_similarity": top1_similarity,
                    "rejected_unknown": int(is_rejected),
                    "top1_correct_thresholded": int((not is_rejected) and ok1),
                    "topk_correct_thresholded": int((not is_rejected) and okk),
                }
            )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_path",
                "true_id",
                "pred_top1",
                "pred_topk",
                "top1_correct",
                "topk_correct",
                "top1_similarity",
                "rejected_unknown",
                "top1_correct_thresholded",
                "topk_correct_thresholded",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if args.unknown_threshold is None:
        top1_ok_thresholded = top1_ok
        topk_ok_thresholded = topk_ok
        accepted = total
        rejected = 0
        false_rejects = 0
        false_accepts = total - top1_ok

    result = {
        "query_dir": str(query_dir),
        "index_path": args.index_path,
        "meta_path": args.meta_path,
        "weights": args.weights,
        "device": device,
        "gallery_vectors": int(index.ntotal),
        "gallery_unique_ids": len(set(labels)),
        "query_count": total,
        "query_unique_ids": len(query_ids),
        "top_k": k,
        "unknown_threshold": args.unknown_threshold,
        "top1_accuracy": (top1_ok / total) if total else 0.0,
        f"top{k}_accuracy": (topk_ok / total) if total else 0.0,
        "top1_accuracy_thresholded": (top1_ok_thresholded / total) if total else 0.0,
        f"top{k}_accuracy_thresholded": (topk_ok_thresholded / total) if total else 0.0,
        "unknown_rejection_rate": (rejected / total) if total else 0.0,
        "false_rejects": false_rejects,
        "false_accepts": false_accepts,
        "accepted_queries": accepted,
        "details_csv": str(out_csv),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
