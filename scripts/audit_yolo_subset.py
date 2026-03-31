from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import cv2


def parse_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            continue
        boxes.append((cls, x, y, w, h))
    return boxes


def draw_boxes(image, boxes):
    h, w = image.shape[:2]
    for cls, x, y, bw, bh in boxes:
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (30, 220, 70), 2)
        cv2.putText(image, f"cls={cls}", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 220, 70), 1)
    return image


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit YOLO subset quality and produce visual samples")
    parser.add_argument("--subset_root", default="data/processed/yolo_aerial_baseline")
    parser.add_argument("--reports_dir", default="reports/subset_audit")
    parser.add_argument("--sample_count", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    subset_root = (project_root / args.subset_root).resolve()
    reports_dir = (project_root / args.reports_dir).resolve()
    samples_dir = reports_dir / "sample_annotated_frames"
    reports_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    if not subset_root.exists():
        raise FileNotFoundError(f"Subset root not found: {subset_root}")

    split_stats = {}
    all_records = []
    class_counter = Counter()
    invalid_bbox_count = 0
    missing_label_count = 0
    malformed_label_count = 0

    for split in ["train", "val"]:
        images_dir = subset_root / "images" / split
        labels_dir = subset_root / "labels" / split
        images = sorted([p for p in images_dir.glob("*") if p.is_file()])

        empty_labels = 0
        object_count = 0
        tiny_boxes = 0

        for img_path in images:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                missing_label_count += 1
                boxes = []
            else:
                boxes = parse_label(label_path)
                raw_lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                non_empty_lines = [ln for ln in raw_lines if ln.strip()]
                if len(non_empty_lines) > 0 and len(boxes) == 0:
                    malformed_label_count += 1

            if len(boxes) == 0:
                empty_labels += 1

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            valid_boxes = 0
            for cls, x, y, bw, bh in boxes:
                class_counter[cls] += 1
                # normalized validity
                valid = 0 <= x <= 1 and 0 <= y <= 1 and 0 < bw <= 1 and 0 < bh <= 1
                if not valid:
                    invalid_bbox_count += 1
                    continue
                valid_boxes += 1
                object_count += 1
                area = bw * bh
                if area < 0.0005:
                    tiny_boxes += 1

            all_records.append(
                {
                    "split": split,
                    "image_path": str(img_path),
                    "label_path": str(label_path),
                    "n_boxes": valid_boxes,
                    "width": w,
                    "height": h,
                    "md5": md5_file(img_path),
                }
            )

        split_stats[split] = {
            "images": len(images),
            "empty_labels": empty_labels,
            "objects": object_count,
            "tiny_boxes": tiny_boxes,
            "empty_pct": (empty_labels / len(images) * 100.0) if images else 0.0,
        }

    # duplicate detection by md5
    md5_counter = Counter(r["md5"] for r in all_records)
    duplicate_images = sum(1 for _, c in md5_counter.items() if c > 1)
    duplicate_files_total = sum(c for _, c in md5_counter.items() if c > 1)

    # save full records csv
    records_csv = reports_dir / "subset_image_records.csv"
    with records_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)

    # sample annotated frames
    rng = random.Random(args.seed)
    sample_pool = [r for r in all_records if r["n_boxes"] > 0]
    rng.shuffle(sample_pool)
    sample_pool = sample_pool[: min(args.sample_count, len(sample_pool))]

    for i, rec in enumerate(sample_pool):
        img_path = Path(rec["image_path"])
        lbl_path = Path(rec["label_path"])
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        boxes = parse_label(lbl_path)
        ann = draw_boxes(img, boxes)
        out_name = f"{rec['split']}_{i:03d}_{img_path.name}"
        cv2.imwrite(str(samples_dir / out_name), ann)

    summary = {
        "subset_root": str(subset_root),
        "train_images": split_stats["train"]["images"],
        "val_images": split_stats["val"]["images"],
        "train_empty_labels": split_stats["train"]["empty_labels"],
        "val_empty_labels": split_stats["val"]["empty_labels"],
        "train_empty_pct": round(split_stats["train"]["empty_pct"], 2),
        "val_empty_pct": round(split_stats["val"]["empty_pct"], 2),
        "train_objects": split_stats["train"]["objects"],
        "val_objects": split_stats["val"]["objects"],
        "train_tiny_boxes": split_stats["train"]["tiny_boxes"],
        "val_tiny_boxes": split_stats["val"]["tiny_boxes"],
        "missing_label_count": missing_label_count,
        "malformed_label_count": malformed_label_count,
        "invalid_bbox_count": invalid_bbox_count,
        "duplicate_image_hash_groups": duplicate_images,
        "duplicate_image_files_total": duplicate_files_total,
        "class_distribution": dict(class_counter),
        "visual_samples_saved": len(sample_pool),
    }

    summary_json = reports_dir / "subset_audit_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_csv = reports_dir / "subset_audit_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(summary.keys()))
        writer.writerow([json.dumps(v) if isinstance(v, dict) else v for v in summary.values()])

    md = [
        "# Subset Audit Report",
        "",
        f"- Train images: {summary['train_images']}",
        f"- Val images: {summary['val_images']}",
        f"- Empty labels train/val: {summary['train_empty_labels']} / {summary['val_empty_labels']}",
        f"- Empty label % train/val: {summary['train_empty_pct']} / {summary['val_empty_pct']}",
        f"- Objects train/val: {summary['train_objects']} / {summary['val_objects']}",
        f"- Tiny boxes train/val: {summary['train_tiny_boxes']} / {summary['val_tiny_boxes']}",
        f"- Missing labels: {summary['missing_label_count']}",
        f"- Malformed labels: {summary['malformed_label_count']}",
        f"- Invalid bbox entries: {summary['invalid_bbox_count']}",
        f"- Duplicate image hash groups: {summary['duplicate_image_hash_groups']}",
        f"- Duplicate image files total: {summary['duplicate_image_files_total']}",
        f"- Class distribution: {summary['class_distribution']}",
        "",
        "## Visual inspection samples",
        "",
        f"- {samples_dir}",
    ]
    (reports_dir / "subset_audit_summary.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Saved audit summary: {summary_json}")
    print(f"Saved visual samples: {samples_dir}")


if __name__ == "__main__":
    main()
