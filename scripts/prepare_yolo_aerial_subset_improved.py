from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sample:
    image_path: Path
    label_path: Path
    object_count: int
    max_box_area: float


def find_corresponding_image(label_path: Path, zenodo_root: Path) -> Path | None:
    parts = label_path.parts
    if "YOLO_1.1" in parts:
        yolo_idx = parts.index("YOLO_1.1")
    elif "YOLO1.1" in parts:
        yolo_idx = parts.index("YOLO1.1")
    else:
        return None

    if yolo_idx + 2 >= len(parts):
        return None

    site_path = Path(*parts[:yolo_idx])
    sequence_name = parts[yolo_idx + 1]
    stem = label_path.stem

    candidates = [
        zenodo_root / site_path / "JPGImages" / sequence_name / f"{stem}.JPG",
        zenodo_root / site_path / "JPGImages" / sequence_name / f"{stem}.jpg",
        zenodo_root / site_path / "JPGImages" / sequence_name / f"{stem}.JPEG",
        zenodo_root / site_path / "JPGImages" / sequence_name / f"{stem}.jpeg",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def parse_valid_lines(label_path: Path, min_box_area: float) -> tuple[list[str], int, float]:
    valid_lines: list[str] = []
    max_area = 0.0
    for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            _cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            continue
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            continue
        area = w * h
        if area < min_box_area:
            continue
        max_area = max(max_area, area)
        valid_lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return valid_lines, len(valid_lines), max_area


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare improved aerial YOLO subset")
    parser.add_argument("--zenodo_root", default="data/raw/zenodo_grazing_cows/extracted/Cattle_drone_images")
    parser.add_argument("--out_root", default="data/processed/yolo_aerial_improved")
    parser.add_argument("--reports_dir", default="reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--non_empty_max", type=int, default=500)
    parser.add_argument("--empty_max", type=int, default=40)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--min_box_area", type=float, default=0.0005)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    zenodo_root = (project_root / args.zenodo_root).resolve()
    out_root = (project_root / args.out_root).resolve()
    reports_dir = (project_root / args.reports_dir).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    if out_root.exists():
        shutil.rmtree(out_root)

    label_files = [p for p in zenodo_root.rglob("*.txt") if "obj_train_data" in str(p)]
    label_files.sort()

    valid_samples: list[Sample] = []
    missing_image = 0

    for lbl in label_files:
        img = find_corresponding_image(lbl, zenodo_root)
        if img is None:
            missing_image += 1
            continue
        _lines, count, max_area = parse_valid_lines(lbl, min_box_area=args.min_box_area)
        valid_samples.append(Sample(image_path=img, label_path=lbl, object_count=count, max_box_area=max_area))

    positives = [s for s in valid_samples if s.object_count > 0]
    negatives = [s for s in valid_samples if s.object_count == 0]

    # prioritize positives with larger objects to make initial learning easier in aerial context
    positives.sort(key=lambda s: s.max_box_area, reverse=True)

    rng = random.Random(args.seed)
    rng.shuffle(negatives)

    positives = positives[: min(args.non_empty_max, len(positives))]
    negatives = negatives[: min(args.empty_max, len(negatives))]

    selected = positives + negatives
    rng.shuffle(selected)

    val_count = max(1, int(len(selected) * args.val_ratio))
    train_count = len(selected) - val_count
    train_samples = selected[:train_count]
    val_samples = selected[train_count:]

    for split in ["train", "val"]:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    def write_split(samples: list[Sample], split: str) -> tuple[int, int]:
        n_objects = 0
        n_empty = 0
        for i, s in enumerate(samples):
            stem = f"{split}_{i:05d}_{s.image_path.stem}"
            out_img = out_root / "images" / split / f"{stem}{s.image_path.suffix.lower()}"
            out_lbl = out_root / "labels" / split / f"{stem}.txt"

            shutil.copy2(s.image_path, out_img)
            lines, obj_count, _ = parse_valid_lines(s.label_path, min_box_area=args.min_box_area)
            out_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            n_objects += obj_count
            if obj_count == 0:
                n_empty += 1
        return n_objects, n_empty

    train_objects, train_empty = write_split(train_samples, "train")
    val_objects, val_empty = write_split(val_samples, "val")

    summary = {
        "subset_name": "yolo_aerial_improved",
        "source_root": str(zenodo_root),
        "output_root": str(out_root),
        "total_label_files_found": len(label_files),
        "valid_samples_found": len(valid_samples),
        "missing_image_count": missing_image,
        "selected_total": len(selected),
        "train_images": len(train_samples),
        "val_images": len(val_samples),
        "train_objects": train_objects,
        "val_objects": val_objects,
        "train_empty_labels": train_empty,
        "val_empty_labels": val_empty,
        "non_empty_max": args.non_empty_max,
        "empty_max": args.empty_max,
        "val_ratio": args.val_ratio,
        "min_box_area": args.min_box_area,
        "seed": args.seed,
    }

    summary_json = reports_dir / "yolo_aerial_improved_subset_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_csv = reports_dir / "yolo_aerial_improved_subset_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(summary.keys()))
        writer.writerow(list(summary.values()))

    md = [
        "# Improved YOLO Aerial Subset",
        "",
        f"- Train images: {summary['train_images']}",
        f"- Val images: {summary['val_images']}",
        f"- Train objects: {summary['train_objects']}",
        f"- Val objects: {summary['val_objects']}",
        f"- Empty labels train/val: {summary['train_empty_labels']} / {summary['val_empty_labels']}",
        f"- Filtering min box area: {summary['min_box_area']}",
        "",
        "This subset prioritizes non-empty samples with larger boxes to improve initial aerial-domain learning.",
    ]
    (reports_dir / "yolo_aerial_improved_subset_summary.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Prepared improved subset: {out_root}")
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")


if __name__ == "__main__":
    main()
