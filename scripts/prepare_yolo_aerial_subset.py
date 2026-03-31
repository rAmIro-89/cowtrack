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


def find_corresponding_image(label_path: Path, zenodo_root: Path) -> Path | None:
    parts = label_path.parts
    if "YOLO_1.1" in parts:
        yolo_idx = parts.index("YOLO_1.1")
    elif "YOLO1.1" in parts:
        yolo_idx = parts.index("YOLO1.1")
    else:
        return None

    # Expected pattern: <site>/<YOLO_X>/<sequence>/obj_train_data/<file>.txt
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


def parse_and_validate_label(label_path: Path) -> tuple[list[str], int]:
    valid_lines: list[str] = []
    for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue

        cls, x, y, w, h = parts
        try:
            _ = int(float(cls))
            xf = float(x)
            yf = float(y)
            wf = float(w)
            hf = float(h)
        except ValueError:
            continue

        if not (0.0 <= xf <= 1.0 and 0.0 <= yf <= 1.0 and 0.0 < wf <= 1.0 and 0.0 < hf <= 1.0):
            continue

        # Single-class detector baseline: map all bovine boxes to class 0.
        valid_lines.append(f"0 {xf:.6f} {yf:.6f} {wf:.6f} {hf:.6f}")

    return valid_lines, len(valid_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare initial YOLO subset for aerial detector fine-tuning")
    parser.add_argument("--zenodo_root", default="data/raw/zenodo_grazing_cows/extracted/Cattle_drone_images")
    parser.add_argument("--out_root", default="data/processed/yolo_aerial_baseline")
    parser.add_argument("--reports_dir", default="reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--non_empty_max", type=int, default=220)
    parser.add_argument("--empty_max", type=int, default=80)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    zenodo_root = (project_root / args.zenodo_root).resolve()
    out_root = (project_root / args.out_root).resolve()
    reports_dir = (project_root / args.reports_dir).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not zenodo_root.exists():
        raise FileNotFoundError(f"Zenodo root not found: {zenodo_root}")

    label_files = [p for p in zenodo_root.rglob("*.txt") if "obj_train_data" in str(p)]
    label_files.sort()

    valid_samples: list[Sample] = []
    missing_image = 0
    invalid_label = 0

    for lbl in label_files:
        img = find_corresponding_image(lbl, zenodo_root)
        if img is None:
            missing_image += 1
            continue

        lines, obj_count = parse_and_validate_label(lbl)
        if obj_count == 0 and lbl.stat().st_size > 0:
            invalid_label += 1

        # Keep both positive and empty labels (negatives are useful for detector).
        valid_samples.append(Sample(image_path=img, label_path=lbl, object_count=obj_count))

    positives = [s for s in valid_samples if s.object_count > 0]
    negatives = [s for s in valid_samples if s.object_count == 0]

    rng = random.Random(args.seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    positives = positives[: min(args.non_empty_max, len(positives))]
    negatives = negatives[: min(args.empty_max, len(negatives))]

    selected = positives + negatives
    rng.shuffle(selected)

    if not selected:
        raise RuntimeError("No valid samples selected for subset")

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
            lines, obj_count = parse_and_validate_label(s.label_path)
            out_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            n_objects += obj_count
            if obj_count == 0:
                n_empty += 1
        return n_objects, n_empty

    train_objects, train_empty = write_split(train_samples, "train")
    val_objects, val_empty = write_split(val_samples, "val")

    summary = {
        "zenodo_root": str(zenodo_root),
        "output_root": str(out_root),
        "total_label_files_found": len(label_files),
        "valid_samples_found": len(valid_samples),
        "missing_image_count": missing_image,
        "invalid_label_count": invalid_label,
        "selected_total": len(selected),
        "train_images": len(train_samples),
        "val_images": len(val_samples),
        "train_objects": train_objects,
        "val_objects": val_objects,
        "train_empty_labels": train_empty,
        "val_empty_labels": val_empty,
        "selection_non_empty_max": args.non_empty_max,
        "selection_empty_max": args.empty_max,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
    }

    summary_json = reports_dir / "yolo_aerial_subset_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_csv = reports_dir / "yolo_aerial_subset_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(summary.keys()))
        writer.writerow(list(summary.values()))

    md_lines = [
        "# YOLO Aerial Subset Summary",
        "",
        f"- Train images: {summary['train_images']}",
        f"- Val images: {summary['val_images']}",
        f"- Train objects: {summary['train_objects']}",
        f"- Val objects: {summary['val_objects']}",
        f"- Empty labels (train/val): {summary['train_empty_labels']} / {summary['val_empty_labels']}",
        "",
        "## Notes",
        "",
        "- Labels were validated for YOLO format and normalized coordinates.",
        "- A single-class detector baseline is used (all bovine boxes mapped to class 0).",
    ]
    if summary["train_empty_labels"] + summary["val_empty_labels"] > (0.5 * summary["selected_total"]):
        md_lines.append("- Warning: subset has a high proportion of empty labels (potential imbalance).")

    summary_md = reports_dir / "yolo_aerial_subset_summary.md"
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Prepared subset at: {out_root}")
    print(f"Train images: {len(train_samples)} | Val images: {len(val_samples)}")
    print(f"Summary: {summary_md}")


if __name__ == "__main__":
    main()
