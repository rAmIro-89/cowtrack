from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class Box:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float


def parse_yolo_labels(label_path: Path, img_w: int, img_h: int) -> list[Box]:
    boxes: list[Box] = []
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
            xc, yc, bw, bh = map(float, parts[1:])
        except ValueError:
            continue
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            continue

        x1 = (xc - bw / 2.0) * img_w
        y1 = (yc - bh / 2.0) * img_h
        x2 = (xc + bw / 2.0) * img_w
        y2 = (yc + bh / 2.0) * img_h
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(img_w), x2), min(float(img_h), y2)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(Box(cls=cls, x1=x1, y1=y1, x2=x2, y2=y2))
    return boxes


def tile_positions(size: int, tile_size: int, stride: int) -> list[int]:
    if size <= tile_size:
        return [0]
    pos = list(range(0, max(1, size - tile_size + 1), stride))
    last = size - tile_size
    if pos[-1] != last:
        pos.append(last)
    return pos


def intersection_ratio(a: Box, tx1: int, ty1: int, tx2: int, ty2: int) -> tuple[float, tuple[float, float, float, float] | None]:
    ix1 = max(a.x1, tx1)
    iy1 = max(a.y1, ty1)
    ix2 = min(a.x2, tx2)
    iy2 = min(a.y2, ty2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0, None
    inter = (ix2 - ix1) * (iy2 - iy1)
    area = max(1e-9, (a.x2 - a.x1) * (a.y2 - a.y1))
    return inter / area, (ix1, iy1, ix2, iy2)


def clipped_to_yolo(ix1: float, iy1: float, ix2: float, iy2: float, tx1: int, ty1: int, tile_w: int, tile_h: int) -> tuple[float, float, float, float]:
    cx = ((ix1 + ix2) / 2.0 - tx1) / tile_w
    cy = ((iy1 + iy2) / 2.0 - ty1) / tile_h
    bw = (ix2 - ix1) / tile_w
    bh = (iy2 - iy1) / tile_h
    return cx, cy, bw, bh


def draw_boxes(img, yolo_lines: list[str]) -> None:
    h, w = img.shape[:2]
    for line in yolo_lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = parts[0]
        xc, yc, bw, bh = map(float, parts[1:])
        x1 = int((xc - bw / 2.0) * w)
        y1 = int((yc - bh / 2.0) * h)
        x2 = int((xc + bw / 2.0) * w)
        y2 = int((yc + bh / 2.0) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (40, 220, 80), 2)
        cv2.putText(img, f"c={cls}", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 220, 80), 1)


def build_split(
    input_root: Path,
    output_root: Path,
    split: str,
    tile_size: int,
    overlap: float,
    min_visibility: float,
    negative_ratio: float,
    min_box_area_norm: float,
    seed: int,
) -> dict:
    in_images = input_root / "images" / split
    in_labels = input_root / "labels" / split
    out_images = output_root / "images" / split
    out_labels = output_root / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    stride = max(1, int(tile_size * (1.0 - overlap)))
    rng = random.Random(seed + (11 if split == "val" else 0))

    originals = sorted([p for p in in_images.iterdir() if p.is_file()])
    pos_tiles: list[tuple[Path, list[str], str]] = []
    neg_tiles: list[tuple[Path, list[str], str]] = []

    original_with_objects = 0
    original_without_objects = 0
    original_objects = 0

    for img_path in originals:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        label_path = in_labels / f"{img_path.stem}.txt"
        boxes = parse_yolo_labels(label_path, img_w=w, img_h=h)
        if boxes:
            original_with_objects += 1
        else:
            original_without_objects += 1
        original_objects += len(boxes)

        xs = tile_positions(w, tile_size=tile_size, stride=stride)
        ys = tile_positions(h, tile_size=tile_size, stride=stride)

        for y in ys:
            for x in xs:
                tx1, ty1 = x, y
                tx2, ty2 = min(w, x + tile_size), min(h, y + tile_size)
                tile = img[ty1:ty2, tx1:tx2]
                tile_h, tile_w = tile.shape[:2]
                if tile_h < 8 or tile_w < 8:
                    continue

                lines: list[str] = []
                for b in boxes:
                    vis, clip = intersection_ratio(b, tx1, ty1, tx2, ty2)
                    if vis < min_visibility or clip is None:
                        continue
                    ix1, iy1, ix2, iy2 = clip
                    cx, cy, bw, bh = clipped_to_yolo(ix1, iy1, ix2, iy2, tx1, ty1, tile_w, tile_h)
                    if bw * bh < min_box_area_norm:
                        continue
                    lines.append(f"{b.cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                tile_id = f"{img_path.stem}_x{x}_y{y}"
                if lines:
                    pos_tiles.append((tile.copy(), lines, tile_id))
                else:
                    neg_tiles.append((tile.copy(), lines, tile_id))

    max_negs = int(len(pos_tiles) * negative_ratio)
    if max_negs < 1 and len(pos_tiles) > 0:
        max_negs = 1
    rng.shuffle(neg_tiles)
    neg_kept = neg_tiles[:max_negs] if max_negs > 0 else []

    all_tiles = pos_tiles + neg_kept
    rng.shuffle(all_tiles)

    saved_tiles = 0
    saved_pos = 0
    saved_neg = 0

    for idx, (tile, lines, tid) in enumerate(all_tiles):
        out_name = f"{split}_{idx:06d}_{tid}.jpg"
        out_img = out_images / out_name
        out_lbl = out_labels / f"{Path(out_name).stem}.txt"

        cv2.imwrite(str(out_img), tile)
        out_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        saved_tiles += 1
        if lines:
            saved_pos += 1
        else:
            saved_neg += 1

    return {
        "split": split,
        "original_images": len(originals),
        "original_with_objects": original_with_objects,
        "original_without_objects": original_without_objects,
        "original_objects": original_objects,
        "generated_pos_tiles": len(pos_tiles),
        "generated_neg_tiles": len(neg_tiles),
        "kept_neg_tiles": len(neg_kept),
        "saved_tiles": saved_tiles,
        "saved_pos_tiles": saved_pos,
        "saved_neg_tiles": saved_neg,
        "stride": stride,
    }


def create_preview(output_root: Path, reports_dir: Path, sample_count: int, seed: int) -> int:
    preview_dir = reports_dir / "dataset_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    candidates: list[tuple[Path, Path]] = []
    for split in ["train", "val"]:
        images_dir = output_root / "images" / split
        labels_dir = output_root / "labels" / split
        for img_path in images_dir.glob("*.jpg"):
            lbl_path = labels_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            lines = [ln.strip() for ln in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
            if lines:
                candidates.append((img_path, lbl_path))

    rng.shuffle(candidates)
    candidates = candidates[: min(sample_count, len(candidates))]

    for i, (img_path, lbl_path) in enumerate(candidates):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        lines = [ln.strip() for ln in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        draw_boxes(img, lines)
        out_name = f"preview_{i:03d}_{img_path.name}"
        cv2.imwrite(str(preview_dir / out_name), img)

    return len(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tiled YOLO dataset for small aerial objects")
    parser.add_argument("--input_root", default="data/processed/yolo_aerial_baseline")
    parser.add_argument("--output_root", default="data/processed/yolo_aerial_tiled")
    parser.add_argument("--reports_dir", default="reports/tiling")
    parser.add_argument("--tile_size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--min_visibility", type=float, default=0.35)
    parser.add_argument("--negative_ratio", type=float, default=0.6)
    parser.add_argument("--min_box_area_norm", type=float, default=0.0008)
    parser.add_argument("--preview_count", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    input_root = (project_root / args.input_root).resolve()
    output_root = (project_root / args.output_root).resolve()
    reports_dir = (project_root / args.reports_dir).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input YOLO dataset not found: {input_root}")

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_stats = build_split(
        input_root=input_root,
        output_root=output_root,
        split="train",
        tile_size=args.tile_size,
        overlap=args.overlap,
        min_visibility=args.min_visibility,
        negative_ratio=args.negative_ratio,
        min_box_area_norm=args.min_box_area_norm,
        seed=args.seed,
    )

    val_stats = build_split(
        input_root=input_root,
        output_root=output_root,
        split="val",
        tile_size=args.tile_size,
        overlap=args.overlap,
        min_visibility=args.min_visibility,
        negative_ratio=args.negative_ratio,
        min_box_area_norm=args.min_box_area_norm,
        seed=args.seed,
    )

    preview_saved = create_preview(output_root=output_root, reports_dir=reports_dir, sample_count=args.preview_count, seed=args.seed)

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "min_visibility": args.min_visibility,
        "negative_ratio": args.negative_ratio,
        "min_box_area_norm": args.min_box_area_norm,
        "train_original_images": train_stats["original_images"],
        "val_original_images": val_stats["original_images"],
        "train_tiles_generated_pos": train_stats["generated_pos_tiles"],
        "train_tiles_generated_neg": train_stats["generated_neg_tiles"],
        "train_tiles_kept_neg": train_stats["kept_neg_tiles"],
        "train_tiles_saved": train_stats["saved_tiles"],
        "train_tiles_saved_pos": train_stats["saved_pos_tiles"],
        "train_tiles_saved_neg": train_stats["saved_neg_tiles"],
        "val_tiles_generated_pos": val_stats["generated_pos_tiles"],
        "val_tiles_generated_neg": val_stats["generated_neg_tiles"],
        "val_tiles_kept_neg": val_stats["kept_neg_tiles"],
        "val_tiles_saved": val_stats["saved_tiles"],
        "val_tiles_saved_pos": val_stats["saved_pos_tiles"],
        "val_tiles_saved_neg": val_stats["saved_neg_tiles"],
        "preview_images_saved": preview_saved,
    }

    summary_json = reports_dir / "tiled_subset_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_csv = reports_dir / "tiled_subset_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(summary.keys()))
        writer.writerow(list(summary.values()))

    summary_md = reports_dir / "tiled_subset_summary.md"
    md_lines = [
        "# Tiled YOLO Subset Summary",
        "",
        f"- Input dataset: {summary['input_root']}",
        f"- Output dataset: {summary['output_root']}",
        f"- Tile size: {summary['tile_size']}",
        f"- Overlap: {summary['overlap']}",
        f"- Min visibility: {summary['min_visibility']}",
        f"- Negative ratio kept: {summary['negative_ratio']}",
        "",
        "## Original images",
        f"- Train: {summary['train_original_images']}",
        f"- Val: {summary['val_original_images']}",
        "",
        "## Tiles generated and saved",
        f"- Train tiles saved: {summary['train_tiles_saved']} (pos={summary['train_tiles_saved_pos']}, neg={summary['train_tiles_saved_neg']})",
        f"- Val tiles saved: {summary['val_tiles_saved']} (pos={summary['val_tiles_saved_pos']}, neg={summary['val_tiles_saved_neg']})",
        "",
        f"- Preview images with labels: {summary['preview_images_saved']}",
        f"- Preview folder: {reports_dir / 'dataset_preview'}",
    ]
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved tiled dataset: {output_root}")
    print(f"Saved summary: {summary_csv}")


if __name__ == "__main__":
    main()
