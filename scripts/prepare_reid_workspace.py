from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_HINTS = ("train", "val", "valid", "test", "query", "gallery")


@dataclass
class IdentityRecord:
    identity: str
    image_path: Path
    split: str


def _looks_like_split(token: str) -> bool:
    t = token.lower()
    return any(h == t or h in t for h in SPLIT_HINTS)


def _infer_split(path: Path) -> str:
    for part in path.parts:
        if _looks_like_split(part):
            return part.lower()
    return "unspecified"


def _find_identity_records(dataset_root: Path) -> list[IdentityRecord]:
    records: list[IdentityRecord] = []
    if not dataset_root.exists():
        return records

    images = [
        p
        for p in dataset_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    for img in images:
        parent = img.parent
        identity = parent.name.strip() or "unknown_id"
        if _looks_like_split(identity) and parent.parent != dataset_root:
            identity = parent.parent.name.strip() or "unknown_id"
        records.append(
            IdentityRecord(
                identity=identity,
                image_path=img,
                split=_infer_split(img.relative_to(dataset_root)),
            )
        )
    return records


def _copy_samples(
    grouped: dict[str, list[IdentityRecord]],
    target_root: Path,
    per_identity: int,
    max_ids: int | None = None,
) -> int:
    target_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    for idx, (identity, samples) in enumerate(sorted(grouped.items())):
        if max_ids is not None and idx >= max_ids:
            break
        id_dir = target_root / identity
        id_dir.mkdir(parents=True, exist_ok=True)
        for sample in samples[:per_identity]:
            dst = id_dir / sample.image_path.name
            if not dst.exists():
                shutil.copy2(sample.image_path, dst)
                copied += 1
    return copied


def _write_id_map(records: list[IdentityRecord], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[IdentityRecord]] = defaultdict(list)
    for r in records:
        grouped[r.identity].append(r)

    ordered = sorted(grouped.keys())
    id_map = {
        identity: {
            "label_idx": idx,
            "num_images": len(grouped[identity]),
            "splits": sorted({r.split for r in grouped[identity]}),
        }
        for idx, identity in enumerate(ordered)
    }

    json_path = out_dir / "id_map.json"
    json_path.write_text(json.dumps(id_map, indent=2), encoding="utf-8")

    csv_path = out_dir / "id_map.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["identity", "label_idx", "num_images", "splits"])
        for identity, meta in id_map.items():
            writer.writerow(
                [identity, meta["label_idx"], meta["num_images"], "|".join(meta["splits"])]
            )

    return json_path, csv_path


def _build_fallback_subset(fallback_glob: str, out_subset_root: Path) -> int:
    files = sorted(Path().glob(fallback_glob))
    if not files:
        return 0

    pseudo_ids = ["demo_id_000", "demo_id_001"]
    out_subset_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    for idx, src in enumerate(files[:8]):
        identity = pseudo_ids[idx % len(pseudo_ids)]
        dst_dir = out_subset_root / identity
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ReID workspace for identity datasets")
    parser.add_argument("--dataset_name", default="opencows2020")
    parser.add_argument("--dataset_root", default="data/raw/opencows2020")
    parser.add_argument("--output_root", default="data/reid/opencows2020")
    parser.add_argument(
        "--fallback_glob",
        default="reports/tiling_smallobj/dataset_preview/*.jpg",
        help="Fallback images to create a tiny debug subset when identity data is missing",
    )
    parser.add_argument(
        "--allow_fallback",
        action="store_true",
        help="Allow fallback placeholder subset when no identity records are found",
    )
    parser.add_argument("--gallery_per_id", type=int, default=2)
    parser.add_argument("--subset_per_id", type=int, default=3)
    parser.add_argument("--subset_max_ids", type=int, default=5)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    manifests_dir = output_root / "manifests"
    gallery_dir = output_root / "gallery"
    subset_dir = output_root / "subset_debug"
    reports_dir = Path("reports/dataset_inspection")
    reports_dir.mkdir(parents=True, exist_ok=True)

    records = _find_identity_records(dataset_root)
    grouped: dict[str, list[IdentityRecord]] = defaultdict(list)
    for r in records:
        grouped[r.identity].append(r)

    id_json_path: Path | None = None
    id_csv_path: Path | None = None
    gallery_copied = 0
    subset_copied = 0
    fallback_used = False

    id_json_path, id_csv_path = _write_id_map(records, manifests_dir)

    if records:
        gallery_copied = _copy_samples(grouped, gallery_dir, per_identity=args.gallery_per_id)
        subset_copied = _copy_samples(
            grouped,
            subset_dir,
            per_identity=args.subset_per_id,
            max_ids=args.subset_max_ids,
        )
    else:
        if args.allow_fallback:
            fallback_used = True
            subset_copied = _build_fallback_subset(args.fallback_glob, subset_dir)

    manifest_path = manifests_dir / "dataset_manifest.csv"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["identity", "split", "image_path"])
        for r in records:
            writer.writerow([r.identity, r.split, str(r.image_path)])

    prep_summary = {
        "dataset_name": args.dataset_name,
        "dataset_root": str(dataset_root.resolve()) if dataset_root.exists() else str(dataset_root),
        "records_found": len(records),
        "identities_found": len(grouped),
        "id_map_json": str(id_json_path) if id_json_path else None,
        "id_map_csv": str(id_csv_path) if id_csv_path else None,
        "dataset_manifest_csv": str(manifest_path),
        "gallery_dir": str(gallery_dir),
        "subset_debug_dir": str(subset_dir),
        "gallery_images_copied": gallery_copied,
        "subset_images_copied": subset_copied,
        "gallery_total_images": len(list(gallery_dir.rglob("*.jpg"))) + len(list(gallery_dir.rglob("*.png"))),
        "subset_total_images": len(list(subset_dir.rglob("*.jpg"))) + len(list(subset_dir.rglob("*.png"))),
        "fallback_used": fallback_used,
        "fallback_glob": args.fallback_glob,
    }

    summary_path = reports_dir / f"{args.dataset_name}_reid_prep_summary.json"
    summary_path.write_text(json.dumps(prep_summary, indent=2), encoding="utf-8")
    print(json.dumps(prep_summary, indent=2))


if __name__ == "__main__":
    main()
