from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v"}
ANNOT_EXTS = {".txt", ".xml", ".json", ".csv", ".yaml", ".yml"}
SPLIT_HINTS = ("train", "val", "valid", "test", "query", "gallery")


@dataclass
class DatasetSummary:
    dataset_name: str
    dataset_path: str
    exists: bool
    total_files: int
    total_dirs: int
    total_size_gb: float
    image_count: int
    video_count: int
    annotation_like_count: int
    has_train_split: bool
    has_val_split: bool
    has_test_split: bool
    has_query_split: bool
    has_gallery_split: bool
    id_candidate_dirs: int
    likely_has_cow_ids: bool
    likely_has_tracklets: bool
    likely_has_multiview: bool
    temporary_file_count: int
    top_extensions: list[tuple[str, int]]
    top_level_dirs: list[str]
    key_notes: list[str]


def inspect_dataset(dataset_name: str, dataset_path: Path) -> DatasetSummary:
    if not dataset_path.exists():
        return DatasetSummary(
            dataset_name=dataset_name,
            dataset_path=str(dataset_path),
            exists=False,
            total_files=0,
            total_dirs=0,
            total_size_gb=0.0,
            image_count=0,
            video_count=0,
            annotation_like_count=0,
            has_train_split=False,
            has_val_split=False,
            has_test_split=False,
            has_query_split=False,
            has_gallery_split=False,
            id_candidate_dirs=0,
            likely_has_cow_ids=False,
            likely_has_tracklets=False,
            likely_has_multiview=False,
            temporary_file_count=0,
            top_extensions=[],
            top_level_dirs=[],
            key_notes=["Dataset path is missing."],
        )

    files = [p for p in dataset_path.rglob("*") if p.is_file()]
    dirs = [p for p in dataset_path.rglob("*") if p.is_dir()]

    ext_counts: dict[str, int] = {}
    for f in files:
        ext = f.suffix.lower() if f.suffix else "<no_ext>"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    image_count = sum(1 for f in files if f.suffix.lower() in IMAGE_EXTS)
    video_count = sum(1 for f in files if f.suffix.lower() in VIDEO_EXTS)
    annotation_count = sum(1 for f in files if f.suffix.lower() in ANNOT_EXTS)
    temporary_file_count = sum(1 for f in files if f.suffix.lower() in {".tmp", ".part", ".crdownload"})
    total_size = sum(f.stat().st_size for f in files)

    split_tokens = {part.lower() for p in dirs for part in p.parts}
    has_train = any("train" in s for s in split_tokens)
    has_val = any(s in {"val", "valid", "validation"} or "val" in s for s in split_tokens)
    has_test = any("test" in s for s in split_tokens)
    has_query = any("query" in s for s in split_tokens)
    has_gallery = any("gallery" in s for s in split_tokens)

    id_dirs = 0
    for d in dirs:
        imgs = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if imgs:
            id_dirs += 1

    likely_has_ids = id_dirs >= 2
    lower_paths = [str(p).lower() for p in dirs]
    likely_tracklets = any("tracklet" in p or "track" in p or "sequence" in p for p in lower_paths)
    likely_multiview = any("cam" in p or "view" in p or "multi" in p for p in lower_paths)

    top_level_dirs = sorted([p.name for p in dataset_path.iterdir() if p.is_dir()])
    top_ext = sorted(ext_counts.items(), key=lambda kv: kv[1], reverse=True)[:12]

    notes: list[str] = []
    if len(files) == 0:
        notes.append("Directory exists but contains no files. Dataset is not extracted or download is incomplete.")
    material_ready = image_count > 0 or video_count > 0 or annotation_count > 0
    if temporary_file_count > 0:
        if material_ready:
            notes.append("Temporary download files detected (.tmp/.part/.crdownload), but extracted dataset content is available.")
        else:
            notes.append("Temporary download files detected (.tmp/.part/.crdownload). Local copy appears incomplete.")
    if image_count == 0:
        notes.append("No image files detected.")
    if video_count == 0:
        notes.append("No video files detected.")
    if not (has_train or has_val or has_test):
        notes.append("No explicit train/val/test split directories detected.")
    if not likely_has_ids:
        notes.append("No strong evidence of per-cow identity folders.")

    return DatasetSummary(
        dataset_name=dataset_name,
        dataset_path=str(dataset_path),
        exists=True,
        total_files=len(files),
        total_dirs=len(dirs),
        total_size_gb=round(total_size / (1024**3), 4),
        image_count=image_count,
        video_count=video_count,
        annotation_like_count=annotation_count,
        has_train_split=has_train,
        has_val_split=has_val,
        has_test_split=has_test,
        has_query_split=has_query,
        has_gallery_split=has_gallery,
        id_candidate_dirs=id_dirs,
        likely_has_cow_ids=likely_has_ids,
        likely_has_tracklets=likely_tracklets,
        likely_has_multiview=likely_multiview,
        temporary_file_count=temporary_file_count,
        top_extensions=top_ext,
        top_level_dirs=top_level_dirs,
        key_notes=notes,
    )


def _utility_label(score: int) -> str:
    return {0: "none", 1: "low", 2: "medium", 3: "high"}.get(score, "unknown")


def score_dataset(summary: DatasetSummary, dataset_name: str) -> dict[str, int | str]:
    # Conservative scoring based on local evidence plus dataset role from project docs.
    if summary.total_files == 0:
        if dataset_name == "opencows2020":
            det, reid, similarity = 0, 2, 1
        else:
            det, reid, similarity = 0, 1, 1
    else:
        det = 2 if summary.image_count > 1000 else 1
        reid = 3 if summary.likely_has_cow_ids else 1
        similarity = 2 if summary.likely_has_multiview else 1

    return {
        "detection_generic_score": det,
        "detection_generic_label": _utility_label(det),
        "reid_score": reid,
        "reid_label": _utility_label(reid),
        "project_similarity_score": similarity,
        "project_similarity_label": _utility_label(similarity),
    }


def write_markdown(summary: DatasetSummary, score: dict[str, int | str], out_path: Path) -> None:
    material_ready = summary.image_count > 0 or summary.video_count > 0 or summary.annotation_like_count > 0
    split_line = (
        f"train={summary.has_train_split}, val={summary.has_val_split}, "
        f"test={summary.has_test_split}, query={summary.has_query_split}, gallery={summary.has_gallery_split}"
    )
    top_ext = ", ".join([f"{k}:{v}" for k, v in summary.top_extensions]) if summary.top_extensions else "none"
    top_dirs = ", ".join(summary.top_level_dirs) if summary.top_level_dirs else "none"
    notes = "\n".join([f"- {n}" for n in summary.key_notes]) if summary.key_notes else "- none"

    text = (
        f"# {summary.dataset_name} inspection summary\n\n"
        f"## Local scan\n"
        f"- Path: {summary.dataset_path}\n"
        f"- Exists: {summary.exists}\n"
        f"- Total files: {summary.total_files}\n"
        f"- Total directories: {summary.total_dirs}\n"
        f"- Total size (GB): {summary.total_size_gb}\n"
        f"- Image files: {summary.image_count}\n"
        f"- Video files: {summary.video_count}\n"
        f"- Annotation-like files: {summary.annotation_like_count}\n"
        f"- Split hints: {split_line}\n"
        f"- Candidate ID folders: {summary.id_candidate_dirs}\n"
        f"- Likely has per-cow IDs: {summary.likely_has_cow_ids}\n"
        f"- Likely has tracklets: {summary.likely_has_tracklets}\n"
        f"- Likely has multi-view setup: {summary.likely_has_multiview}\n"
        f"- Temporary download files: {summary.temporary_file_count}\n"
        f"- Top extensions: {top_ext}\n"
        f"- Top-level dirs: {top_dirs}\n\n"
        f"## Utility diagnosis\n"
        f"- Generic cow detection utility: {score['detection_generic_label']} ({score['detection_generic_score']}/3)\n"
        f"- Individual ID / Re-ID utility: {score['reid_label']} ({score['reid_score']}/3)\n"
        f"- Similarity to final project target: {score['project_similarity_label']} ({score['project_similarity_score']}/3)\n\n"
        f"## Recommendation role\n"
        f"- Bridge dataset for Re-ID: {'yes' if score['reid_score'] >= 2 else 'no'}\n"
        f"- Benchmarking dataset: {'yes' if material_ready else 'not yet (missing extracted files)'}\n"
        f"- Gallery-by-identity source: {'yes' if summary.likely_has_cow_ids else 'not yet (no local identity evidence)'}\n"
        f"- Secondary reference only: {'yes' if not material_ready else 'no'}\n\n"
        f"## Findings and caveats\n"
        f"{notes}\n"
    )
    out_path.write_text(text, encoding="utf-8")


def main() -> None:
    root = Path("data/raw")
    targets = {
        "opencows2020": root / "opencows2020",
        "multicamcows2024": root / "multicamcows2024",
    }
    out_dir = Path("reports/dataset_inspection")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    inventory: dict[str, dict[str, object]] = {}

    for name, path in targets.items():
        summary = inspect_dataset(name, path)
        score = score_dataset(summary, dataset_name=name)

        row = asdict(summary)
        row.update(score)
        row["top_extensions"] = json.dumps(summary.top_extensions)
        row["top_level_dirs"] = json.dumps(summary.top_level_dirs)
        row["key_notes"] = json.dumps(summary.key_notes)
        rows.append(row)

        inventory[name] = {**asdict(summary), **score}
        md_path = out_dir / f"{name}_summary.md"
        write_markdown(summary, score, md_path)

    json_path = out_dir / "dataset_inventory.json"
    json_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    csv_path = out_dir / "dataset_inventory.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
