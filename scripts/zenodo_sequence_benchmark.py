from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}


@dataclass
class ProbeMetrics:
    sequence_name: str
    sequence_path: str
    frames_probed: int
    detections_total: int
    detections_avg_per_frame: float
    frames_with_detection_pct: float
    continuity_score: float


@dataclass
class RunMetrics:
    sequence_name: str
    config_name: str
    frames_processed: int
    detections_total: int
    detections_avg_per_frame: float
    valid_track_ids: int
    crops_saved: int
    frames_with_detection_pct: float
    output_dir: str
    status: str


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", text).strip("_").lower()


def df_to_markdown(df: pd.DataFrame, columns: list[str]) -> str:
    rows = [columns]
    for _, r in df.iterrows():
        rows.append([str(r[c]) for c in columns])

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join([header, sep] + body)


def collect_sequence_dirs(root: Path, min_frames: int) -> list[Path]:
    image_parents: dict[Path, int] = {}
    for img in root.rglob("*"):
        if img.is_file() and img.suffix in IMAGE_EXTS:
            image_parents[img.parent] = image_parents.get(img.parent, 0) + 1

    seq_dirs = [p for p, count in image_parents.items() if count >= min_frames]
    seq_dirs.sort()
    return seq_dirs


def iter_frames(seq_dir: Path, limit: int | None = None, stride: int = 1) -> Iterable[Path]:
    frames = [p for p in seq_dir.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS]
    frames.sort()
    if stride > 1:
        frames = frames[::stride]
    if limit and limit > 0:
        frames = frames[:limit]
    return frames


def probe_sequence(model: YOLO, seq_dir: Path, probe_frames: int, probe_stride: int, conf: float) -> ProbeMetrics:
    frame_paths = list(iter_frames(seq_dir, limit=probe_frames, stride=probe_stride))
    if not frame_paths:
        return ProbeMetrics(seq_dir.name, str(seq_dir), 0, 0, 0.0, 0.0, 0.0)

    per_frame_det: list[int] = []
    for frame_path in frame_paths:
        results = model.predict(source=str(frame_path), classes=[19], conf=conf, device=0, verbose=False)
        det_count = 0
        if results and results[0].boxes is not None and results[0].boxes.xyxy is not None:
            det_count = len(results[0].boxes)
        per_frame_det.append(det_count)

    frames_probed = len(per_frame_det)
    detections_total = sum(per_frame_det)
    frames_with_det = sum(1 for d in per_frame_det if d > 0)
    frames_with_detection_pct = (frames_with_det / frames_probed) * 100.0 if frames_probed else 0.0

    if frames_probed > 1:
        continuity_pairs = sum(1 for i in range(1, frames_probed) if per_frame_det[i] > 0 and per_frame_det[i - 1] > 0)
        continuity_score = continuity_pairs / (frames_probed - 1)
    else:
        continuity_score = 0.0

    return ProbeMetrics(
        sequence_name=seq_dir.name,
        sequence_path=str(seq_dir),
        frames_probed=frames_probed,
        detections_total=detections_total,
        detections_avg_per_frame=(detections_total / frames_probed) if frames_probed else 0.0,
        frames_with_detection_pct=frames_with_detection_pct,
        continuity_score=continuity_score,
    )


def run_sprint1(
    project_root: Path,
    seq_dir: Path,
    output_dir: Path,
    max_frames: int,
    min_quality: float,
    conf: float,
    iou: float,
    tracker_cfg: str,
    config_name: str,
    timeout_sec: int,
    imgsz: int,
    device: str,
) -> RunMetrics:
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "sprint1_track_crops.py"),
        "--input_dir",
        str(seq_dir),
        "--output_dir",
        str(output_dir),
        "--max_frames",
        str(max_frames),
        "--min_quality",
        str(min_quality),
        "--conf",
        str(conf),
        "--iou",
        str(iou),
        "--tracker_cfg",
        tracker_cfg,
        "--imgsz",
        str(imgsz),
        "--device",
        str(device),
    ]
    status = "ok"
    try:
        subprocess.run(cmd, cwd=project_root, check=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        status = "timeout"
    except subprocess.CalledProcessError:
        status = "error"

    csv_path = output_dir / "reports" / "detections_tracks.csv"
    rows: list[dict[str, str]] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    frame_paths = list(iter_frames(seq_dir, limit=max_frames, stride=1))
    frames_processed = len(frame_paths)
    detections_total = len(rows)

    frame_indices = set()
    valid_track_ids = set()
    for row in rows:
        try:
            frame_indices.add(int(row["frame_index"]))
        except Exception:
            pass
        try:
            track_id = int(float(row["track_id"]))
            if track_id >= 0:
                valid_track_ids.add(track_id)
        except Exception:
            pass

    crops_saved = 0
    crops_root = output_dir / "crops_by_track"
    if crops_root.exists():
        crops_saved = sum(1 for _ in crops_root.rglob("*") if _.is_file())

    frames_with_detection_pct = 0.0
    if frames_processed > 0:
        frames_with_detection_pct = (len(frame_indices) / frames_processed) * 100.0

    return RunMetrics(
        sequence_name=seq_dir.name,
        config_name=config_name,
        frames_processed=frames_processed,
        detections_total=detections_total,
        detections_avg_per_frame=(detections_total / frames_processed) if frames_processed else 0.0,
        valid_track_ids=len(valid_track_ids),
        crops_saved=crops_saved,
        frames_with_detection_pct=frames_with_detection_pct,
        output_dir=str(output_dir),
        status=status,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-select top Zenodo sequences and benchmark Sprint 1")
    parser.add_argument("--zenodo_root", default="data/raw/zenodo_grazing_cows/extracted/Cattle_drone_images")
    parser.add_argument("--reports_dir", default="reports")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--min_frames_per_sequence", type=int, default=60)
    parser.add_argument("--probe_frames", type=int, default=80)
    parser.add_argument("--probe_stride", type=int, default=1)
    parser.add_argument("--probe_conf", type=float, default=0.08)
    parser.add_argument("--top_k_sequences", type=int, default=3)
    parser.add_argument("--run_max_frames", type=int, default=60)
    parser.add_argument("--run_timeout_sec", type=int, default=600)
    parser.add_argument("--run_imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    zenodo_root = (project_root / args.zenodo_root).resolve()
    reports_dir = (project_root / args.reports_dir).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not zenodo_root.exists():
        raise FileNotFoundError(f"Zenodo root not found: {zenodo_root}")

    print("Collecting candidate sequences...")
    seq_dirs = collect_sequence_dirs(zenodo_root, min_frames=args.min_frames_per_sequence)
    if not seq_dirs:
        raise RuntimeError("No sequence directories found with enough frames")
    print(f"Found {len(seq_dirs)} candidate sequences")

    print("Probing sequence quality with YOLO detections...")
    model = YOLO(args.model)
    probe_results: list[ProbeMetrics] = []
    for i, seq_dir in enumerate(seq_dirs, start=1):
        metrics = probe_sequence(
            model=model,
            seq_dir=seq_dir,
            probe_frames=args.probe_frames,
            probe_stride=args.probe_stride,
            conf=args.probe_conf,
        )
        probe_results.append(metrics)
        if i % 10 == 0 or i == len(seq_dirs):
            print(f"Probed {i}/{len(seq_dirs)} sequences")

    probe_df = pd.DataFrame([m.__dict__ for m in probe_results])
    probe_df = probe_df.sort_values(
        by=["detections_avg_per_frame", "continuity_score", "frames_with_detection_pct"],
        ascending=[False, False, False],
    )
    ranking_csv = reports_dir / "zenodo_sequence_ranking.csv"
    probe_df.to_csv(ranking_csv, index=False)

    selected = probe_df.head(args.top_k_sequences)
    selected_csv = reports_dir / "zenodo_top_sequences.csv"
    selected.to_csv(selected_csv, index=False)

    run_rows: list[RunMetrics] = []
    configs = [
        {
            "name": "current",
            "conf": 0.2,
            "iou": 0.45,
            "tracker_cfg": "configs/bytetrack_no_gmc.yaml",
        },
        {
            "name": "lowconf_relaxed",
            "conf": 0.08,
            "iou": 0.45,
            "tracker_cfg": "configs/bytetrack_relaxed_no_gmc.yaml",
        },
    ]

    print("Running Sprint 1 benchmark on selected sequences...")
    for _, row in selected.iterrows():
        seq_path = Path(row["sequence_path"])
        seq_slug = _slug(row["sequence_name"])
        for cfg in configs:
            out_dir = project_root / "data" / "interim" / "sprint1_benchmark" / seq_slug / cfg["name"]
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"Running {seq_path.name} | {cfg['name']}")
            run_metrics = run_sprint1(
                project_root=project_root,
                seq_dir=seq_path,
                output_dir=out_dir,
                max_frames=args.run_max_frames,
                min_quality=0.0,
                conf=cfg["conf"],
                iou=cfg["iou"],
                tracker_cfg=cfg["tracker_cfg"],
                config_name=cfg["name"],
                timeout_sec=args.run_timeout_sec,
                imgsz=args.run_imgsz,
                device=args.device,
            )
            run_rows.append(run_metrics)

    run_df = pd.DataFrame([r.__dict__ for r in run_rows])
    run_df = run_df.sort_values(by=["detections_avg_per_frame", "frames_with_detection_pct", "valid_track_ids"], ascending=[False, False, False])
    comparative_csv = reports_dir / "zenodo_sprint1_comparative.csv"
    run_df.to_csv(comparative_csv, index=False)

    score_df = run_df[run_df["status"] == "ok"].copy()
    if score_df.empty:
        raise RuntimeError("No successful sprint runs were produced. Try lower run_max_frames or higher timeout.")
    score_df["score"] = (
        score_df["detections_avg_per_frame"] * 0.5
        + (score_df["frames_with_detection_pct"] / 100.0) * 0.3
        + (score_df["valid_track_ids"].clip(upper=20) / 20.0) * 0.2
    )
    score_df = score_df.sort_values(by=["score"], ascending=False)

    best = score_df.iloc[0]

    md_lines = [
        "# Zenodo Sprint 1 Comparative Report",
        "",
        "## Top 3 Sequences Selected Automatically",
        "",
        df_to_markdown(
            selected,
            [
                "sequence_name",
                "frames_probed",
                "detections_total",
                "detections_avg_per_frame",
                "frames_with_detection_pct",
                "continuity_score",
            ],
        ),
        "",
        "## Comparative Results (2 configs per sequence)",
        "",
        df_to_markdown(
            run_df,
            [
                "sequence_name",
                "config_name",
                "status",
                "frames_processed",
                "detections_total",
                "detections_avg_per_frame",
                "valid_track_ids",
                "crops_saved",
                "frames_with_detection_pct",
            ],
        ),
        "",
        "## Recommended Sequence",
        "",
        f"Best candidate: **{best['sequence_name']}** with config **{best['config_name']}**.",
        "",
        "Selection rationale: highest combined score using detections/frame, frame coverage with detections, and valid track IDs.",
    ]

    comparative_md = reports_dir / "zenodo_sprint1_comparative.md"
    comparative_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved ranking: {ranking_csv}")
    print(f"Saved top sequences: {selected_csv}")
    print(f"Saved comparative CSV: {comparative_csv}")
    print(f"Saved comparative MD: {comparative_md}")
    print(f"Recommended best: {best['sequence_name']} | {best['config_name']}")


if __name__ == "__main__":
    main()
