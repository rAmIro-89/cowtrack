from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


FIELDS = [
    "frames_processed",
    "detections_total",
    "detections_avg_per_frame",
    "valid_track_ids",
    "crops_saved",
    "frames_with_detection_pct",
]


def run_one(
    project_root: Path,
    sequence_dir: Path,
    model_path: str,
    output_dir: Path,
    max_frames: int,
    tracker_cfg: str,
    conf: float,
    iou: float,
    imgsz: int,
    annotated_every_n: int,
    timeout_sec: int,
    device: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "sprint1_track_crops.py"),
        "--input_dir",
        str(sequence_dir),
        "--model",
        model_path,
        "--output_dir",
        str(output_dir),
        "--max_frames",
        str(max_frames),
        "--min_quality",
        "0",
        "--tracker_cfg",
        tracker_cfg,
        "--conf",
        str(conf),
        "--iou",
        str(iou),
        "--imgsz",
        str(imgsz),
        "--device",
        str(device),
        "--save_annotated_frames",
        "--annotated_every_n",
        str(max(1, annotated_every_n)),
    ]

    status = "ok"
    try:
        subprocess.run(cmd, cwd=project_root, check=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        status = "timeout"
    except subprocess.CalledProcessError:
        status = "error"

    summary_json = output_dir / "reports" / "summary.json"
    metrics = {k: 0 for k in FIELDS}
    metrics["detections_avg_per_frame"] = 0.0
    metrics["frames_with_detection_pct"] = 0.0
    if summary_json.exists():
        data = json.loads(summary_json.read_text(encoding="utf-8"))
        for k in FIELDS:
            if k in data:
                metrics[k] = data[k]

    metrics["status"] = status
    metrics["model_path"] = model_path
    metrics["output_dir"] = str(output_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare detector models on winner aerial sequence")
    parser.add_argument("--sequence_dir", default="data/raw/zenodo_grazing_cows/extracted/Cattle_drone_images/Derval/JPGImages/DJI_202308091442_012")
    parser.add_argument("--reports_dir", default="reports")
    parser.add_argument("--output_root", default="data/interim/model_comparison")
    parser.add_argument("--max_frames", type=int, default=400)
    parser.add_argument("--tracker_cfg", default="configs/bytetrack_relaxed_no_gmc.yaml")
    parser.add_argument("--conf", type=float, default=0.08)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--annotated_every_n", type=int, default=15)
    parser.add_argument("--timeout_sec", type=int, default=600)
    parser.add_argument("--device", default="0")
    parser.add_argument("--baseline_model", default="runs/detect/outputs/detector_aerial/yolov8n_aerial_baseline/weights/best.pt")
    parser.add_argument("--candidate_models_csv", default="reports/detector_aerial_experiments.csv")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    sequence_dir = (project_root / args.sequence_dir).resolve()
    reports_dir = (project_root / args.reports_dir).resolve()
    output_root = (project_root / args.output_root).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    candidates = [
        {"name": "base_yolov8n", "model": "yolov8n.pt"},
        {"name": "baseline_prev_ft", "model": str((project_root / args.baseline_model).resolve())},
    ]

    csv_path = (project_root / args.candidate_models_csv).resolve()
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                candidates.append({"name": r["experiment"], "model": r["best_pt"]})

    rows = []
    for c in candidates:
        print(f"Evaluating {c['name']}")
        out_dir = output_root / c["name"]
        m = run_one(
            project_root=project_root,
            sequence_dir=sequence_dir,
            model_path=c["model"],
            output_dir=out_dir,
            max_frames=args.max_frames,
            tracker_cfg=args.tracker_cfg,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            annotated_every_n=args.annotated_every_n,
            timeout_sec=args.timeout_sec,
            device=args.device,
        )
        row = {"model_name": c["name"]}
        row.update(m)
        rows.append(row)

    # baseline for deltas: previous ft
    baseline = next((r for r in rows if r["model_name"] == "baseline_prev_ft"), rows[0])
    for r in rows:
        for k in FIELDS:
            if isinstance(r[k], (int, float)) and isinstance(baseline[k], (int, float)):
                r[f"delta_vs_baseline_{k}"] = r[k] - baseline[k]
            else:
                r[f"delta_vs_baseline_{k}"] = 0

    out_csv = reports_dir / "detector_models_comparative.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda r: (r["detections_avg_per_frame"], r["frames_with_detection_pct"], r["crops_saved"]))

    out_md = reports_dir / "detector_models_comparative.md"
    lines = [
        "# Detector Model Comparison on Winner Sequence",
        "",
        "| model | frames | det_total | det_avg/frame | %frames_det | valid_tracks | crops | delta det_total vs prev baseline |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['model_name']} | {r['frames_processed']} | {r['detections_total']} | {r['detections_avg_per_frame']:.4f} | {r['frames_with_detection_pct']:.2f} | {r['valid_track_ids']} | {r['crops_saved']} | {r['delta_vs_baseline_detections_total']} |"
        )

    lines.extend([
        "",
        "## Best Detector",
        "",
        f"- Selected: {best['model_name']}",
        f"- Model path: {best['model_path']}",
    ])

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved comparative report: {out_csv}")


if __name__ == "__main__":
    main()
