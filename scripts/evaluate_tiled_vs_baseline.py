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


def run_eval(
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
    metrics["output_dir"] = str(output_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tiled detectors and compare to previous baseline")
    parser.add_argument("--sequence_dir", default="data/raw/zenodo_grazing_cows/extracted/Cattle_drone_images/Derval/JPGImages/DJI_202308091442_012")
    parser.add_argument("--reports_dir", default="reports/tiling")
    parser.add_argument("--output_root", default="data/interim/tiling_model_comparison")
    parser.add_argument("--max_frames", type=int, default=400)
    parser.add_argument("--tracker_cfg", default="configs/bytetrack_relaxed_no_gmc.yaml")
    parser.add_argument("--conf", type=float, default=0.08)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--annotated_every_n", type=int, default=20)
    parser.add_argument("--device", default="0")
    parser.add_argument("--timeout_sec", type=int, default=900)
    parser.add_argument("--tiled_experiments_csv", default="reports/tiling/tiled_experiments.csv")
    parser.add_argument("--baseline_report_csv", default="reports/overnight/final_comparative.csv")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    sequence_dir = (project_root / args.sequence_dir).resolve()
    reports_dir = (project_root / args.reports_dir).resolve()
    output_root = (project_root / args.output_root).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    if not sequence_dir.exists():
        raise FileNotFoundError(f"Sequence not found: {sequence_dir}")

    baseline_values = {k: 0.0 for k in FIELDS}
    baseline_label = "baseline_prev_ft"
    baseline_csv = (project_root / args.baseline_report_csv).resolve()
    if baseline_csv.exists():
        with baseline_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        row = next((r for r in rows if r.get("model_name") == baseline_label), rows[0] if rows else None)
        if row is not None:
            for k in FIELDS:
                try:
                    baseline_values[k] = float(row.get(k, 0))
                except ValueError:
                    baseline_values[k] = 0.0

    models: list[dict] = [
        {
            "experiment": "baseline_prev_ft",
            "dataset": "reports/overnight/final_comparative.csv",
            "tile_size": "n/a",
            "overlap": "n/a",
            "base_model": "yolov8n_pt_finetuned_previous",
            "epochs": "n/a",
            "imgsz": "n/a",
            "batch": "n/a",
            "device": "n/a",
            "best_pt": str((project_root / "runs/detect/outputs/detector_aerial/yolov8n_aerial_baseline/weights/best.pt").resolve()),
        }
    ]

    tiled_csv = (project_root / args.tiled_experiments_csv).resolve()
    if not tiled_csv.exists():
        raise FileNotFoundError(f"Tiled experiments CSV not found: {tiled_csv}")

    with tiled_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    models.extend(rows)

    out_rows = []
    for m in models:
        name = m["experiment"]
        model_path = m["best_pt"]
        print(f"Evaluating {name}")

        eval_out = output_root / name
        metrics = run_eval(
            project_root=project_root,
            sequence_dir=sequence_dir,
            model_path=model_path,
            output_dir=eval_out,
            max_frames=args.max_frames,
            tracker_cfg=args.tracker_cfg,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            annotated_every_n=args.annotated_every_n,
            timeout_sec=args.timeout_sec,
            device=args.device,
        )

        row = {
            "model": name,
            "dataset_used": m.get("dataset", "n/a"),
            "tile_size": m.get("tile_size", "n/a"),
            "overlap": m.get("overlap", "n/a"),
            "train_imgsz": m.get("imgsz", "n/a"),
            "epochs": m.get("epochs", "n/a"),
            "batch": m.get("batch", "n/a"),
            "device": m.get("device", "n/a"),
            "model_path": model_path,
            "eval_status": metrics["status"],
            "frames_processed": metrics["frames_processed"],
            "detections_total": metrics["detections_total"],
            "detections_avg_per_frame": metrics["detections_avg_per_frame"],
            "frames_with_detection_pct": metrics["frames_with_detection_pct"],
            "valid_track_ids": metrics["valid_track_ids"],
            "crops_saved": metrics["crops_saved"],
            "output_dir": metrics["output_dir"],
        }

        for k in FIELDS:
            v = float(row[k]) if isinstance(row[k], (int, float)) else 0.0
            row[f"delta_vs_baseline_{k}"] = v - baseline_values[k]

        out_rows.append(row)

    out_csv = reports_dir / "tiled_vs_baseline_comparative.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    best = max(out_rows, key=lambda r: (float(r["detections_avg_per_frame"]), float(r["frames_with_detection_pct"]), float(r["crops_saved"])))
    improved = any(float(r["delta_vs_baseline_detections_total"]) > 0 or float(r["delta_vs_baseline_frames_with_detection_pct"]) > 0 for r in out_rows if r["model"] != "baseline_prev_ft")

    if improved:
        conclusion = "Tiling mejora parcialmente la deteccion y ya conviene reabrir iteracion de tracking con el mejor detector tileado."
    else:
        conclusion = "Tiling en esta primera corrida no mejora aun; el siguiente cuello principal sigue siendo calidad/cantidad de etiquetas y escala extrema del objeto."

    out_md = reports_dir / "tiled_vs_baseline_comparative.md"
    lines = [
        "# Tiled vs Baseline Detector Comparison",
        "",
        "| modelo | tile size | overlap | train imgsz | epochs | detecciones totales | det/frame | %frames con deteccion | tracks validos | crops | delta detecciones vs baseline |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in out_rows:
        lines.append(
            f"| {r['model']} | {r['tile_size']} | {r['overlap']} | {r['train_imgsz']} | {r['epochs']} | {r['detections_total']} | {float(r['detections_avg_per_frame']):.4f} | {float(r['frames_with_detection_pct']):.2f} | {r['valid_track_ids']} | {r['crops_saved']} | {float(r['delta_vs_baseline_detections_total']):.2f} |"
        )

    lines.extend([
        "",
        "## Conclusion",
        "",
        f"- Best model: {best['model']}",
        f"- Best model path: {best['model_path']}",
        f"- Assessment: {conclusion}",
    ])
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved comparative CSV: {out_csv}")
    print(f"Saved comparative MD: {out_md}")


if __name__ == "__main__":
    main()
