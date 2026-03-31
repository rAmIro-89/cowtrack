from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def run_window(
    project_root: Path,
    sequence_dir: Path,
    window: int,
    output_dir: Path,
    model_path: str,
    tracker_cfg: str,
    conf: float,
    iou: float,
    imgsz: int,
    timeout_sec: int,
    device: str,
) -> dict[str, str | int | float]:
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
        str(window),
        "--min_quality",
        "0",
        "--save_annotated_frames",
        "--annotated_every_n",
        "1",
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
    ]
    status = "ok"
    try:
        subprocess.run(cmd, cwd=project_root, check=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        status = "timeout"
    except subprocess.CalledProcessError:
        status = "error"

    summary_csv = output_dir / "reports" / "summary.csv"
    if not summary_csv.exists():
        return {
            "window_requested": window,
            "frames_processed": 0,
            "detections_total": 0,
            "detections_avg_per_frame": 0.0,
            "valid_track_ids": 0,
            "crops_saved": 0,
            "frames_with_detection_pct": 0.0,
            "output_dir": str(output_dir),
            "status": status,
        }

    with summary_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {
            "window_requested": window,
            "frames_processed": 0,
            "detections_total": 0,
            "detections_avg_per_frame": 0.0,
            "valid_track_ids": 0,
            "crops_saved": 0,
            "frames_with_detection_pct": 0.0,
            "output_dir": str(output_dir),
            "status": status,
        }

    row = rows[0]
    return {
        "window_requested": window,
        "frames_processed": int(float(row["frames_processed"])),
        "detections_total": int(float(row["detections_total"])),
        "detections_avg_per_frame": float(row["detections_avg_per_frame"]),
        "valid_track_ids": int(float(row["valid_track_ids"])),
        "crops_saved": int(float(row["crops_saved"])),
        "frames_with_detection_pct": float(row["frames_with_detection_pct"]),
        "output_dir": str(output_dir),
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate winner sequence across larger windows")
    parser.add_argument(
        "--sequence_dir",
        default="data/raw/zenodo_grazing_cows/extracted/Cattle_drone_images/Derval/JPGImages/DJI_202308091442_012",
    )
    parser.add_argument("--windows", nargs="+", type=int, default=[100, 200, 400])
    parser.add_argument("--reports_dir", default="reports")
    parser.add_argument("--output_root", default="data/interim/winner_windows")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--tracker_cfg", default="configs/bytetrack_relaxed_no_gmc.yaml")
    parser.add_argument("--conf", type=float, default=0.08)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", default="0")
    parser.add_argument("--timeout_sec", type=int, default=600)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    sequence_dir = (project_root / args.sequence_dir).resolve()
    reports_dir = (project_root / args.reports_dir).resolve()
    output_root = (project_root / args.output_root).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    if not sequence_dir.exists():
        raise FileNotFoundError(f"Sequence not found: {sequence_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}
    total_available = len([p for p in sequence_dir.iterdir() if p.is_file() and p.suffix in exts])

    rows: list[dict[str, str | int | float]] = []
    for window in args.windows:
        out_dir = output_root / f"window_{window}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Running window={window} (available={total_available})")
        result = run_window(
            project_root=project_root,
            sequence_dir=sequence_dir,
            window=window,
            output_dir=out_dir,
            model_path=args.model,
            tracker_cfg=args.tracker_cfg,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            timeout_sec=args.timeout_sec,
        )
        result["sequence_name"] = sequence_dir.name
        result["model_path"] = args.model
        result["total_frames_available"] = total_available
        rows.append(result)

    comparative_csv = reports_dir / "winner_window_comparative.csv"
    with comparative_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sequence_name",
            "window_requested",
            "model_path",
            "total_frames_available",
            "frames_processed",
            "detections_total",
            "detections_avg_per_frame",
            "valid_track_ids",
            "crops_saved",
            "frames_with_detection_pct",
            "status",
            "output_dir",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_path = reports_dir / "winner_window_comparative.md"
    md_lines = [
        "# Winner Sequence Window Comparison",
        "",
        f"Sequence: {sequence_dir.name}",
        "",
        "| frames solicitados | frames procesados | detecciones totales | detecciones promedio/frame | track_id validos | crops guardados | % frames con deteccion |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['window_requested']} | {r['frames_processed']} | {r['detections_total']} | {r['detections_avg_per_frame']:.4f} | {r['valid_track_ids']} | {r['crops_saved']} | {r['frames_with_detection_pct']:.2f} |"
        )

    md_lines.extend([
        "",
        "Estado por ventana:",
    ])
    for r in rows:
        md_lines.append(f"- ventana {r['window_requested']}: {r['status']}")

    md_lines.extend(
        [
            "",
            "## Interpretacion rapida",
            "",
            "Si las metricas no mejoran al aumentar ventana, el cuello principal probablemente sea adaptacion de detector al dominio aereo y no solo longitud temporal.",
        ]
    )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved comparative CSV: {comparative_csv}")
    print(f"Saved comparative MD: {md_path}")


if __name__ == "__main__":
    main()
