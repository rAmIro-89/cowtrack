from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


FIELDS = [
    "frames_processed",
    "detections_total",
    "detections_avg_per_frame",
    "valid_track_ids",
    "crops_saved",
    "frames_with_detection_pct",
]


def run_sprint(
    project_root: Path,
    sequence_dir: Path,
    model_path: str,
    output_dir: Path,
    window: int,
    tracker_cfg: str,
    conf: float,
    iou: float,
    imgsz: int,
    timeout_sec: int,
    save_annotated: bool,
    annotated_every_n: int,
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
        str(window),
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
    ]

    if save_annotated:
        cmd += ["--save_annotated_frames", "--annotated_every_n", str(max(1, annotated_every_n))]

    status = "ok"
    try:
        subprocess.run(cmd, cwd=project_root, check=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        status = "timeout"
    except subprocess.CalledProcessError:
        status = "error"

    summary_json = output_dir / "reports" / "summary.json"
    metrics = {
        "frames_processed": 0,
        "detections_total": 0,
        "detections_avg_per_frame": 0.0,
        "valid_track_ids": 0,
        "crops_saved": 0,
        "frames_with_detection_pct": 0.0,
    }

    if summary_json.exists():
        data = json.loads(summary_json.read_text(encoding="utf-8"))
        for k in metrics:
            if k in data:
                metrics[k] = data[k]

    metrics["status"] = status
    metrics["model_path"] = model_path
    metrics["run_output_dir"] = str(output_dir)
    return metrics


def write_metrics_files(target_dir: Path, prefix: str, metrics: dict) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / f"metrics_{prefix}.json"
    csv_path = target_dir / f"metrics_{prefix}.csv"

    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def copy_sample_frames(before_dir: Path, after_dir: Path, sample_dir: Path, max_samples: int) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)

    def _copy(src_dir: Path, prefix: str) -> int:
        ann_dir = src_dir / "annotated_frames"
        if not ann_dir.exists():
            return 0
        files = sorted([p for p in ann_dir.iterdir() if p.is_file()])[:max_samples]
        for i, p in enumerate(files):
            out_name = f"{prefix}_{i:03d}_{p.name}"
            shutil.copy2(p, sample_dir / out_name)
        return len(files)

    copied_before = _copy(before_dir, "before")
    copied_after = _copy(after_dir, "after")

    note = sample_dir / "README.txt"
    note.write_text(
        f"sample frames copied: before={copied_before}, after={copied_after}\n",
        encoding="utf-8",
    )


def md_table_row(window: int, before: dict, after: dict) -> str:
    return (
        f"| {window} | {before['frames_processed']} | {after['frames_processed']} | "
        f"{before['detections_total']} | {after['detections_total']} | {after['detections_total'] - before['detections_total']} | "
        f"{before['detections_avg_per_frame']:.4f} | {after['detections_avg_per_frame']:.4f} | "
        f"{after['detections_avg_per_frame'] - before['detections_avg_per_frame']:.4f} | "
        f"{before['valid_track_ids']} | {after['valid_track_ids']} | {after['valid_track_ids'] - before['valid_track_ids']} | "
        f"{before['crops_saved']} | {after['crops_saved']} | {after['crops_saved'] - before['crops_saved']} | "
        f"{before['frames_with_detection_pct']:.2f} | {after['frames_with_detection_pct']:.2f} | "
        f"{after['frames_with_detection_pct'] - before['frames_with_detection_pct']:.2f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust before/after comparison runner")
    parser.add_argument(
        "--sequence_dir",
        default="data/raw/zenodo_grazing_cows/extracted/Cattle_drone_images/Derval/JPGImages/DJI_202308091442_012",
    )
    parser.add_argument("--windows", nargs="+", type=int, default=[100, 200, 400])
    parser.add_argument("--before_model", default="yolov8n.pt")
    parser.add_argument("--after_model", default="runs/detect/outputs/detector_aerial/yolov8n_aerial_baseline/weights/best.pt")
    parser.add_argument("--tracker_cfg", default="configs/bytetrack_relaxed_no_gmc.yaml")
    parser.add_argument("--conf", type=float, default=0.08)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--device", default="0")
    parser.add_argument("--timeout_sec", type=int, default=420)
    parser.add_argument("--reports_root", default="reports/before_after")
    parser.add_argument("--runs_root", default="data/interim/before_after_runs")
    parser.add_argument("--sample_frames", type=int, default=6)
    parser.add_argument("--annotated_every_n", type=int, default=10)
    parser.add_argument("--heavy_window_threshold", type=int, default=350)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    sequence_dir = (project_root / args.sequence_dir).resolve()
    if not sequence_dir.exists():
        raise FileNotFoundError(f"Sequence not found: {sequence_dir}")

    before_model = args.before_model
    after_model = args.after_model
    if not Path(after_model).is_absolute():
        after_model = str((project_root / after_model).resolve())

    reports_root = (project_root / args.reports_root).resolve()
    runs_root = (project_root / args.runs_root).resolve()
    reports_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    final_rows: list[dict] = []

    for window in args.windows:
        window_report_dir = reports_root / f"window_{window}"
        window_report_dir.mkdir(parents=True, exist_ok=True)

        unique_root = runs_root / run_tag / f"window_{window}"
        before_out = unique_root / "before"
        after_out = unique_root / "after"

        save_ann = True
        ann_every_n = args.annotated_every_n
        if window >= args.heavy_window_threshold:
            ann_every_n = max(20, args.annotated_every_n * 2)

        print(f"Running BEFORE window={window}")
        before_metrics = run_sprint(
            project_root=project_root,
            sequence_dir=sequence_dir,
            model_path=before_model,
            output_dir=before_out,
            window=window,
            tracker_cfg=args.tracker_cfg,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            timeout_sec=args.timeout_sec,
            save_annotated=save_ann,
            annotated_every_n=ann_every_n,
            device=args.device,
        )

        print(f"Running AFTER window={window}")
        after_metrics = run_sprint(
            project_root=project_root,
            sequence_dir=sequence_dir,
            model_path=after_model,
            output_dir=after_out,
            window=window,
            tracker_cfg=args.tracker_cfg,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            timeout_sec=args.timeout_sec,
            save_annotated=save_ann,
            annotated_every_n=ann_every_n,
            device=args.device,
        )

        write_metrics_files(window_report_dir, "before", before_metrics)
        write_metrics_files(window_report_dir, "after", after_metrics)

        sample_dir = window_report_dir / "sample_annotated_frames"
        copy_sample_frames(before_out, after_out, sample_dir, max_samples=args.sample_frames)

        delta = {
            k: (after_metrics[k] - before_metrics[k]) if isinstance(after_metrics[k], (int, float)) else "n/a"
            for k in FIELDS
        }

        summary_md = window_report_dir / "summary.md"
        summary_lines = [
            f"# Window {window} Before vs After",
            "",
            f"- before model: {before_metrics['model_path']}",
            f"- after model: {after_metrics['model_path']}",
            f"- before status: {before_metrics['status']}",
            f"- after status: {after_metrics['status']}",
            "",
            "| metric | before | after | delta (after-before) |",
            "| --- | ---: | ---: | ---: |",
        ]
        for k in FIELDS:
            b = before_metrics[k]
            a = after_metrics[k]
            d = delta[k]
            if isinstance(b, float):
                summary_lines.append(f"| {k} | {b:.4f} | {a:.4f} | {d:.4f} |")
            else:
                summary_lines.append(f"| {k} | {b} | {a} | {d} |")

        summary_lines.extend(
            [
                "",
                "sample frames:",
                f"- {sample_dir}",
            ]
        )
        summary_md.write_text("\n".join(summary_lines), encoding="utf-8")

        final_rows.append(
            {
                "window": window,
                "before_model": before_metrics["model_path"],
                "after_model": after_metrics["model_path"],
                "before_status": before_metrics["status"],
                "after_status": after_metrics["status"],
                "before_frames_processed": before_metrics["frames_processed"],
                "after_frames_processed": after_metrics["frames_processed"],
                "delta_frames_processed": delta["frames_processed"],
                "before_detections_total": before_metrics["detections_total"],
                "after_detections_total": after_metrics["detections_total"],
                "delta_detections_total": delta["detections_total"],
                "before_detections_avg_per_frame": before_metrics["detections_avg_per_frame"],
                "after_detections_avg_per_frame": after_metrics["detections_avg_per_frame"],
                "delta_detections_avg_per_frame": delta["detections_avg_per_frame"],
                "before_valid_track_ids": before_metrics["valid_track_ids"],
                "after_valid_track_ids": after_metrics["valid_track_ids"],
                "delta_valid_track_ids": delta["valid_track_ids"],
                "before_crops_saved": before_metrics["crops_saved"],
                "after_crops_saved": after_metrics["crops_saved"],
                "delta_crops_saved": delta["crops_saved"],
                "before_frames_with_detection_pct": before_metrics["frames_with_detection_pct"],
                "after_frames_with_detection_pct": after_metrics["frames_with_detection_pct"],
                "delta_frames_with_detection_pct": delta["frames_with_detection_pct"],
                "window_report_dir": str(window_report_dir),
            }
        )

    final_csv = reports_root / "final_comparative.csv"
    with final_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(final_rows[0].keys()))
        writer.writeheader()
        for r in final_rows:
            writer.writerow(r)

    overall_delta_det = sum(float(r["delta_detections_total"]) for r in final_rows)
    overall_delta_cov = sum(float(r["delta_frames_with_detection_pct"]) for r in final_rows)
    overall_delta_tracks = sum(float(r["delta_valid_track_ids"]) for r in final_rows)

    if overall_delta_det > 0 or overall_delta_cov > 0 or overall_delta_tracks > 0:
        conclusion = "Fine-tuning muestra mejora parcial y vale seguir iterando tracking/crops con mas datos y ajuste de detector."
    else:
        conclusion = "Fine-tuning baseline no mejora aun en esta secuencia; el siguiente paso es ampliar/mejorar subset etiquetado y ajustar hiperparametros del detector."

    final_md = reports_root / "final_comparative.md"
    md_lines = [
        "# Final Before vs After Comparative",
        "",
        "| window | frames before | frames after | det before | det after | delta det | avg/frame before | avg/frame after | delta avg/frame | tracks before | tracks after | delta tracks | crops before | crops after | delta crops | %frames det before | %frames det after | delta % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in final_rows:
        md_lines.append(
            md_table_row(
                int(r["window"]),
                {
                    "frames_processed": float(r["before_frames_processed"]),
                    "detections_total": float(r["before_detections_total"]),
                    "detections_avg_per_frame": float(r["before_detections_avg_per_frame"]),
                    "valid_track_ids": float(r["before_valid_track_ids"]),
                    "crops_saved": float(r["before_crops_saved"]),
                    "frames_with_detection_pct": float(r["before_frames_with_detection_pct"]),
                },
                {
                    "frames_processed": float(r["after_frames_processed"]),
                    "detections_total": float(r["after_detections_total"]),
                    "detections_avg_per_frame": float(r["after_detections_avg_per_frame"]),
                    "valid_track_ids": float(r["after_valid_track_ids"]),
                    "crops_saved": float(r["after_crops_saved"]),
                    "frames_with_detection_pct": float(r["after_frames_with_detection_pct"]),
                },
            )
        )

    md_lines.extend(
        [
            "",
            "## Conclusion",
            "",
            conclusion,
        ]
    )
    final_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved final comparative CSV: {final_csv}")
    print(f"Saved final comparative MD: {final_md}")


if __name__ == "__main__":
    main()
