from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from ultralytics import YOLO


def resolve_device(device: str) -> str | int:
    if device == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if device.isdigit():
        return int(device)
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description="Run small-object focused tiled detector experiments")
    parser.add_argument("--data", default="configs/yolo_aerial_tiled_smallobj.yaml")
    parser.add_argument("--reports_dir", default="reports/tiling_smallobj")
    parser.add_argument("--project", default="runs/detect/outputs/detector_tiled_smallobj")
    parser.add_argument("--device", default="0")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.4)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_path = (project_root / args.data).resolve()
    reports_dir = (project_root / args.reports_dir).resolve()
    project_dir = (project_root / args.project).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    project_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Data yaml not found: {data_path}")

    selected_device = resolve_device(args.device)
    if isinstance(selected_device, int) and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    print(f"Using device: {selected_device}")

    experiments = [
        {
            "name": "smallobj_yolov8n_img1024_e24",
            "base_model": "yolov8n.pt",
            "epochs": 24,
            "imgsz": 1024,
            "batch": 10,
            "patience": 14,
            "close_mosaic": 4,
        },
        {
            "name": "smallobj_yolov8s_img1280_e30",
            "base_model": "yolov8s.pt",
            "epochs": 30,
            "imgsz": 1280,
            "batch": 6,
            "patience": 18,
            "close_mosaic": 4,
        },
    ]

    rows = []
    for exp in experiments:
        print(f"Running {exp['name']}")
        model = YOLO(exp["base_model"])
        result = model.train(
            data=str(data_path),
            epochs=exp["epochs"],
            imgsz=exp["imgsz"],
            batch=exp["batch"],
            workers=0,
            device=selected_device,
            project=str(project_dir),
            name=exp["name"],
            patience=exp["patience"],
            close_mosaic=exp["close_mosaic"],
            cache=True,
        )

        save_dir = Path(result.save_dir)
        best_pt = save_dir / "weights" / "best.pt"
        results_csv = save_dir / "results.csv"

        rows.append(
            {
                "experiment": exp["name"],
                "dataset": str(data_path),
                "tile_size": args.tile_size,
                "overlap": args.overlap,
                "base_model": exp["base_model"],
                "epochs": exp["epochs"],
                "imgsz": exp["imgsz"],
                "batch": exp["batch"],
                "patience": exp["patience"],
                "close_mosaic": exp["close_mosaic"],
                "device": selected_device,
                "save_dir": str(save_dir),
                "best_pt": str(best_pt),
                "results_csv": str(results_csv),
            }
        )

    out_csv = reports_dir / "tiled_smallobj_experiments.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    out_md = reports_dir / "tiled_smallobj_experiments.md"
    lines = [
        "# Tiled Small-Object Detector Experiments",
        "",
        "| experiment | base_model | epochs | imgsz | batch | tile_size | overlap | close_mosaic | device | best_pt |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['experiment']} | {r['base_model']} | {r['epochs']} | {r['imgsz']} | {r['batch']} | {r['tile_size']} | {r['overlap']} | {r['close_mosaic']} | {r['device']} | {r['best_pt']} |"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved experiments summary: {out_csv}")


if __name__ == "__main__":
    main()
