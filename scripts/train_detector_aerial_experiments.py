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
    parser = argparse.ArgumentParser(description="Run multiple YOLO aerial fine-tuning experiments")
    parser.add_argument("--data", default="configs/yolo_aerial_improved.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--reports_dir", default="reports")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_path = (project_root / args.data).resolve()
    reports_dir = (project_root / args.reports_dir).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    selected_device = resolve_device(args.device)

    if isinstance(selected_device, int) and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    print(f"Using device: {selected_device}")

    experiments = [
        {
            "name": "exp_n_img640_e20",
            "model": "yolov8n.pt",
            "epochs": 8,
            "imgsz": 640,
            "batch": 8,
            "workers": 0,
        },
        {
            "name": "exp_n_img960_e30",
            "model": "yolov8n.pt",
            "epochs": 12,
            "imgsz": 800,
            "batch": 6,
            "workers": 0,
        },
    ]

    rows = []
    for exp in experiments:
        print(f"Running {exp['name']}")
        model = YOLO(exp["model"])
        result = model.train(
            data=str(data_path),
            epochs=exp["epochs"],
            imgsz=exp["imgsz"],
            batch=exp["batch"],
            workers=exp["workers"],
            device=selected_device,
            project="outputs/detector_aerial_experiments",
            name=exp["name"],
            patience=20,
        )

        save_dir = Path(result.save_dir)
        best_pt = save_dir / "weights" / "best.pt"
        results_csv = save_dir / "results.csv"
        rows.append(
            {
                "experiment": exp["name"],
                "base_model": exp["model"],
                "epochs": exp["epochs"],
                "imgsz": exp["imgsz"],
                "batch": exp["batch"],
                "device": selected_device,
                "save_dir": str(save_dir),
                "best_pt": str(best_pt),
                "results_csv": str(results_csv),
            }
        )

    out_csv = reports_dir / "detector_aerial_experiments.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    out_md = reports_dir / "detector_aerial_experiments.md"
    lines = [
        "# Detector Aerial Experiments",
        "",
        "| experiment | base_model | epochs | imgsz | batch | device | best_pt |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['experiment']} | {r['base_model']} | {r['epochs']} | {r['imgsz']} | {r['batch']} | {r['device']} | {r['best_pt']} |"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved experiment summary: {out_csv}")


if __name__ == "__main__":
    main()
