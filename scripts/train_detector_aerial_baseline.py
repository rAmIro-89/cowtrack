from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Baseline fine-tuning for aerial cow detector with YOLO")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained YOLO weights")
    parser.add_argument("--data", default="configs/yolo_aerial_baseline.yaml", help="YOLO dataset yaml")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="auto, cpu, or cuda device id (e.g. 0)")
    parser.add_argument("--project", default="outputs/detector_aerial")
    parser.add_argument("--name", default="yolov8n_aerial_baseline")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    selected_device = resolve_device(args.device)

    if isinstance(selected_device, int) and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    print(f"Using device: {selected_device}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=selected_device,
        project=args.project,
        name=args.name,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
