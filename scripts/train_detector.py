from __future__ import annotations

import argparse

import torch
from ultralytics import YOLO


def resolve_device(device: str) -> str | int:
    if device == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if device.isdigit():
        return int(device)
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO detector (optional custom dataset)")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--data", required=True, help="YOLO data yaml path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="auto, cpu, or cuda device id (e.g. 0)")
    args = parser.parse_args()

    selected_device = resolve_device(args.device)

    if isinstance(selected_device, int) and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    print(f"Using device: {selected_device}")

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, device=selected_device)


if __name__ == "__main__":
    main()
