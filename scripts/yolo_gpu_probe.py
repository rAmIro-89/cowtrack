from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Ultralytics GPU probe by repeated inference")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--out_json", default="reports/device_check/yolo_gpu_probe.json")
    args = parser.parse_args()

    t0 = time.time()
    model = YOLO(args.model)

    total_dets = 0
    for _ in range(args.iters):
        results = model.predict(
            source=args.image,
            classes=[19],
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        if results and results[0].boxes is not None:
            total_dets += len(results[0].boxes)

    elapsed = time.time() - t0
    out = {
        "pid": os.getpid(),
        "image": str(Path(args.image).resolve()),
        "model": args.model,
        "device": args.device,
        "iters": args.iters,
        "imgsz": args.imgsz,
        "total_detections": int(total_dets),
        "elapsed_sec": round(elapsed, 4),
        "avg_iter_sec": round(elapsed / max(1, args.iters), 6),
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
