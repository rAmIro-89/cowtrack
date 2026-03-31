from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class CowDetection:
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int
    track_id: Optional[int] = None


class CowDetector:
    """Wrapper around Ultralytics YOLO for cow detection and tracking."""

    def __init__(self, model_path: str = "yolov8n.pt", class_id: int = 19, device: str = "auto") -> None:
        self.model = YOLO(model_path)
        self.class_id = class_id
        self.device = self._resolve_device(device)

    @staticmethod
    def _resolve_device(device: str) -> str | int:
        if device == "auto":
            return 0 if torch.cuda.is_available() else "cpu"
        if device.isdigit():
            return int(device)
        return device

    def predict(self, frame: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> List[CowDetection]:
        results = self.model.predict(
            frame,
            classes=[self.class_id],
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False,
        )
        return self._parse_results(results)

    def track(
        self,
        frame: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
        persist: bool = True,
        tracker: str | None = None,
        imgsz: int = 640,
    ) -> List[CowDetection]:
        kwargs = {
            "source": frame,
            "classes": [self.class_id],
            "conf": conf,
            "iou": iou,
            "persist": persist,
            "imgsz": imgsz,
            "device": self.device,
            "verbose": False,
        }
        if tracker:
            kwargs["tracker"] = tracker
        results = self.model.track(**kwargs)
        return self._parse_results(results)

    def _parse_results(self, results: list) -> List[CowDetection]:
        if not results:
            return []

        parsed: List[CowDetection] = []
        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None:
            return parsed

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
            score = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0
            cls = int(boxes.cls[i].item()) if boxes.cls is not None else self.class_id
            track_id = int(boxes.id[i].item()) if boxes.id is not None else None
            parsed.append(
                CowDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=score,
                    class_id=cls,
                    track_id=track_id,
                )
            )
        return parsed
